# sgl_kernel 集成方案（不依赖 Runtime）

目标：在 mini-sglang 中直接复用 sglang 的 C++/CUDA 核心（RadixAttention + flash/varlen attention + 分页 KV），而不是通过 sglang Runtime。保持现有入口（engine/bench_suite），让同一 engine 既可走纯 torch 路径（开发/CPU/MPS）又可在 GPU 上切换到 sgl_kernel。

## 组件划分
- **注意力后端抽象**：新增 `SglKernelAttentionBackend`，接口 `prefill(qkv, cache_state, positions)` / `decode(last_token, cache_state)`，内部根据可用性选择 `sgl_kernel.flash_attn.flash_attn_with_kvcache` 或 torch SDPA 回退。对外暴露 `available()`，便于在 Mac/MPS 退回。
- **KV 布局**：使用分页 KV（page_size 默认 512），维护：
  - `k_cache`, `v_cache`：形状 `[num_pages, page_size, num_heads, head_dim]`。
  - `page_table`：每个 request 的页索引列表，用于 flash_attn_with_kvcache 的 `page_table`/`cache_seqlens`。
  - `req_to_token`：映射 request -> token 索引（用于 torch 回退）。
  - 兼容当前 prefix/prefill cache：prefill 时写入 KV，decode 直接追加。
- **位置与 RoPE**：沿用 HF 权重，仍用 transformers 的 RoPE 实现；SglKernel 后端只消费旋转后的 q/k/v。
- **模型侧**：保留 HF 模型加载，但暴露 Q/K/V 投影结果给注意力后端：
  - 在 `ModelBackend` 内对 decoder block 做 monkey-patch：替换自注意力 forward，使其输出 logits 同时把 `qkv` 送入 `SglKernelAttentionBackend`，不再调用 HF 自带的注意力。HF MLP/Norm/Head 复用。
  - 后端切换由 env 控制：`ATTN_BACKEND=sgl_kernel|torch`，默认 torch。
- **回退策略**：若未安装 `sgl_kernel` 或非 CUDA 环境，自动回退 torch SDPA；日志提示可用的后端。

## Qwen3 解码链路对齐（sglang → mini-sglang 实现清单）
- **堆叠顺序**：参考 `python/sglang/srt/models/qwen3.py` + `qwen2.py`：嵌入 → 每层（自注意力 → MLP → 残差/Norm）→ 末尾 RMSNorm → 词表线性。
- **嵌入/输出头**：`VocabParallelEmbedding` / `ParallelLMHead` (`layers/vocab_parallel_embedding.py`)，`LogitsProcessor` (`layers/logits_processor.py`)；单卡可先用 HF Embedding/LMHead，接口预留并行/权重共享。
- **归一化/残差**：`RMSNorm` + 可选 fused add (`layers/layernorm.py`)，`LayerCommunicator` 仅分布式需要；单卡保留 RMSNorm + 残差。
- **自注意力算子**：`QKVParallelLinear` + `RowParallelLinear` (`layers/linear.py`)，RoPE (`layers/rotary_embedding.py`)，核心 `RadixAttention` (`layers/radix_attention.py`) 调用选定 backend。mini-sglang 需对齐 q/k/v/o 投影接口、head_dim/num_heads/num_kv_heads，RoPE 旋转，注意力后端可插拔。
- **注意力后端/Flash**：后端注册在 `layers/attention/attention_registry.py`；实现如 `flashattention_backend.py`、`flashinfer_backend.py`、`triton_backend.py`、`torch_native_backend.py`，底层核在 `sgl-kernel/csrc`。mini-sglang 提供后端选择，默认 torch SDPA，CUDA 时首选 `sgl_kernel.flash_attn_with_kvcache`（分页 KV 接口对齐）。
- **MLP**：`Qwen2MLP` 使用 `MergedColumnParallelLinear + RowParallelLinear` (`layers/linear.py`) + `SiluAndMul` (`layers/activation.py`)；单卡可用普通线性 + SiLU*门控，接口预留 tp_size。
- **KV cache/调度**：前向元数据在 `model_executor/forward_batch_info.py`，分配在 `mem_cache/memory_pool.py`。mini-sglang 用分页 KV（page_table/cache_seqlens/req_to_token），prefill/decode 兼容 prefix/prefill cache，engine 设定 forward 模式。
- **集成策略**：加载 HF Qwen3 权重，替换自注意力调用为自定义 AttentionBackend（保持 q_proj/k_proj/v_proj/o_proj 权重）；保留 HF MLP/Norm/Head 或按需替换。提供开发路径（torch 回退）与 GPU 路径（sgl_kernel + 分页 KV），日志打印当前后端/维度。
- **验证**：CPU/MPS 用 torch 路径对齐 HF generate；GPU 启用 flash 内核后跑 `bench_suite`/`local_bench`，记录 TTFB/吞吐，按需调 page_size/block 配置。

## 流程
1) **初始化**：`ModelBackend` 读取 env，尝试导入 `sgl_kernel.flash_attn`; 构建 `SglKernelAttentionBackend` 与 KV 管理器。
2) **prefill**：对 prompt 生成 qkv -> RoPE -> `SglKernelAttentionBackend.prefill`，写 KV，得到第一 token logits。输出/缓存保持兼容 engine。
3) **decode**：单步 decode 时仅计算上一 token 的 qkv，调用 `SglKernelAttentionBackend.decode`（读取 KV page table），得到下 token。
4) **并发/分页**：page 管理与 cache 命中逻辑复用现有 prefix/prefill cache；超出 token 预算触发 LRU/LFU 驱逐。
5) **可选优化**：GPU 环境可切换 `flashattention_backend` / `flashinfer_backend`，通过 env 或构造参数。

## 测试与验证
- **正确性**：小模型（CPU/MPS）下启用 torch 回退，比较原路径与新路径生成结果是否一致；构造 prefix 命中与 cache 驱逐场景。
- **性能**（云端 GPU）：启用 `ATTN_BACKEND=sgl_kernel`，运行 `bench_suite.py`/`local_bench.py`，记录 TTFB/吞吐，对比 HF TextIteratorStreamer。

## 已知风险 / 待补工作（需 GPU 环境）
- 需要 `sgl_kernel` 对应 CUDA/ROCm wheel；Mac/MPS 无法验证性能。
- flash_attn_with_kvcache 的参数签名需与 KV 布局匹配；在 GPU 环境调整 page_table/seq_lens 细节。
- HF 各模型的自注意力实现可能不同，monkey-patch 需按实际结构适配（Qwen/Llama 等）。
