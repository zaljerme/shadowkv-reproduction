\# Findings log



\## Phase 3: KV cache scaling (CPU, one Llama-3-8B shaped layer, fp32)

\- Cache size matches formula exactly: 8 MB per 1K tokens per layer.

\- Decode latency roughly linear in context above 4K: 25 ms at 1K, 413 ms at 32K.

\- Caveat: naive cache uses torch.cat and repeat\_interleave, so latency overstates pure attention cost.



\## Phase 4-5: key rank, pre vs post RoPE (4K tokens, 2 WikiText prompts)

\- Pre-RoPE keys are more compressible than post-RoPE on both Qwen2.5-0.5B and TinyLlama v1.1.

&#x20; Rank 64 relative error: Qwen 0.063 pre vs 0.149 post; TinyLlama 0.107 pre vs 0.255 post.

\- Qwen layers 0, 1, 2, 8 look rank 1 because the k\_proj bias dominates the keys

&#x20; (key norm roughly equals bias norm). Centered, they need 89 to 95 of 128 dims.

\- TinyLlama (no bias) still has a large shared mean direction: raw rank\_99 about 55 to 97 of 256,

&#x20; centered about 135 to 170.

\- On TinyLlama, after removing the mean, pre and post RoPE error are nearly equal

&#x20; (0.255 vs 0.276 at rank 64). Observation: much of RoPE's damage to low rank comes from

&#x20; rotating the shared mean direction. On Qwen, RoPE hurts beyond the mean (0.150 vs 0.237).

\- Layer variation: TinyLlama layers 0 and 1 need 11 and 19 dims pre-RoPE, middle layers 70 to 97.

\- Open question: energy-based error may not reflect attention error. Test in Phase 7.

