# Frozen protocol: controlled 5,000-step rank confirmation

This protocol was written before inspecting any confirmation result. It confirms,
without mixing data, the separate 1,000-step rank screening.

| Field | Fixed value |
|---|---|
| Task | WikiText-2 causal language modeling |
| Backbone | frozen GPT-2 (`gpt2`) |
| Placement | `transformer.h[0].attn.c_proj`, zero-based block 0 only |
| Methods | LoRA; Symmetric Quadratic |
| Local ranks | 1, 2, 4, 8 |
| Seeds | 42, 123, 456 |
| Training | 5,000 fresh optimizer steps |
| Sequence length | 128 |
| Gradient accumulation | 4 |
| Optimizer | AdamW, learning rate 3e-4 |
| Alpha | 4 |
| Validation | same fresh eight-block validation protocol as the screening, every 25 steps plus final step |
| Test | WikiText-2 public test, final step-5,000 adapter checkpoint only |

`best_recorded_validation_loss` is the minimum fresh validation measurement
within a run. `final_validation_loss` is the step-5,000 fresh validation. The
test is never used to select an adapter or a checkpoint. A rank changes both
parameter count and `alpha/r`; this primary confirmation studies the configured
rank behavior and does not isolate rank from scaling.
