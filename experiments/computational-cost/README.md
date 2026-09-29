# Computational-cost benchmark

The existing benchmark compares rank-4, seed-42 LoRA and Symmetric final
checkpoints on identical hardware with batch size 1 and sequence length 128.
It uses ten warm-up iterations and thirty synchronized repetitions. It records
forward latency, forward-plus-backward latency, token throughput, and peak
allocated CUDA memory.

- **Private checkpoints:** controlled 5,000-step confirmation tree.
- **Canonical data:**
  [`results/tables/computational-cost/computational_cost.csv`](../../results/tables/computational-cost/computational_cost.csv).
- **Script:** `scripts/analysis/benchmark_adapter_cost.py`.
- **Limitation:** one GPU, rank, seed, batch size, and sequence length.
