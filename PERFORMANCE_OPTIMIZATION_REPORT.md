# Performance optimization report

## Protocol

All microbatch candidates used effective batch size 4 and the same short LoRA workload. Measurements exclude full campaign training. GPU utilization was sampled with `nvidia-smi` and is approximate.

| Microbatch | Accumulation | Effective batch | Optimizer steps/s | Tokens/s | Peak GPU MiB | Mean GPU util. | Wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 4 | 2.540 | 217.1 | 355.4 | 10.62087912087912 | 41.27 |
| 2 | 2 | 4 | 4.771 | 407.8 | 387.9 | 12.395833333333334 | 32.73 |
| 4 | 1 | 4 | 9.345 | 798.7 | 454.9 | 12.546762589928058 | 31.37 |

Selected candidate: **microbatch 4 / accumulation 1**, based on the highest finite optimizer throughput while fitting in memory. The historical configuration was microbatch 1 / accumulation 4.

## Independent-run concurrency

| Parallel runs | Total steps/s | Peak per-process GPU MiB | Mean GPU util. | Wall s |
|---:|---:|---:|---:|---:|
| 1 | 0.197 | 453.6 | 12.118959107806692 | 60.86 |
| 2 | 0.290 | 454.9 | 51.568306010928964 | 41.36 |

Observed N=2 whole-workload speedup: **1.47×**. Parallel execution remains opt-in; the default is 1.

## Reproducibility and workload estimate

A repeated microbatch-4 / accumulation-1 run with the same seed reproduced all four checked NLL summaries **bit for bit**. The unit test also verifies that batched loss equals the mean of the corresponding per-example losses. Different microbatch shapes are not expected to produce bit-identical trajectories because dropout and batched floating-point kernels consume randomness and operations differently; all candidates nevertheless preserve the declared effective batch size and objective weighting.

- Optimizer-throughput speedup, microbatch 4/1 versus historical 1/4: **3.68x**.
- Whole-workload N=2 concurrency speedup over N=1: **1.47x**.
- Existing principal cells retained: 27. New EWC cells remaining: 9 (67,500 optimizer steps).
- Historical-rate estimate for the nine remaining cells: approximately **6.3 GPU hours** serial.
- Benchmark-based estimate with microbatch 4/1 and two concurrent runs: approximately **1.2 hours** for optimizer work; allow roughly **1.3--2.0 hours** including evaluation, model loading, checkpoint I/O, and Fisher estimation.

The concurrency benchmark used two independent processes. Each reported about 454.9 MiB peak allocated VRAM, so the observed aggregate allocation remained well within the 4,096 MiB device.
