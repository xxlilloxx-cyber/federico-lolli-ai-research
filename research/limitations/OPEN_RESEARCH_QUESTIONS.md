# Open research questions

The AG News downstream study, controlled rank studies, spectral/Jacobian
analysis, adapter-only Hessian analysis, and focused cost benchmark have been
completed. The following gaps remain:

1. replicate the constant-effective-scale ablation beyond seed 42;
2. repeat the controlled rank design on another language-modeling dataset and
   a second Transformer family;
3. predeclare and measure full-block/model interaction responses rather than
   relying only on adapter-local derivatives;
4. run long controlled MLP and attention-plus-MLP placement studies;
5. test downstream tasks beyond AG News and longer contexts;
6. broaden optimizer, learning-rate, normalization, and alpha schedules;
7. benchmark inference and training cost across devices, batches, sequence
   lengths, ranks, and insertion counts.

These are future experiments, not missing values in current tables.
