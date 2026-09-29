# Experiment configurations

- `historical/`: original tracked YAML files, byte-preserved through `git mv`.
- `downstream/`: publication-safe AG News controlled protocol.
- `rank/`: controlled 1,000-step screening and independent 5,000-step protocol.
- `scaling/`: constant-effective-scale ablation protocol.

Private per-run `config.json` and generated launch YAML files remain beside raw
runs in ignored result directories. Public JSON files omit local absolute paths.
