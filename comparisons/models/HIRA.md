# HiRA

- **Publication:** Huang et al., ICLR 2025.
- **Core mechanism:** uses a Hadamard product involving the frozen pretrained
  weight and a learned low-rank construction to retain high-rank update
  capacity with a small trainable budget.
- **Nature:** multiplicative structure in weight space; the resulting adapted
  layer remains linear in the current activation.
- **Backbones/tasks:** Llama-2-7B and Llama-3-8B with commonsense reasoning,
  ConvAI2 dialogue, and mathematical reasoning in the original study.
- **Methodology:** the paper includes rank, placement, ablation, and efficiency
  analyses.
- **Relationship:** both methods use elementwise multiplication, but on
  different objects. HiRA modulates weights; this study squares projected
  activations.
- **Status here:** not implemented; published values are context only.
