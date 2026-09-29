# DoRA

- **Publication:** Liu et al., ICML 2024.
- **Core formulation:** decomposes a pretrained weight into magnitude and
  direction and applies a LoRA-style update to the directional component.
- **Architecture:** weight-space reparameterization; it does not introduce an
  activation-quadratic branch.
- **Parameters/rank:** includes a trainable magnitude vector plus low-rank
  directional factors; budget depends on the adapted weight shape and rank.
- **Scaling/initialization/insertion:** follows a LoRA-compatible selected-weight
  workflow; implementation details must be read from the original paper/code.
- **Backbones/tasks:** the paper reports LLaMA, LLaVA, and VL-BART experiments
  on language and vision-language downstream tasks.
- **Limitations for comparison:** DoRA was not implemented here. Its published
  scores are not comparable with the GPT-2 WikiText-2/AG News protocols.
- **Relationship:** architecturally related PEFT, mathematically different.
