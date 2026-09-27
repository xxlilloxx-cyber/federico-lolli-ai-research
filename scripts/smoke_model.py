import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, 'src')
from adapters import freeze_and_insert, parameter_counts


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float16).cuda()
    model.config.use_cache = False
    x = tokenizer('A small controlled test of the quadratic adapter.', return_tensors='pt').input_ids.cuda()
    with torch.no_grad():
        original = model(x).logits.float()
    freeze_and_insert(model, 'quadratic', 4, 4.0, (0,), ('attn.c_proj',), True)
    with torch.no_grad():
        modified = model(x).logits.float()
    error = (original - modified).abs().max().item()
    assert error < 0.01, error
    loss = model(x, labels=x).loss
    loss.backward()
    adapter = model.transformer.h[0].attn.c_proj
    assert adapter.up.grad is not None and torch.isfinite(adapter.up.grad).all()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    print(json.dumps({'model': 'gpt2', 'identity_max_abs_error': error,
                      'loss': loss.item(), 'peak_vram_bytes': torch.cuda.max_memory_allocated(),
                      **parameter_counts(model), 'diagnostics': adapter.last_stats}, indent=2))


if __name__ == '__main__':
    main()
