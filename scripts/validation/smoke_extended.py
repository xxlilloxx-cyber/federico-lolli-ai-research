import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, 'src')
from adapters import freeze_and_insert, parameter_counts


def run(kind, **kwargs):
    tok = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float16).cuda()
    model.config.use_cache = False
    x = tok('A controlled adapter identity and gradient test.', return_tensors='pt').input_ids.cuda()
    with torch.no_grad(): original = model(x).logits.float()
    freeze_and_insert(model, kind, layers=(0,), projections=('attn.c_proj',), diagnostics=True, **kwargs)
    with torch.no_grad(): modified = model(x).logits.float()
    error = (original - modified).abs().max().item()
    loss = model(x, labels=x).loss
    loss.backward()
    module = model.transformer.h[0].attn.c_proj
    trainable_grads = {n: (p.grad is not None and bool(torch.isfinite(p.grad).all())) for n, p in module.named_parameters() if p.requires_grad}
    frozen_ok = all(p.grad is None for p in model.parameters() if not p.requires_grad)
    return {'kind': kind, 'identity_max_abs_error': error, 'loss': loss.item(),
            'frozen_backbone_no_grad': frozen_ok, 'trainable_grads_finite': trainable_grads,
            'peak_vram_bytes': torch.cuda.max_memory_allocated(), **parameter_counts(model),
            'adapter_stats': module.last_stats}


if __name__ == '__main__':
    torch.manual_seed(42)
    results = [run('symmetric_quadratic', rank=4, alpha=4.0),
               run('linear_quadratic', rank_l=1, rank_q=2, alpha_l=4.0, alpha_q=4.0)]
    print(json.dumps(results, indent=2))
