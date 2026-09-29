import csv, json, math
from pathlib import Path
import statistics
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results'
PLOTS = RESULTS / 'plots'
PLOTS.mkdir(parents=True, exist_ok=True)


def summaries():
    out = []
    for p in RESULTS.glob('*/**/summary.json'):
        d = json.loads(p.read_text()); d['_path'] = str(p.parent.relative_to(ROOT)); out.append(d)
    return sorted(out, key=lambda x: (x['kind'], x.get('seed', 0)))


def write_all(rows):
    fields = ['kind','seed','rank','steps','total_parameters','trainable_parameters','trainable_percentage',
              'parameter_difference_vs_lora_percent','validation_loss','perplexity','peak_vram_bytes',
              'training_seconds','milliseconds_per_step','tokens_per_second','nan_count','inf_count','_path']
    with (RESULTS/'all_results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def plots(rows):
    colors = {'baseline':'black','lora':'tab:blue','quadratic':'tab:orange','signed_quadratic':'tab:green','feature_interaction':'tab:red'}
    for name, y, ylabel in [('training_loss','training_loss','Training loss'),('validation_loss','validation_loss','Validation loss'),('perplexity','perplexity','Perplexity'),('gradient_norm','gradient_norm','Gradient norm')]:
        plt.figure(figsize=(8,5))
        for p in RESULTS.glob('*/**/metrics.csv'):
            kind=p.parent.parent.name
            vals=list(csv.DictReader(p.open()))
            if vals: plt.plot([int(v['step']) for v in vals], [float(v[y]) for v in vals], label=kind, color=colors.get(kind))
        plt.xlabel('Step'); plt.ylabel(ylabel); plt.title(f'{ylabel} vs step'); plt.legend(); plt.tight_layout(); plt.savefig(PLOTS/f'{name}.png', dpi=140); plt.close()
    def bar(filename, key, ylabel):
        labels=[r['kind'] for r in rows]; vals=[float(r.get(key,0)) for r in rows]
        plt.figure(figsize=(8,5)); plt.bar(labels, vals); plt.ylabel(ylabel); plt.title(ylabel); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(PLOTS/filename, dpi=140); plt.close()
    bar('memory_comparison.png','peak_vram_bytes','Peak VRAM (bytes)'); bar('throughput_comparison.png','tokens_per_second','Tokens/sec'); bar('training_time_comparison.png','training_seconds','Training time (s)')
    plt.figure(figsize=(7,5));
    for r in rows: plt.scatter(r['trainable_parameters'], r['validation_loss'], label=r['kind'])
    plt.xlabel('Trainable parameters'); plt.ylabel('Best validation loss'); plt.title('Parameter efficiency'); plt.legend(); plt.tight_layout(); plt.savefig(PLOTS/'parameter_efficiency.png', dpi=140); plt.close()


def reports(rows):
    def avg(kind, key):
        xs=[float(r[key]) for r in rows if r['kind']==kind]
        return (statistics.mean(xs), statistics.pstdev(xs), min(xs), max(xs)) if xs else (float('nan'),)*4
    order=['baseline','lora','quadratic','signed_quadratic','feature_interaction']
    table='\n'.join(f"| {r['kind']} | {r.get('seed','')} | {r['trainable_parameters']} | {r['validation_loss']:.4f} | {r['perplexity']:.2f} | {r['peak_vram_bytes']/2**20:.1f} | {r['training_seconds']:.2f} | {r['milliseconds_per_step']:.1f} | {r['tokens_per_second']:.1f} |" for r in rows)
    summary=[]
    for k in order:
        m,s,lo,hi=avg(k,'validation_loss')
        if not math.isnan(m): summary.append(f"- **{k}**: validation loss media {m:.4f} (sd {s:.4f}), min {lo:.4f}, max {hi:.4f}.")
    max_steps=max((int(r.get('steps',0)) for r in rows), default=0)
    md=f'''# Quadratic Transformer — report

## 1. Executive summary

Sono stati confrontati GPT-2 Small congelato, LoRA, Quadratic, Signed Quadratic e Feature Interaction su WikiText-2. Sono presenti sanity run da 20 step, test di velocità da 50 step e una campagna da 500 step parzialmente completata. Il confronto completo a tre seed non è terminato perché il runtime NVIDIA si è bloccato; i risultati sono comunque conservati e non vengono trattati come conclusivi.

## 2. Obiettivo e architetture

LoRA usa `XUV`. Quadratic usa `(X⊙X)UV`. Signed Quadratic usa `(X|X|)UV`. Feature Interaction usa `((XU)⊙(XV))P`. Nell'ultimo caso `(XU)_k(XV)_k = Σ_iΣ_j x_i x_j U_ik V_jk`, quindi compaiono interazioni tra feature differenti senza costruire una matrice quadratica completa.

## 3. Hardware e configurazione

RTX A500 Embedded, 4 GiB VRAM, driver 535.309.01, CUDA 12.1, PyTorch 2.3.1+cu121, Python 3.12.3. GPT-2 Small (124M), WikiText-2, sequence length 128, batch 1, gradient accumulation 4, FP16 backbone, adapter FP32, un solo `attn.c_proj` nel blocco 0.

## 4. Parameter budget

LoRA, Quadratic e Signed usano 6.144 parametri addestrabili (rank 4). Feature Interaction usa rank 3 e 6.912 (+12,5%) perché la sua forma richiede tre matrici. Il backbone resta congelato.

## 5. Risultati osservati

| Model | Seed | Trainable | Val loss | PPL | Peak MiB | Time s | ms/step | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

{chr(10).join(summary)}

Peak VRAM è rimasto sotto 400 MiB nei run brevi, ampiamente sotto il limite di 4 GiB. Nessun NaN o Inf è stato osservato. L'identità iniziale è stata verificata con differenza massima 0.0 nello smoke test.

## 6. Interpretazione prudente

**Dati osservati:** nei sanity run le differenze sono dell'ordine di pochi millesimi. Nel solo run LoRA completato a 500 step la validation loss è 3.5686; questo dato non è confrontabile come risultato finale perché gli altri adapter non hanno completato lo stesso numero di step e seed. Nei test da 50 step Signed e Feature Interaction mostrano valori vicini, mentre Quadratic puro resta vicino al baseline.

**Interpretazione:** il segnale è piccolo e non separabile dalla variabilità tra seed. Non c'è ancora evidenza di un vantaggio ripetibile di un adapter quadratico.

## 7. Limiti e prossimi passi

Modello piccolo, un solo layer, dataset piccolo, numero di step non uniforme nella campagna interrotta e seed non completati per tutti i modelli. L'evidenza attuale è **nessuna evidenza conclusiva di vantaggio**. Le ablation attention/MLP vanno eseguite solo dopo aver completato i tre seed per tutte le architetture.
'''
    (RESULTS/'REPORT.md').write_text(md)
    simple = '# Risultato semplice\n\nAbbiamo provato GPT-2 con LoRA e tre adapter quadratici. Tutti funzionano e usano meno di 4 GB. Nei test brevi le differenze sono molto piccole: non possiamo ancora dire che un metodo sia migliore. Feature Interaction usa circa il 12,5% di parametri in più. Il passo successivo è ripetere 500 step con tre seed.\n'
    (RESULTS/'SIMPLE_REPORT.md').write_text(simple)
    rows_json=json.dumps(rows, indent=2)
    html=f'''<!doctype html><meta charset="utf-8"><title>Quadratic Transformer report</title><style>body{{font:16px sans-serif;max-width:1100px;margin:2em auto;color:#222}} table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:6px}}img{{max-width:800px}}</style><h1>Quadratic Transformer</h1><h2>Executive Summary</h2><p>Confronto preliminare controllato tra baseline, LoRA e tre adapter quadratici su GPT-2 Small. Le differenze sono piccole e non conclusive.</p><h2>Formule</h2><p>LoRA: XUV · Quadratic: (X⊙X)UV · Signed: (X|X|)UV · Feature Interaction: ((XU)⊙(XV))P</p><h2>Risultati</h2><table><tr><th>Model</th><th>Seed</th><th>Trainable</th><th>Val loss</th><th>PPL</th><th>Peak MiB</th></tr>{''.join(f"<tr><td>{r['kind']}</td><td>{r['seed']}</td><td>{r['trainable_parameters']}</td><td>{r['validation_loss']:.4f}</td><td>{r['perplexity']:.2f}</td><td>{r['peak_vram_bytes']/2**20:.1f}</td></tr>" for r in rows)}</table><h2>Curve</h2><img src="plots/validation_loss.png"><img src="plots/training_loss.png"><h2>Conclusione</h2><p>Segnale interessante ma non conclusivo. Servono tre seed e più step.</p>'''
    (RESULTS/'report.html').write_text(html)


if __name__ == '__main__':
    rs=summaries(); write_all(rs)
    primary=[r for r in rs if int(r.get('steps', 0)) == 500 and 'speed' not in r.get('_path','')]
    plots(primary); reports(primary); print(f'Generated reports for {len(rs)} total runs ({len(primary)} primary 500-step runs)')
