#!/usr/bin/env python3
"""Build the definitive reviewer-level submission manuscript."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "FINAL_PAPER_SUBMISSION.md"
TEX = ROOT / "FINAL_PAPER_SUBMISSION.tex"

TITLE = (
    "Stability–Plasticity Control in Parameter-Efficient Continual Factual Learning: "
    "EWC across Linear, Quadratic, and Combined Low-Rank Adapters"
)
AUTHOR = "Federico Lolli"
SUBJECT = "Parameter-efficient continual factual learning with Elastic Weight Consolidation"
KEYWORDS = "continual learning, EWC, PEFT, LoRA, quadratic adapters, factual learning, GPT-2"

FIGURE_CAPTIONS = {
    "figure_1_exploratory_stability_plasticity": "Exploratory single-seed stability–plasticity trajectories across seven EWC strengths. Lower-left is better on both axes.",
    "figure_2_confirmation_stability_plasticity": "Candidate evaluation over seeds 42, 123, and 456. Points show descriptive means and bars show sample standard deviations; seed 42 was used for selection.",
    "figure_absolute_retention_nll": "Absolute NLL trajectory for factual set A after phases A, B, and C. Curves show three-seed means and sample standard deviations. Lower NLL is better; forgetting values are differences between these absolute losses.",
    "figure_3_final_balanced_memory": "Final balanced memory after phase C. Bars show mean NLL across sets A, B, and C; black points are individual seeds.",
    "figure_4_forgetting_decomposition": "Forgetting decomposed into A after B, A after C, and B after C. Points show three-seed means with sample standard deviations.",
    "figure_5_plasticity_exact_generation": "New-fact acquisition NLL and final exact-token-match generation for the repeated candidate configurations.",
    "figure_6_mechanistic_evidence": "Exploratory relationships among EWC strength, cumulative adapter displacement, realized phase-C penalty, and online Fisher effective support.",
}

TABLE_METADATA = [
    ("Matched adapter configurations.", "tab:adapters"),
    ("Unregularized exploratory stability–plasticity coordinates.", "tab:exploratory-zero"),
    ("Regularization coefficients selected on seed 42 for subsequent replication.", "tab:replication-grid"),
    ("Three-seed descriptive aggregates, reported as mean $\\pm$ sample standard deviation.", "tab:confirmation"),
    ("Absolute set-B NLL after acquisition and after phase C, reported as mean $\\pm$ sample standard deviation. Lower is better.", "tab:b-retention"),
    ("Association between cumulative adapter displacement and mean forgetting in the exploratory sweep.", "tab:disp-forgetting"),
    ("Association between EWC strength and cumulative adapter displacement. The log transform is a scale-sensitivity check.", "tab:lambda-disp"),
    ("Association between EWC strength and mean forgetting. The log transform is a scale-sensitivity check.", "tab:lambda-forgetting"),
    ("Association between EWC strength and acquisition NLL. The log transform is a scale-sensitivity check.", "tab:lambda-acquisition"),
    ("Association between realized phase-C EWC penalty and mean forgetting.", "tab:penalty-forgetting"),
    ("Association between realized phase-C EWC penalty and acquisition NLL.", "tab:penalty-acquisition"),
]


def load_base():
    path = ROOT / "tools" / "build_final_paper.py"
    spec = importlib.util.spec_from_file_location("base_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_abstract(markdown: str) -> str:
    match = re.search(r"^## Abstract\n\n(.+?)\n\n\*\*Keywords:", markdown, re.M | re.S)
    if not match:
        raise RuntimeError("Abstract not found")
    return match.group(1).strip()


def main() -> None:
    markdown = SOURCE.read_text()
    base = load_base()
    base.TITLE = TITLE
    base.AUTHOR = AUTHOR
    base.ABSTRACT = extract_abstract(markdown)
    base.FIGURE_CAPTIONS = FIGURE_CAPTIONS

    table_index = 0

    def table_latex(lines: list[str]) -> str:
        nonlocal table_index
        cells = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
        header, body = cells[0], cells[2:]
        caption, label = TABLE_METADATA[table_index]
        table_index += 1
        spec = "".join([r">{\raggedright\arraybackslash}X"] * len(header))
        out = [
            r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
            rf"\caption{{{caption}}}", rf"\label{{{label}}}",
            r"\begin{adjustbox}{max width=\textwidth}",
            rf"\begin{{tabularx}}{{\textwidth}}{{{spec}}}", r"\toprule",
            " & ".join(base.inline(x) for x in header) + r" \\", r"\midrule",
        ]
        out += [" & ".join(base.inline(x) for x in row) + r" \\" for row in body]
        out += [r"\bottomrule", r"\end{tabularx}", r"\end{adjustbox}", r"\end{table}"]
        return "\n".join(out)

    base.table_latex = table_latex
    body = base.markdown_body_to_latex(markdown)
    body = body.replace(r"\texttt{cc505ec681369c5f9aa4c7123cfff11da878bce17c0669890d416b5eb07f1923}", r"\texttt{cc505ec6\allowbreak 81369c5f\allowbreak 9aa4c712\allowbreak 3cfff11d\allowbreak a878bce1\allowbreak 7c066989\allowbreak 0d416b5e\allowbreak b07f1923}")
    if table_index != len(TABLE_METADATA):
        raise RuntimeError(f"Expected {len(TABLE_METADATA)} tables, rendered {table_index}")

    preamble = rf'''\documentclass[12pt,a4paper]{{article}}
\usepackage[a4paper,margin=3.0cm]{{geometry}}
\usepackage{{fontspec}}
\usepackage{{amsmath,amssymb,mathtools}}
\usepackage{{graphicx,booktabs,tabularx,array,adjustbox}}
\usepackage[section]{{placeins}}
\usepackage{{microtype}}
\usepackage{{enumitem}}
\usepackage{{caption}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{bookmark}}
\setmainfont{{DejaVu Serif}}
\setsansfont{{DejaVu Sans}}
\setmonofont{{DejaVu Sans Mono}}
\linespread{{1.35}}
\setlist{{nosep,leftmargin=*}}
\emergencystretch=3em
\captionsetup{{font=small,labelfont=bf}}
\hypersetup{{
  pdftitle={{{TITLE}}},
  pdfauthor={{{AUTHOR}}},
  pdfsubject={{{SUBJECT}}},
  pdfkeywords={{{KEYWORDS}}}
}}
\title{{\textbf{{{TITLE}}}}}
\author{{{AUTHOR}}}
\date{{October 2026}}
\begin{{document}}
'''
    TEX.write_text(preamble + body + "\n" + base.BIBLIOGRAPHY + "\n\\end{document}\n")
    print(TEX)


if __name__ == "__main__":
    main()
