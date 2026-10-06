"""Corre el experimento comparativo y guarda un reporte en markdown.

Uso:  python scripts/run_evaluation.py [--unseen] [--alpha 0.05] [--delta 0.05]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from priomed_classification.evaluation import run_experiment, to_markdown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args()

    args.out.mkdir(exist_ok=True)
    sections = []
    for unseen in (False, True):
        metrics, info = run_experiment(alpha=args.alpha, delta=args.delta, seed=args.seed, unseen_paraphrases=unseen)
        title = "Prueba con paráfrasis vistas" if not unseen else "Prueba de estrés: paráfrasis NO vistas"
        sections.append(f"## {title}\n\n{to_markdown(metrics, info)}")
    report = "# Reporte de evaluación (datos sintéticos)\n\n" + "\n".join(sections)
    path = args.out / "evaluation_report.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nGuardado en {path}")


if __name__ == "__main__":
    main()
