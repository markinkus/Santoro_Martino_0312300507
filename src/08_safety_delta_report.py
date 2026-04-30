import argparse
from pathlib import Path

from utils_ops import carica_json, garantisci_cartella_padre, salva_json


def _section_metrics(payload: dict, section: str) -> dict:
    if section == "test_split":
        sec = payload.get("test_split", {})
    else:
        sec = payload.get("benchmarks", {}).get(section, {})

    dep = sec.get("department", {})
    sent = sec.get("sentiment", {})

    dep_rec = dep.get("recall_by_class", {}) if isinstance(dep, dict) else {}
    sent_rec = sent.get("recall_by_class", {}) if isinstance(sent, dict) else {}

    return {
        "department_f1": float(dep.get("f1_macro", 0.0)),
        "sentiment_f1": float(sent.get("f1_macro", 0.0)),
        "sentiment_recall_neg": float(sent_rec.get("neg", 0.0)),
        "sentiment_recall_pos": float(sent_rec.get("pos", 0.0)),
        "coverage": sec.get("coverage"),
        "needs_review_rate": sec.get("needs_review_rate"),
    }


def _delta(a: dict, b: dict) -> dict:
    out = {}
    for k in sorted(set(a.keys()) | set(b.keys())):
        va = a.get(k)
        vb = b.get(k)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            out[k] = float(vb - va)
        else:
            out[k] = None
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=str, required=True, help="metrics.json della run base/prima")
    parser.add_argument("--after", type=str, required=True, help="metrics.json run migliorato/dopo")
    parser.add_argument("--safety_benchmark", type=str, default="reviews_safety_critical.csv")
    parser.add_argument("--out_json", type=str, default="outputs/safety_delta_report.json")
    parser.add_argument("--out_md", type=str, default="outputs/safety_delta_report.md")
    args = parser.parse_args()

    before = carica_json(args.before, predefinito={})
    after = carica_json(args.after, predefinito={})

    sections = [
        "test_split",
        "reviews_in_distribution.csv",
        "reviews_ambiguous.csv",
        "reviews_noisy.csv",
        "reviews_colloquial.csv",
        args.safety_benchmark,
    ]

    report = {
        "before": str(args.before),
        "after": str(args.after),
        "sections": {},
    }

    for sec in sections:
        m_before = _section_metrics(before, sec)
        m_after = _section_metrics(after, sec)
        report["sections"][sec] = {
            "before": m_before,
            "after": m_after,
            "delta_after_minus_before": _delta(m_before, m_after),
        }

    salva_json(report, args.out_json)

    lines = [
        "# Report delta sicurezza",
        "",
        f"- prima: `{args.before}`",
        f"- dopo: `{args.after}`",
        "",
    ]

    for sec in sections:
        row = report["sections"][sec]
        d = row["delta_after_minus_before"]
        lines.append(f"## {sec}")
        lines.append(
            "- delta F1 reparto: "
            f"{d.get('department_f1', 0.0):+.4f} | "
            "delta F1 sentiment: "
            f"{d.get('sentiment_f1', 0.0):+.4f}"
        )
        lines.append(
            "- delta richiamo sentiment negativo: "
            f"{d.get('sentiment_recall_neg', 0.0):+.4f} | "
            "delta richiamo sentiment positivo: "
            f"{d.get('sentiment_recall_pos', 0.0):+.4f}"
        )
        cov = d.get("coverage")
        nrr = d.get("needs_review_rate")
        if cov is not None and nrr is not None:
            lines.append(f"- delta copertura: {cov:+.4f} | delta tasso controllo umano: {nrr:+.4f}")
        lines.append("")

    garantisci_cartella_padre(args.out_md)
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Report JSON: {args.out_json}")
    print(f"[OK] Report Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
