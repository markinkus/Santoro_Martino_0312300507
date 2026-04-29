import argparse
from datetime import datetime

import joblib
import pandas as pd

from utils_ops import (
    DEFAULT_THRESHOLDS,
    applica_campi_operativi,
    applica_guardrail_sentiment_df,
    applica_reparti_impattati_df,
    carica_json,
    contributi_principali_token,
    garantisci_cartella_padre,
    normalizza_mappa_soglie,
    salva_json,
    simula_sla,
)
from utils_text import normalizza_testo, unisci_titolo_corpo


def ottieni_classi_modello(modello):
    """Recupera l'elenco delle classi da un modello o da una pipeline sklearn."""
    if hasattr(modello, "named_steps") and "clf" in modello.named_steps:
        clf = modello.named_steps["clf"]
        if hasattr(clf, "classes_"):
            return [str(c) for c in clf.classes_]
    if hasattr(modello, "classes_"):
        return [str(c) for c in modello.classes_]
    raise ValueError("Impossibile recuperare le classi dal modello.")


def supporta_spiegabilita_token(modello) -> bool:
    """Verifica se il modello supporta spiegazioni token-level via TF-IDF."""
    return hasattr(modello, "named_steps") and "tfidf" in modello.named_steps and "clf" in modello.named_steps


def esegui_predizione_lotto():
    """Applica i modelli baseline a un CSV e salva risultati più sintesi SLA."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--percorso_input", "--input", dest="percorso_input", type=str, required=True, help="CSV con colonne title, body (opzionale id)")
    parser.add_argument(
        "--percorso_output",
        "--out",
        dest="percorso_output",
        type=str,
        default=None,
        help="CSV output (default: outputs/predictions_TIMESTAMP.csv)",
    )
    parser.add_argument("--modello_reparto", "--dep_model", dest="modello_reparto", type=str, default="models/department_model.joblib")
    parser.add_argument("--modello_sentiment", "--sent_model", dest="modello_sentiment", type=str, default="models/sentiment_model.joblib")
    parser.add_argument("--soglie", "--thresholds", dest="soglie", type=str, default="outputs/thresholds.json")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--revisione_diagnostica",
        "--diagnostic_review",
        dest="revisione_diagnostica",
        action="store_true",
        help="Aggiunge needs_review diagnostico",
    )
    parser.add_argument(
        "--disabilita_guardrail_sentiment",
        "--disable_sentiment_guardrail",
        dest="disabilita_guardrail_sentiment",
        action="store_true",
        help="Disabilita il guardrail lessicale anti-falso-positivo sul sentiment",
    )
    parser.add_argument(
        "--disabilita_spiegazioni",
        "--disable_explanations",
        dest="disabilita_spiegazioni",
        action="store_true",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.percorso_input)
    if "title" not in df.columns or "body" not in df.columns:
        raise ValueError("Il CSV deve contenere le colonne 'title' e 'body'.")

    modello_reparto = joblib.load(args.modello_reparto)
    modello_sentiment = joblib.load(args.modello_sentiment)

    classi_reparto = ottieni_classi_modello(modello_reparto)
    classi_sentiment = ottieni_classi_modello(modello_sentiment)

    payload_soglie = carica_json(args.soglie, predefinito=DEFAULT_THRESHOLDS)
    soglie_reparto = normalizza_mappa_soglie(payload_soglie.get("department", {}), classi_reparto, fallback=0.55)
    soglie_sentiment = normalizza_mappa_soglie(payload_soglie.get("sentiment", {}), classi_sentiment, fallback=0.55)

    out_df = df.copy()
    out_df["text"] = out_df.apply(
        lambda r: normalizza_testo(unisci_titolo_corpo(str(r["title"]), str(r["body"]))),
        axis=1,
    )

    dep_pred = modello_reparto.predict(out_df["text"].values)
    dep_proba = modello_reparto.predict_proba(out_df["text"].values)

    sent_pred = modello_sentiment.predict(out_df["text"].values)
    sent_proba = modello_sentiment.predict_proba(out_df["text"].values)

    out_df["pred_department"] = dep_pred
    out_df["pred_sentiment"] = sent_pred
    out_df["department_confidence"] = dep_proba.max(axis=1)
    out_df["sentiment_confidence"] = sent_proba.max(axis=1)

    for idx, c in enumerate(classi_reparto):
        out_df[f"proba_department_{c}"] = dep_proba[:, idx]
    for idx, c in enumerate(classi_sentiment):
        out_df[f"proba_sentiment_{c}"] = sent_proba[:, idx]

    if not args.disabilita_guardrail_sentiment:
        out_df = applica_guardrail_sentiment_df(out_df)
    out_df = applica_reparti_impattati_df(out_df)

    out_df = applica_campi_operativi(
        out_df,
        soglie_reparto=soglie_reparto,
        soglie_sentiment=soglie_sentiment,
        revisione_diagnostica=args.revisione_diagnostica,
    )

    if not args.revisione_diagnostica and "needs_review_diag" in out_df.columns:
        out_df = out_df.drop(columns=["needs_review_diag"])

    if not args.disabilita_spiegazioni and supporta_spiegabilita_token(modello_reparto) and supporta_spiegabilita_token(modello_sentiment):
        token_reparto = []
        token_sentiment = []
        for txt in out_df["text"].values:
            dep_exp = contributi_principali_token(modello_reparto, txt, top_k=args.top_k)
            sent_exp = contributi_principali_token(modello_sentiment, txt, top_k=args.top_k)
            token_reparto.append(", ".join([tok for tok, _ in dep_exp]))
            token_sentiment.append(", ".join([tok for tok, _ in sent_exp]))

        out_df["explain_dep_top_tokens"] = token_reparto
        out_df["explain_sent_top_tokens"] = token_sentiment
    elif not args.disabilita_spiegazioni:
        print("[INFO] Explainability token-level non disponibile per questo tipo di modello.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    percorso_output = args.percorso_output or f"outputs/predictions_{ts}.csv"
    garantisci_cartella_padre(percorso_output)
    out_df.to_csv(percorso_output, index=False)

    sla_df = simula_sla(out_df)
    sla_path = f"outputs/sla_batch_summary_{ts}.json"
    salva_json({"rows": sla_df.to_dict(orient="records")}, sla_path)

    print(f"[OK] Predizioni salvate: {percorso_output}")
    print(f"[OK] SLA summary: {sla_path}")
    if "sentiment_guardrail_applied" in out_df.columns:
        print(f"[GUARDRAIL] applied_rate={out_df['sentiment_guardrail_applied'].mean():.4f}")
    if args.revisione_diagnostica and "needs_review_diag" in out_df.columns:
        print(
            f"[DIAGNOSTIC] coverage={(1.0 - out_df['needs_review_diag'].mean()):.4f} | "
            f"needs_review_rate={out_df['needs_review_diag'].mean():.4f}"
        )
    print(out_df.head(5).to_string(index=False))


if __name__ == "__main__":
    esegui_predizione_lotto()
