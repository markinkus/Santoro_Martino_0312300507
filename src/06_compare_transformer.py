import argparse
from datetime import datetime
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from utils_ops import applica_guardrail_sentiment_df, salva_json
from utils_text import normalizza_testo, unisci_titolo_corpo


def _prepara_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge la colonna testuale normalizzata per il confronto transformer."""
    out = df.copy()
    out["text"] = out.apply(
        lambda r: normalizza_testo(unisci_titolo_corpo(str(r.get("title", "")), str(r.get("body", "")))),
        axis=1,
    )
    return out


def _calcola_metriche_task(y_true, y_pred) -> dict:
    """Calcola accuracy e macro-F1 per un singolo task."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def _classi_modello(modello) -> list[str]:
    """Recupera le classi esposte da pipeline sklearn o classificatori calibrati."""
    if hasattr(modello, "classes_"):
        return [str(classe) for classe in modello.classes_]

    named_steps = getattr(modello, "named_steps", {})
    clf = named_steps.get("clf") if named_steps else None
    if clf is not None and hasattr(clf, "classes_"):
        return [str(classe) for classe in clf.classes_]

    raise ValueError("Impossibile recuperare le classi del modello.")


def _predici_sentiment_con_guardrail(
    modello_sentiment,
    x_modello: np.ndarray,
    testi_guardrail: np.ndarray | None = None,
) -> np.ndarray:
    """Predice il sentiment applicando lo stesso guardrail usato dal runtime."""
    pred = modello_sentiment.predict(x_modello)
    proba = modello_sentiment.predict_proba(x_modello)
    classi = _classi_modello(modello_sentiment)
    testi = testi_guardrail if testi_guardrail is not None else x_modello

    df_pred = pd.DataFrame(
        {
            "text": [str(testo) for testo in testi],
            "pred_sentiment": pred,
            "sentiment_confidence": proba.max(axis=1),
        }
    )
    for idx, classe in enumerate(classi):
        df_pred[f"proba_sentiment_{classe}"] = proba[:, idx]

    df_pred = applica_guardrail_sentiment_df(df_pred)
    return df_pred["pred_sentiment"].values


def _valuta_coppia_modelli(
    modello_reparto,
    modello_sentiment,
    x: np.ndarray,
    y_dep: np.ndarray,
    y_sent: np.ndarray,
    testi_guardrail: np.ndarray | None = None,
) -> dict:
    """Valuta una coppia di modelli reparto/sentiment sullo stesso split."""
    dep_pred = modello_reparto.predict(x)
    sent_pred = _predici_sentiment_con_guardrail(modello_sentiment, x, testi_guardrail=testi_guardrail)
    return {
        "department": _calcola_metriche_task(y_dep, dep_pred),
        "sentiment": _calcola_metriche_task(y_sent, sent_pred),
    }


def _addestra_modelli_transformer(
    x_train_text: np.ndarray,
    y_dep_train: np.ndarray,
    y_sent_train: np.ndarray,
    nome_modello: str,
):
    """Addestra encoder MiniLM e classificatori calibrati sui suoi embedding."""
    # Forza Transformers per evitare di caricare i path di TensorFlow/Keras.
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "Dipendenze transformer non coerenti. Installa gli extra dedicati con: "
            "pip install -r requisiti-transformer.txt"
        ) from exc

    encoder = SentenceTransformer(nome_modello)
    emb_train = encoder.encode(list(x_train_text), batch_size=32, show_progress_bar=False, convert_to_numpy=True)

    dep_base = LogisticRegression(max_iter=4000, solver="lbfgs")
    sent_base = LogisticRegression(max_iter=4000, solver="lbfgs")

    dep_model = CalibratedClassifierCV(estimator=dep_base, method="sigmoid", cv=3)
    sent_model = CalibratedClassifierCV(estimator=sent_base, method="sigmoid", cv=3)

    dep_model.fit(emb_train, y_dep_train)
    sent_model.fit(emb_train, y_sent_train)

    return encoder, dep_model, sent_model


def esegui_confronto_transformer():
    """Confronta baseline e MiniLM sul test held-out e sui benchmark separati."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dati", "--data", dest="percorso_dati", type=str, default="data/reviews_synth.csv")
    parser.add_argument("--cartella_benchmark", "--benchmarks_dir", dest="cartella_benchmark", type=str, default="data/benchmarks")
    parser.add_argument("--modello_reparto_baseline", "--baseline_dep_model", dest="modello_reparto_baseline", type=str, default="models/department_model.joblib")
    parser.add_argument("--modello_sentiment_baseline", "--baseline_sent_model", dest="modello_sentiment_baseline", type=str, default="models/sentiment_model.joblib")
    parser.add_argument(
        "--modello_transformer",
        "--transformer_model",
        dest="modello_transformer",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--quota_test", "--test_size", dest="quota_test", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--percorso_output", "--out", dest="percorso_output", type=str, default=None)
    args = parser.parse_args()

    df = _prepara_dataframe(pd.read_csv(args.percorso_dati))

    x = df["text"].values
    y_dep = df["department"].values
    y_sent = df["sentiment"].values
    y_joint = np.array([f"{a}__{b}" for a, b in zip(y_dep, y_sent)])

    x_train, x_test, y_dep_train, y_dep_test, y_sent_train, y_sent_test = train_test_split(
        x,
        y_dep,
        y_sent,
        test_size=args.quota_test,
        random_state=args.seed,
        stratify=y_joint,
    )

    baseline_dep = joblib.load(args.modello_reparto_baseline)
    baseline_sent = joblib.load(args.modello_sentiment_baseline)

    print("[INFO] Addestramento transformer leggero e calibrazione...")
    encoder, tr_dep, tr_sent = _addestra_modelli_transformer(
        x_train_text=x_train,
        y_dep_train=y_dep_train,
        y_sent_train=y_sent_train,
        nome_modello=args.modello_transformer,
    )

    baseline_test = _valuta_coppia_modelli(baseline_dep, baseline_sent, x_test, y_dep_test, y_sent_test)

    emb_test = encoder.encode(list(x_test), batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    transformer_test = _valuta_coppia_modelli(
        tr_dep,
        tr_sent,
        emb_test,
        y_dep_test,
        y_sent_test,
        testi_guardrail=x_test,
    )

    benchmarks = {}
    bench_dir = Path(args.cartella_benchmark)
    if bench_dir.exists():
        for csv_path in sorted(bench_dir.glob("*.csv")):
            bdf = _prepara_dataframe(pd.read_csv(csv_path))
            bx = bdf["text"].values
            by_dep = bdf["department"].values
            by_sent = bdf["sentiment"].values

            b_base = _valuta_coppia_modelli(baseline_dep, baseline_sent, bx, by_dep, by_sent)
            bx_emb = encoder.encode(list(bx), batch_size=32, show_progress_bar=False, convert_to_numpy=True)
            b_tr = _valuta_coppia_modelli(
                tr_dep,
                tr_sent,
                bx_emb,
                by_dep,
                by_sent,
                testi_guardrail=bx,
            )

            benchmarks[csv_path.name] = {
                "baseline": b_base,
                "transformer": b_tr,
                "delta_f1_macro": {
                    "department": float(b_tr["department"]["f1_macro"] - b_base["department"]["f1_macro"]),
                    "sentiment": float(b_tr["sentiment"]["f1_macro"] - b_base["sentiment"]["f1_macro"]),
                },
            }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.percorso_output or f"outputs/transformer_comparison_{ts}.json"

    payload = {
        "timestamp": ts,
        "transformer_model": args.modello_transformer,
        "test_split": {
            "baseline": baseline_test,
            "transformer": transformer_test,
            "delta_f1_macro": {
                "department": float(transformer_test["department"]["f1_macro"] - baseline_test["department"]["f1_macro"]),
                "sentiment": float(transformer_test["sentiment"]["f1_macro"] - baseline_test["sentiment"]["f1_macro"]),
            },
        },
        "benchmarks": benchmarks,
    }

    salva_json(payload, out_path)

    print(f"[OK] Confronto salvato: {out_path}")
    print(
        "[TEST] base reparto/sentiment F1="
        f"{baseline_test['department']['f1_macro']:.4f}/{baseline_test['sentiment']['f1_macro']:.4f} | "
        "transformer reparto/sentiment F1="
        f"{transformer_test['department']['f1_macro']:.4f}/{transformer_test['sentiment']['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    esegui_confronto_transformer()
