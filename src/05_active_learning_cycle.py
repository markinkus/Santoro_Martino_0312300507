import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils_ops import (
    DEFAULT_THRESHOLDS,
    applica_campi_operativi,
    applica_guardrail_sentiment_df,
    applica_reparti_impattati_df,
    carica_json,
    normalizza_mappa_soglie,
)
from utils_text import normalizza_testo, unisci_titolo_corpo


def ottieni_classi_modello(modello):
    """Recupera le classi note da un modello o da una pipeline sklearn."""
    if hasattr(modello, "named_steps") and "clf" in modello.named_steps:
        clf = modello.named_steps["clf"]
        if hasattr(clf, "classes_"):
            return [str(c) for c in clf.classes_]
    if hasattr(modello, "classes_"):
        return [str(c) for c in modello.classes_]
    raise ValueError("Impossibile recuperare le classi dal modello.")


def _calcola_punteggi_novita(testi_pool: np.ndarray, testi_base: np.ndarray) -> np.ndarray:
    """Stima quanto i testi del pool siano diversi dai testi già visti."""
    if len(testi_pool) == 0:
        return np.array([], dtype=float)
    if len(testi_base) == 0:
        return np.ones(len(testi_pool), dtype=float)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=30000)
    joined = list(testi_base) + list(testi_pool)
    x_all = vectorizer.fit_transform(joined)
    x_base = x_all[: len(testi_base)]
    x_pool = x_all[len(testi_base) :]

    sims = cosine_similarity(x_pool, x_base)
    max_sim = sims.max(axis=1)
    novelty = 1.0 - max_sim
    return np.clip(novelty, 0.0, 1.0)


def _calcola_costo_operativo(pred_department: str, pred_sentiment: str, priority: str, risk_score: float) -> float:
    """Combina reparto, sentiment, priorità e rischio in un costo operativo sintetico."""
    dep_cost = {"Reception": 0.95, "Housekeeping": 1.05, "F&B": 1.20}
    sent_cost = {"pos": 0.45, "neg": 1.00}
    priority_cost = {"LOW": 0.25, "MEDIUM": 0.55, "HIGH": 0.80, "URGENT": 1.00}

    d = float(dep_cost.get(str(pred_department), 1.0))
    s = float(sent_cost.get(str(pred_sentiment), 0.7))
    p = float(priority_cost.get(str(priority), 0.5))
    r = float(np.clip(risk_score, 0.0, 1.0))

    raw = 0.45 * s + 0.30 * p + 0.15 * d + 0.10 * r
    return float(np.clip(raw, 0.0, 1.0))


def _prepara_predizioni(
    df: pd.DataFrame,
    df_base: pd.DataFrame,
    modello_reparto,
    modello_sentiment,
    soglie_reparto: dict,
    soglie_sentiment: dict,
    strategia: str,
    peso_incertezza: float,
    peso_diversita: float,
    peso_operativo: float,
) -> pd.DataFrame:
    """Costruisce le predizioni del pool e i punteggi usati nell'active learning."""
    out = df.copy()
    out["text"] = out.apply(
        lambda r: normalizza_testo(unisci_titolo_corpo(str(r.get("title", "")), str(r.get("body", "")))),
        axis=1,
    )
    testi_base = df_base.apply(
        lambda r: normalizza_testo(unisci_titolo_corpo(str(r.get("title", "")), str(r.get("body", "")))),
        axis=1,
    ).values

    dep_pred = modello_reparto.predict(out["text"].values)
    dep_proba = modello_reparto.predict_proba(out["text"].values)
    dep_classes = ottieni_classi_modello(modello_reparto)

    sent_pred = modello_sentiment.predict(out["text"].values)
    sent_proba = modello_sentiment.predict_proba(out["text"].values)
    sent_classes = ottieni_classi_modello(modello_sentiment)

    out["pred_department"] = dep_pred
    out["pred_sentiment"] = sent_pred
    out["department_confidence"] = dep_proba.max(axis=1)
    out["sentiment_confidence"] = sent_proba.max(axis=1)

    for idx, c in enumerate(dep_classes):
        out[f"proba_department_{c}"] = dep_proba[:, idx]
    for idx, c in enumerate(sent_classes):
        out[f"proba_sentiment_{c}"] = sent_proba[:, idx]

    out = applica_guardrail_sentiment_df(out)
    out = applica_reparti_impattati_df(out)

    out = applica_campi_operativi(
        out,
        soglie_reparto=soglie_reparto,
        soglie_sentiment=soglie_sentiment,
        revisione_diagnostica=True,
    )

    # incertezza alta quando le confidenze sono basse
    out["uncertainty_score"] = 1.0 - (0.5 * out["department_confidence"] + 0.5 * out["sentiment_confidence"])
    out["diversity_score"] = _calcola_punteggi_novita(out["text"].values, testi_base=testi_base)
    out["business_cost_score"] = out.apply(
        lambda r: _calcola_costo_operativo(
            pred_department=str(r["pred_department"]),
            pred_sentiment=str(r["pred_sentiment"]),
            priority=str(r["priority"]),
            risk_score=float(r["risk_score"]),
        ),
        axis=1,
    )

    if strategia == "uncertainty_only":
        out["acquisition_score"] = out["uncertainty_score"]
    else:
        total = max(1e-8, float(peso_incertezza + peso_diversita + peso_operativo))
        wu = float(peso_incertezza) / total
        wd = float(peso_diversita) / total
        wb = float(peso_operativo) / total
        out["acquisition_score"] = (
            wu * out["uncertainty_score"] + wd * out["diversity_score"] + wb * out["business_cost_score"]
        )

    return out


def _costruisci_coda(df_predizioni: pd.DataFrame, numero_top: int, solo_revisione_diagnostica: bool) -> pd.DataFrame:
    """Ordina e seleziona i campioni da revisionare nel ciclo di active learning."""
    df = df_predizioni.copy()
    if solo_revisione_diagnostica:
        df = df[df["needs_review_diag"]]

    df = df.sort_values(["acquisition_score", "uncertainty_score", "risk_score"], ascending=[False, False, False])
    df = df.head(numero_top).copy()

    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    queue = df[
        [
            "id",
            "title",
            "body",
            "pred_department",
            "pred_sentiment",
            "department_confidence",
            "sentiment_confidence",
            "uncertainty_score",
            "diversity_score",
            "business_cost_score",
            "acquisition_score",
            "needs_review_diag",
            "priority",
            "risk_score",
        ]
    ].copy()

    queue["true_department"] = ""
    queue["true_sentiment"] = ""
    queue["reviewer_notes"] = ""
    return queue


def _campiona_righe_replay(df_base: pd.DataFrame, dimensione_replay: int, seed_replay: int) -> pd.DataFrame:
    """Estrae righe del dataset storico per il replay controllato."""
    if dimensione_replay <= 0 or df_base.empty:
        return pd.DataFrame(columns=df_base.columns)

    grp_counts = (
        df_base.groupby(["department", "sentiment"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    total = int(grp_counts["count"].sum())
    if total <= 0:
        return pd.DataFrame(columns=df_base.columns)

    # Largest remainder allocation
    grp_counts["quota_raw"] = (grp_counts["count"] / total) * dimensione_replay
    grp_counts["quota"] = np.floor(grp_counts["quota_raw"]).astype(int)
    remainder = int(dimensione_replay - grp_counts["quota"].sum())
    if remainder > 0:
        grp_counts["frac"] = grp_counts["quota_raw"] - grp_counts["quota"]
        for idx in grp_counts.sort_values("frac", ascending=False).head(remainder).index:
            grp_counts.loc[idx, "quota"] += 1

    rng = np.random.default_rng(seed_replay)
    samples = []
    for _, row in grp_counts.iterrows():
        dep = row["department"]
        sent = row["sentiment"]
        n_take = int(row["quota"])
        if n_take <= 0:
            continue
        sub = df_base[(df_base["department"] == dep) & (df_base["sentiment"] == sent)]
        if sub.empty:
            continue
        n_take = min(n_take, len(sub))
        idx = rng.choice(sub.index.values, size=n_take, replace=False)
        samples.append(sub.loc[idx])

    if not samples:
        return pd.DataFrame(columns=base_df.columns)
    return pd.concat(samples, ignore_index=True)


def _aggiungi_campioni_etichettati(
    df_base: pd.DataFrame,
    coda_etichettata: pd.DataFrame,
    dimensione_replay: int,
    seed_replay: int,
) -> pd.DataFrame:
    """Appende al dataset base i campioni etichettati selezionati nella coda."""
    required = ["title", "body", "true_department", "true_sentiment"]
    missing = [c for c in required if c not in coda_etichettata.columns]
    if missing:
        raise ValueError(f"Labeled queue incompleta. Mancano colonne: {missing}")

    labeled = coda_etichettata.copy()
    labeled = labeled[
        labeled["true_department"].notna()
        & labeled["true_sentiment"].notna()
        & labeled["true_department"].astype(str).str.strip().ne("")
        & labeled["true_sentiment"].astype(str).str.strip().ne("")
    ].copy()

    if labeled.empty:
        raise ValueError("Nessun campione etichettato trovato nel file labeled_queue.")

    next_id = int(df_base["id"].max()) + 1 if "id" in df_base.columns and len(df_base) else 1
    new_rows = []
    for _, row in labeled.iterrows():
        new_rows.append(
            {
                "id": next_id,
                "title": str(row["title"]),
                "body": str(row["body"]),
                "department": str(row["true_department"]),
                "sentiment": str(row["true_sentiment"]),
            }
        )
        next_id += 1

    out = pd.concat([df_base, pd.DataFrame(new_rows)], ignore_index=True)

    replay_rows = _campiona_righe_replay(df_base=df_base, dimensione_replay=dimensione_replay, seed_replay=seed_replay)
    if not replay_rows.empty:
        replay_rows = replay_rows.copy()
        replay_rows["id"] = range(next_id, next_id + len(replay_rows))
        out = pd.concat([out, replay_rows], ignore_index=True)
    return out


def esegui_ciclo_apprendimento_attivo():
    """Costruisce la coda active learning e, se richiesto, aggiorna il dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/reviews_synth.csv")
    parser.add_argument("--pool", type=str, default="data/benchmarks/reviews_noisy.csv")
    parser.add_argument("--modello_reparto", "--dep_model", dest="modello_reparto", type=str, default="models/department_model.joblib")
    parser.add_argument("--modello_sentiment", "--sent_model", dest="modello_sentiment", type=str, default="models/sentiment_model.joblib")
    parser.add_argument("--soglie", "--thresholds", dest="soglie", type=str, default="outputs/thresholds.json")
    parser.add_argument("--numero_top", "--top_n", dest="numero_top", type=int, default=60)
    parser.add_argument("--solo_revisione_diagnostica", "--only_diag_review", dest="solo_revisione_diagnostica", action="store_true")
    parser.add_argument("--strategia", "--strategy", dest="strategia", type=str, default="hybrid_v2", choices=["hybrid_v2", "uncertainty_only"])
    parser.add_argument("--peso_incertezza", "--w_uncertainty", dest="peso_incertezza", type=float, default=0.50)
    parser.add_argument("--peso_diversita", "--w_diversity", dest="peso_diversita", type=float, default=0.30)
    parser.add_argument("--peso_operativo", "--w_business", dest="peso_operativo", type=float, default=0.20)
    parser.add_argument("--percorso_coda_output", "--queue_out", dest="percorso_coda_output", type=str, default=None)
    parser.add_argument("--coda_etichettata", "--labeled_queue", dest="coda_etichettata", type=str, default=None)
    parser.add_argument("--percorso_dataset_output", "--dataset_out", dest="percorso_dataset_output", type=str, default="data/reviews_synth_active.csv")
    parser.add_argument("--dimensione_replay", "--replay_size", dest="dimensione_replay", type=int, default=0, help="Numero campioni replay da duplicare nel dataset finale")
    parser.add_argument("--seed_replay", "--replay_seed", dest="seed_replay", type=int, default=42)
    parser.add_argument("--riaddestra", "--retrain", dest="riaddestra", action="store_true")
    args = parser.parse_args()

    df_base = pd.read_csv(args.dataset)
    pool_df = pd.read_csv(args.pool)

    modello_reparto = joblib.load(args.modello_reparto)
    modello_sentiment = joblib.load(args.modello_sentiment)

    classi_reparto = list(modello_reparto.named_steps["clf"].classes_)
    classi_sentiment = list(modello_sentiment.named_steps["clf"].classes_)

    payload_soglie = carica_json(args.soglie, predefinito=DEFAULT_THRESHOLDS)
    soglie_reparto = normalizza_mappa_soglie(payload_soglie.get("department", {}), classi_reparto, fallback=0.55)
    soglie_sentiment = normalizza_mappa_soglie(payload_soglie.get("sentiment", {}), classi_sentiment, fallback=0.55)

    pred_df = _prepara_predizioni(
        df=pool_df,
        df_base=df_base,
        modello_reparto=modello_reparto,
        modello_sentiment=modello_sentiment,
        soglie_reparto=soglie_reparto,
        soglie_sentiment=soglie_sentiment,
        strategia=args.strategia,
        peso_incertezza=args.peso_incertezza,
        peso_diversita=args.peso_diversita,
        peso_operativo=args.peso_operativo,
    )

    queue_df = _costruisci_coda(
        df_predizioni=pred_df,
        numero_top=args.numero_top,
        solo_revisione_diagnostica=args.solo_revisione_diagnostica,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    percorso_coda_output = args.percorso_coda_output or f"outputs/active_learning_queue_{ts}.csv"
    Path(percorso_coda_output).parent.mkdir(parents=True, exist_ok=True)
    queue_df.to_csv(percorso_coda_output, index=False)

    print(f"[OK] Coda active learning salvata: {percorso_coda_output}")
    print(
        f"[INFO] righe_coda={len(queue_df)} | "
        f"incertezza_media={queue_df['uncertainty_score'].mean():.4f} | "
        f"diversita_media={queue_df['diversity_score'].mean():.4f} | "
        f"costo_operativo_medio={queue_df['business_cost_score'].mean():.4f} | "
        f"acquisizione_media={queue_df['acquisition_score'].mean():.4f}"
    )

    if args.coda_etichettata:
        df_etichettato = pd.read_csv(args.coda_etichettata)
        updated_df = _aggiungi_campioni_etichettati(
            df_base=df_base,
            coda_etichettata=df_etichettato,
            dimensione_replay=args.dimensione_replay,
            seed_replay=args.seed_replay,
        )
        Path(args.percorso_dataset_output).parent.mkdir(parents=True, exist_ok=True)
        updated_df.to_csv(args.percorso_dataset_output, index=False)

        added = len(updated_df) - len(df_base)
        print(f"[OK] Dataset aggiornato: {args.percorso_dataset_output}")
        print(f"[INFO] Campioni aggiunti dal ciclo active learning (+replay se attivo): {added}")

        if args.riaddestra:
            cmd = ["python3", "src/02_train_evaluate.py", "--dati", args.percorso_dataset_output]
            print("[INFO] Avvio riaddestramento automatico...")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    esegui_ciclo_apprendimento_attivo()
