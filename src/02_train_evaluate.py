import argparse
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from utils_ops import (
    applica_campi_operativi,
    applica_guardrail_sentiment_df,
    applica_reparti_impattati_df,
    deriva_soglie_per_classe,
    garantisci_cartella,
    garantisci_cartella_padre,
    normalizza_mappa_soglie,
    salva_json,
    simula_sla,
)
from utils_text import normalizza_testo, unisci_titolo_corpo


def leggi_lista_seed(testo_seed: str) -> list[int]:
    """Converte una lista di seed separati da virgola in interi."""
    out = []
    for item in testo_seed.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out or [42]


def salva_matrice_confusione(matrice, etichette, percorso_output, titolo):
    """Salva una confusion matrix come immagine PNG."""
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(matrice)
    plt.title(titolo)
    plt.xticks(range(len(etichette)), etichette, rotation=45, ha="right")
    plt.yticks(range(len(etichette)), etichette)

    for i in range(matrice.shape[0]):
        for j in range(matrice.shape[1]):
            plt.text(j, i, str(matrice[i, j]), ha="center", va="center")

    plt.tight_layout()
    garantisci_cartella_padre(percorso_output)
    fig.savefig(percorso_output, dpi=200)
    plt.close(fig)


def _prepara_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Crea la colonna testuale normalizzata a partire da titolo e corpo."""
    out = df.copy()
    out["text"] = out.apply(
        lambda r: normalizza_testo(unisci_titolo_corpo(str(r.get("title", "")), str(r.get("body", "")))),
        axis=1,
    )
    return out


def _calcola_metriche_task(y_true: np.ndarray, y_pred: np.ndarray, etichette: list[str]) -> dict:
    """Calcola le metriche principali di un task di classificazione."""
    report = classification_report(y_true, y_pred, labels=etichette, output_dict=True, zero_division=0)
    recall_by_class = {c: float(report.get(c, {}).get("recall", 0.0)) for c in etichette}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "recall_by_class": recall_by_class,
        "report": report,
    }


def _crea_pipeline_base() -> Pipeline:
    """Costruisce la pipeline baseline TF-IDF + Logistic Regression calibrata."""
    calibrated_lr = CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=3000, solver="liblinear"),
        method="sigmoid",
        cv=3,
    )

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    sublinear_tf=True,
                ),
            ),
            ("clf", calibrated_lr),
        ]
    )


def _crea_griglia_parametri() -> list[dict]:
    """Definisce la griglia di iperparametri per la baseline."""
    return [
        {
            "tfidf__analyzer": ["word"],
            "tfidf__ngram_range": [(1, 2)],
            "tfidf__min_df": [1, 2],
            "tfidf__max_df": [0.92, 0.98],
            "clf__estimator__C": [0.7, 1.2, 2.0],
        },
        {
            "tfidf__analyzer": ["char_wb"],
            "tfidf__ngram_range": [(3, 5), (4, 6)],
            "tfidf__min_df": [1, 2],
            "tfidf__max_df": [0.98, 1.0],
            "clf__estimator__C": [0.7, 1.2],
        },
    ]


def _addestra_con_grid_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    numero_fold_cv: int,
    numero_job: int,
    verbose: int,
) -> tuple[Pipeline, dict, list[dict]]:
    """Esegue la GridSearchCV e restituisce modello migliore e top combinazioni."""
    cv = StratifiedKFold(n_splits=numero_fold_cv, shuffle=True, random_state=seed)

    grid = GridSearchCV(
        estimator=_crea_pipeline_base(),
        param_grid=_crea_griglia_parametri(),
        scoring="f1_macro",
        cv=cv,
        n_jobs=numero_job,
        refit=True,
        verbose=verbose,
    )
    grid.fit(x_train, y_train)

    cv_df = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")
    top_rows = []
    for _, row in cv_df.head(5).iterrows():
        top_rows.append(
            {
                "rank": int(row["rank_test_score"]),
                "mean_test_score": float(row["mean_test_score"]),
                "std_test_score": float(row["std_test_score"]),
                "params": dict(row["params"]),
            }
        )

    return grid.best_estimator_, dict(grid.best_params_), top_rows


def _valuta_set_benchmark(
    percorso_input: Path,
    modello_reparto: Pipeline,
    modello_sentiment: Pipeline,
    soglie_reparto: dict,
    soglie_sentiment: dict,
    revisione_diagnostica: bool,
) -> dict:
    """Valuta baseline e runtime su un benchmark separato."""
    df = pd.read_csv(percorso_input)
    df = _prepara_dataframe(df)

    dep_pred = modello_reparto.predict(df["text"].values)
    dep_proba = modello_reparto.predict_proba(df["text"].values)
    dep_classes = list(modello_reparto.named_steps["clf"].classes_)

    sent_pred = modello_sentiment.predict(df["text"].values)
    sent_proba = modello_sentiment.predict_proba(df["text"].values)
    sent_classes = list(modello_sentiment.named_steps["clf"].classes_)

    out_df = df.copy()
    out_df["pred_department"] = dep_pred
    out_df["pred_sentiment"] = sent_pred
    out_df["department_confidence"] = dep_proba.max(axis=1)
    out_df["sentiment_confidence"] = sent_proba.max(axis=1)

    for idx, c in enumerate(dep_classes):
        out_df[f"proba_department_{c}"] = dep_proba[:, idx]
    for idx, c in enumerate(sent_classes):
        out_df[f"proba_sentiment_{c}"] = sent_proba[:, idx]

    out_df = applica_guardrail_sentiment_df(out_df)
    out_df = applica_reparti_impattati_df(out_df)

    out_df = applica_campi_operativi(
        out_df,
        soglie_reparto=soglie_reparto,
        soglie_sentiment=soglie_sentiment,
        revisione_diagnostica=revisione_diagnostica,
    )

    result = {
        "rows": int(len(out_df)),
        "priority_distribution": out_df["priority"].value_counts(normalize=True).to_dict() if len(out_df) else {},
    }

    if revisione_diagnostica:
        result["coverage"] = float(1.0 - out_df["needs_review_diag"].mean()) if len(out_df) else 0.0
        result["needs_review_rate"] = float(out_df["needs_review_diag"].mean()) if len(out_df) else 0.0
    else:
        result["coverage"] = None
        result["needs_review_rate"] = None

    if "department" in out_df.columns:
        result["department"] = _calcola_metriche_task(
            out_df["department"].values,
            out_df["pred_department"].values,
            etichette=dep_classes,
        )

    if "sentiment" in out_df.columns:
        result["sentiment"] = _calcola_metriche_task(
            out_df["sentiment"].values,
            out_df["pred_sentiment"].values,
            etichette=sent_classes,
        )

    sla_df = simula_sla(out_df)
    result["sla_summary"] = sla_df.to_dict(orient="records")
    return result


def _esegui_seed_singolo(
    df: pd.DataFrame,
    seed: int,
    quota_test: float,
    parametri_migliori_reparto: dict,
    parametri_migliori_sentiment: dict,
) -> dict:
    """Riesegue il protocollo di train/test su un seed alternativo."""
    x = df["text"].values
    y_dep = df["department"].values
    y_sent = df["sentiment"].values
    y_joint = np.array([f"{a}__{b}" for a, b in zip(y_dep, y_sent)])

    x_train, x_test, y_dep_train, y_dep_test, y_sent_train, y_sent_test = train_test_split(
        x,
        y_dep,
        y_sent,
        test_size=quota_test,
        random_state=seed,
        stratify=y_joint,
    )

    dep_model = _crea_pipeline_base().set_params(**parametri_migliori_reparto)
    sent_model = _crea_pipeline_base().set_params(**parametri_migliori_sentiment)

    dep_model.fit(x_train, y_dep_train)
    sent_model.fit(x_train, y_sent_train)

    dep_pred = dep_model.predict(x_test)
    sent_pred = sent_model.predict(x_test)

    return {
        "seed": int(seed),
        "department_accuracy": float(accuracy_score(y_dep_test, dep_pred)),
        "department_f1_macro": float(f1_score(y_dep_test, dep_pred, average="macro")),
        "sentiment_accuracy": float(accuracy_score(y_sent_test, sent_pred)),
        "sentiment_f1_macro": float(f1_score(y_sent_test, sent_pred, average="macro")),
    }


def _aggrega_metriche_seed(righe: list[dict]) -> dict:
    """Aggrega media, deviazione standard e range delle metriche multi-seed."""
    keys = [
        "department_accuracy",
        "department_f1_macro",
        "sentiment_accuracy",
        "sentiment_f1_macro",
    ]
    out = {"runs": righe}
    for k in keys:
        vals = np.array([r[k] for r in righe], dtype=float)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return out


def esegui_addestramento_e_valutazione():
    """Esegue il training baseline, la calibrazione e la valutazione completa."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dati", "--data", dest="percorso_dati", type=str, default="data/reviews_synth.csv")
    parser.add_argument("--cartella_benchmark", "--benchmarks_dir", dest="cartella_benchmark", type=str, default="data/benchmarks")
    parser.add_argument(
        "--percorso_augment_sicurezza",
        "--augment_safety_path",
        dest="percorso_augment_sicurezza",
        type=str,
        default="",
        help="CSV opzionale con casi critici di sicurezza da aggiungere al training (stesse colonne base).",
    )
    parser.add_argument(
        "--ripetizioni_augment_sicurezza",
        "--augment_safety_repeat",
        dest="ripetizioni_augment_sicurezza",
        type=int,
        default=1,
        help="Numero repliche del dataset di sicurezza da aggiungere (0 disabilita).",
    )
    parser.add_argument("--quota_test", "--test_size", dest="quota_test", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lista_seed", "--seeds", dest="lista_seed", type=str, default="13,21,34,42,55")
    parser.add_argument("--fold_grid_cv", "--grid_cv_folds", dest="fold_grid_cv", type=int, default=3)
    parser.add_argument("--fold_cv", "--cv_folds", dest="fold_cv", type=int, default=5)
    parser.add_argument("--quantile_soglia", "--threshold_quantile", dest="quantile_soglia", type=float, default=0.08)
    parser.add_argument("--minimo_soglia", "--threshold_floor", dest="minimo_soglia", type=float, default=0.45)
    parser.add_argument("--revisione_diagnostica", "--diagnostic_review", dest="revisione_diagnostica", action="store_true")
    parser.add_argument("--cartella_modelli", "--models_dir", dest="cartella_modelli", type=str, default="models")
    parser.add_argument("--cartella_output", "--outputs_dir", dest="cartella_output", type=str, default="outputs")
    parser.add_argument("--numero_job", "--n_jobs", dest="numero_job", type=int, default=-1)
    parser.add_argument("--verbose_grid", "--grid_verbose", dest="verbose_grid", type=int, default=0)
    args = parser.parse_args()

    garantisci_cartella(args.cartella_modelli)
    garantisci_cartella(args.cartella_output)

    df = pd.read_csv(args.percorso_dati)
    df = _prepara_dataframe(df)

    augmentation_info = {
        "enabled": False,
        "path": None,
        "rows_added": 0,
        "repeat": int(max(0, args.ripetizioni_augment_sicurezza)),
    }
    if args.percorso_augment_sicurezza and args.ripetizioni_augment_sicurezza > 0:
        aug_path = Path(args.percorso_augment_sicurezza)
        if aug_path.exists():
            aug_df = pd.read_csv(aug_path)
            required = {"title", "body", "department", "sentiment"}
            if not required.issubset(set(aug_df.columns)):
                missing = sorted(list(required - set(aug_df.columns)))
                raise ValueError(f"Dataset di sicurezza incompleto. Colonne mancanti: {missing}")

            aug_df = _prepara_dataframe(aug_df)
            extras = [aug_df.copy() for _ in range(int(args.ripetizioni_augment_sicurezza))]
            df = pd.concat([df] + extras, ignore_index=True)
            augmentation_info = {
                "enabled": True,
                "path": str(aug_path),
                "rows_added": int(len(aug_df) * int(args.ripetizioni_augment_sicurezza)),
                "repeat": int(args.ripetizioni_augment_sicurezza),
            }
            print(
                f"[INFO] Aumento dati di sicurezza attivo: +{augmentation_info['rows_added']} righe "
                f"da {augmentation_info['path']}"
            )
        else:
            print(f"[AVVISO] Percorso augment_safety_path non trovato: {aug_path}. Proseguo senza aumento dati.")

    x = df["text"].values
    y_dep = df["department"].values
    y_sent = df["sentiment"].values
    y_joint = np.array([f"{a}__{b}" for a, b in zip(y_dep, y_sent)])

    split = train_test_split(
        x,
        y_dep,
        y_sent,
        y_joint,
        test_size=args.quota_test,
        random_state=args.seed,
        stratify=y_joint,
    )
    x_pool, x_test, y_dep_pool, y_dep_test, y_sent_pool, y_sent_test, y_joint_pool, _ = split

    split_pool = train_test_split(
        x_pool,
        y_dep_pool,
        y_sent_pool,
        y_joint_pool,
        test_size=0.20,
        random_state=args.seed + 1,
        stratify=y_joint_pool,
    )
    x_train, x_cal, y_dep_train, y_dep_cal, y_sent_train, y_sent_cal, _, _ = split_pool

    print("[INFO] Ricerca griglia per reparto...")
    dep_model, dep_best_params, dep_grid_top = _addestra_con_grid_search(
        x_train=x_train,
        y_train=np.array(y_dep_train),
        seed=args.seed,
        numero_fold_cv=args.fold_grid_cv,
        numero_job=args.numero_job,
        verbose=args.verbose_grid,
    )

    print("[INFO] Ricerca griglia per sentiment...")
    sent_model, sent_best_params, sent_grid_top = _addestra_con_grid_search(
        x_train=x_train,
        y_train=np.array(y_sent_train),
        seed=args.seed,
        numero_fold_cv=args.fold_grid_cv,
        numero_job=args.numero_job,
        verbose=args.verbose_grid,
    )

    dep_classes = list(dep_model.named_steps["clf"].classes_)
    sent_classes = list(sent_model.named_steps["clf"].classes_)

    dep_cal_proba = dep_model.predict_proba(x_cal)
    sent_cal_proba = sent_model.predict_proba(x_cal)

    dep_thresholds = normalizza_mappa_soglie(
        deriva_soglie_per_classe(
            y_true=np.array(y_dep_cal),
            y_proba=dep_cal_proba,
            classi=dep_classes,
            quantile=args.quantile_soglia,
            floor=args.minimo_soglia,
        ),
        classi=dep_classes,
        fallback=args.minimo_soglia,
    )
    sent_thresholds = normalizza_mappa_soglie(
        deriva_soglie_per_classe(
            y_true=np.array(y_sent_cal),
            y_proba=sent_cal_proba,
            classi=sent_classes,
            quantile=args.quantile_soglia,
            floor=args.minimo_soglia,
        ),
        classi=sent_classes,
        fallback=args.minimo_soglia,
    )

    dep_proba = dep_model.predict_proba(x_test)
    sent_proba = sent_model.predict_proba(x_test)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    joblib.dump(dep_model, f"{args.cartella_modelli}/department_model.joblib")
    joblib.dump(sent_model, f"{args.cartella_modelli}/sentiment_model.joblib")

    test_pred_df = pd.DataFrame(
        {
            "text": x_test,
            "true_department": y_dep_test,
            "pred_department": dep_model.predict(x_test),
            "true_sentiment": y_sent_test,
            "pred_sentiment": sent_model.predict(x_test),
            "department_confidence": dep_proba.max(axis=1),
            "sentiment_confidence": sent_proba.max(axis=1),
        }
    )

    for idx, c in enumerate(dep_classes):
        test_pred_df[f"proba_department_{c}"] = dep_proba[:, idx]
    for idx, c in enumerate(sent_classes):
        test_pred_df[f"proba_sentiment_{c}"] = sent_proba[:, idx]

    test_pred_df = applica_guardrail_sentiment_df(test_pred_df)
    test_pred_df = applica_reparti_impattati_df(test_pred_df)

    test_pred_df = applica_campi_operativi(
        test_pred_df,
        soglie_reparto=dep_thresholds,
        soglie_sentiment=sent_thresholds,
        revisione_diagnostica=args.revisione_diagnostica,
    )

    dep_pred = test_pred_df["pred_department"].values
    sent_pred = test_pred_df["pred_sentiment"].values

    dep_metrics = _calcola_metriche_task(np.array(y_dep_test), dep_pred, etichette=dep_classes)
    sent_metrics = _calcola_metriche_task(np.array(y_sent_test), sent_pred, etichette=sent_classes)

    dep_cm = confusion_matrix(y_dep_test, dep_pred, labels=dep_classes)
    sent_cm = confusion_matrix(y_sent_test, sent_pred, labels=sent_classes)

    salva_matrice_confusione(
        dep_cm,
        dep_classes,
        f"{args.cartella_output}/confusion_department.png",
        "Matrice di confusione - Reparto",
    )
    salva_matrice_confusione(
        sent_cm,
        sent_classes,
        f"{args.cartella_output}/confusion_sentiment.png",
        "Matrice di confusione - Sentiment",
    )

    test_pred_path = f"{args.cartella_output}/test_predictions_{ts}.csv"
    garantisci_cartella_padre(test_pred_path)
    test_pred_df.to_csv(test_pred_path, index=False)

    test_sla_df = simula_sla(test_pred_df)

    seeds = leggi_lista_seed(args.lista_seed)
    multi_seed_rows = [
        _esegui_seed_singolo(
            df=df,
            seed=s,
            quota_test=args.quota_test,
            parametri_migliori_reparto=dep_best_params,
            parametri_migliori_sentiment=sent_best_params,
        )
        for s in seeds
    ]
    multi_seed = _aggrega_metriche_seed(multi_seed_rows)

    cv = StratifiedKFold(n_splits=args.fold_cv, shuffle=True, random_state=args.seed)

    dep_cv = cross_val_score(
        _crea_pipeline_base().set_params(**dep_best_params),
        df["text"].values,
        df["department"].values,
        cv=cv,
        scoring="f1_macro",
        n_jobs=args.numero_job,
    )
    sent_cv = cross_val_score(
        _crea_pipeline_base().set_params(**sent_best_params),
        df["text"].values,
        df["sentiment"].values,
        cv=cv,
        scoring="f1_macro",
        n_jobs=args.numero_job,
    )

    benchmark_results = {}
    bench_dir = Path(args.cartella_benchmark)
    if bench_dir.exists():
        for csv_path in sorted(bench_dir.glob("*.csv")):
            benchmark_results[csv_path.name] = _valuta_set_benchmark(
                percorso_input=csv_path,
                modello_reparto=dep_model,
                modello_sentiment=sent_model,
                soglie_reparto=dep_thresholds,
                soglie_sentiment=sent_thresholds,
                revisione_diagnostica=args.revisione_diagnostica,
            )

    if args.revisione_diagnostica:
        coverage = float(1.0 - test_pred_df["needs_review_diag"].mean())
        needs_review_rate = float(test_pred_df["needs_review_diag"].mean())
    else:
        coverage = None
        needs_review_rate = None

    metrics = {
        "timestamp": ts,
        "mode": "full_auto",
        "diagnostic_review_enabled": bool(args.revisione_diagnostica),
        "dataset": {
            "rows": int(len(df)),
            "test_size": args.quota_test,
            "seed": args.seed,
            "augmentation": augmentation_info,
        },
        "optimization": {
            "grid_cv_folds": args.fold_grid_cv,
            "department_best_params": dep_best_params,
            "sentiment_best_params": sent_best_params,
            "department_grid_top": dep_grid_top,
            "sentiment_grid_top": sent_grid_top,
        },
        "thresholds": {
            "department": dep_thresholds,
            "sentiment": sent_thresholds,
            "quantile": args.quantile_soglia,
            "floor": args.minimo_soglia,
            "note": "Solo diagnostica, la modalità automatica non blocca le predizioni",
        },
        "test_split": {
            "department": dep_metrics,
            "sentiment": sent_metrics,
            "coverage": coverage,
            "needs_review_rate": needs_review_rate,
            "priority_distribution": test_pred_df["priority"].value_counts(normalize=True).to_dict(),
            "sla_summary": test_sla_df.to_dict(orient="records"),
        },
        "robustness": {
            "multi_seed": multi_seed,
            "cross_validation": {
                "department_f1_macro": {
                    "mean": float(dep_cv.mean()),
                    "std": float(dep_cv.std(ddof=0)),
                    "folds": [float(v) for v in dep_cv],
                },
                "sentiment_f1_macro": {
                    "mean": float(sent_cv.mean()),
                    "std": float(sent_cv.std(ddof=0)),
                    "folds": [float(v) for v in sent_cv],
                },
            },
        },
        "benchmarks": benchmark_results,
        "artifacts": {
            "department_model": f"{args.cartella_modelli}/department_model.joblib",
            "sentiment_model": f"{args.cartella_modelli}/sentiment_model.joblib",
            "thresholds": f"{args.cartella_output}/thresholds.json",
            "test_predictions": test_pred_path,
            "confusion_department": f"{args.cartella_output}/confusion_department.png",
            "confusion_sentiment": f"{args.cartella_output}/confusion_sentiment.png",
        },
    }

    metrics_path = f"{args.cartella_output}/metrics.json"
    thresholds_path = f"{args.cartella_output}/thresholds.json"

    salva_json(metrics, metrics_path)
    salva_json(
        {
            "timestamp": ts,
            "department": dep_thresholds,
            "sentiment": sent_thresholds,
            "diagnostic_only": True,
        },
        thresholds_path,
    )

    sla_path = f"{args.cartella_output}/sla_test_summary_{ts}.json"
    salva_json({"rows": test_sla_df.to_dict(orient="records")}, sla_path)

    print("[OK] Addestramento e valutazione completati.")
    print(
        f"[METRICHE] Reparto accuratezza={dep_metrics['accuracy']:.4f} | "
        f"F1 macro={dep_metrics['f1_macro']:.4f}"
    )
    print(
        f"[METRICHE] Sentiment accuratezza={sent_metrics['accuracy']:.4f} | "
        f"F1 macro={sent_metrics['f1_macro']:.4f}"
    )
    if args.revisione_diagnostica:
        print(f"[DIAGNOSTICA] copertura={coverage:.4f} | tasso_controllo_umano={needs_review_rate:.4f}")
    print(f"[ROBUSTEZZA] media_F1_reparto_multi_seed={multi_seed['department_f1_macro']['mean']:.4f}")
    print(f"[ROBUSTEZZA] media_F1_sentiment_multi_seed={multi_seed['sentiment_f1_macro']['mean']:.4f}")
    print(f"[SALVATO] {metrics_path}")
    print(f"[SALVATO] {thresholds_path}")
    print(f"[SALVATO] {test_pred_path}")


if __name__ == "__main__":
    esegui_addestramento_e_valutazione()
