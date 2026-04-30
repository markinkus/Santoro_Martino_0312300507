from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from advanced_light_models import (
    SplitConformal,
    StackedTextEnsemble,
    errore_atteso_calibrazione,
    punteggio_brier_multiclasse,
    sonda_calibrazione_logistica_word,
)
from utils_ops import (
    DEFAULT_THRESHOLDS,
    applica_campi_operativi,
    applica_guardrail_sentiment_df,
    applica_reparti_impattati_df,
    carica_json,
    deriva_soglie_per_classe,
    garantisci_cartella,
    garantisci_cartella_padre,
    normalizza_mappa_soglie,
    salva_json,
    simula_sla,
)
from utils_text import normalizza_testo, unisci_titolo_corpo


def _prepara_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizza il testo e crea la colonna unica usata dalla pipeline advanced."""
    out = df.copy()
    out["text"] = out.apply(
        lambda r: normalizza_testo(unisci_titolo_corpo(str(r.get("title", "")), str(r.get("body", "")))),
        axis=1,
    )
    return out


def _calcola_metriche_task(y_true: np.ndarray, y_pred: np.ndarray, etichette: list[str]) -> dict:
    """Calcola metriche di classificazione e recall per classe."""
    report = classification_report(y_true, y_pred, labels=etichette, output_dict=True, zero_division=0)
    recall_by_class = {c: float(report.get(c, {}).get("recall", 0.0)) for c in etichette}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "recall_by_class": recall_by_class,
        "report": report,
    }


def _salva_matrice_confusione(cm: np.ndarray, etichette: list[str], percorso_output: str, titolo: str) -> None:
    """Salva una confusion matrix come PNG per il report advanced."""
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title(titolo)
    plt.xticks(range(len(etichette)), etichette, rotation=45, ha="right")
    plt.yticks(range(len(etichette)), etichette)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    garantisci_cartella_padre(percorso_output)
    fig.savefig(percorso_output, dpi=200)
    plt.close(fig)


def _costruisci_dataframe_predizioni(
    df: pd.DataFrame,
    dep_pred: np.ndarray,
    sent_pred: np.ndarray,
    dep_proba: np.ndarray,
    sent_proba: np.ndarray,
    classi_reparto: list[str],
    classi_sentiment: list[str],
    conf_sets: list[list[str]] | None = None,
) -> pd.DataFrame:
    """Trasforma predizioni e probabilità in un DataFrame leggibile dal runtime."""
    out = df.copy()
    out["pred_department"] = dep_pred
    out["pred_sentiment"] = sent_pred
    out["department_confidence"] = dep_proba.max(axis=1)
    out["sentiment_confidence"] = sent_proba.max(axis=1)

    for idx, c in enumerate(classi_reparto):
        out[f"proba_department_{c}"] = dep_proba[:, idx]
    for idx, c in enumerate(classi_sentiment):
        out[f"proba_sentiment_{c}"] = sent_proba[:, idx]

    if conf_sets is not None:
        out["department_conformal_set"] = ["|".join(s) for s in conf_sets]
        out["department_conformal_size"] = [len(s) for s in conf_sets]

    return out


def _valuta_split(
    df: pd.DataFrame,
    modello_reparto: StackedTextEnsemble,
    modello_sentiment: StackedTextEnsemble,
    soglie_reparto: dict[str, float],
    soglie_sentiment: dict[str, float],
    modello_conformal: SplitConformal | None,
    revisione_diagnostica: bool,
) -> tuple[dict, pd.DataFrame]:
    """Valuta un singolo split advanced e produce metriche più DataFrame predizioni."""
    x = df["text"].values
    y_dep = df["department"].values
    y_sent = df["sentiment"].values

    dep_classes = [str(c) for c in modello_reparto.classes_]
    sent_classes = [str(c) for c in modello_sentiment.classes_]

    dep_proba = modello_reparto.predict_proba(x)
    sent_proba = modello_sentiment.predict_proba(x)
    dep_pred = np.array([dep_classes[i] for i in np.argmax(dep_proba, axis=1)], dtype=object)
    sent_pred = np.array([sent_classes[i] for i in np.argmax(sent_proba, axis=1)], dtype=object)

    conf_sets = modello_conformal.predict_sets(dep_proba) if modello_conformal else None
    conf_stats = (
        modello_conformal.evaluate(y_dep, conf_sets)
        if modello_conformal
        else {"coverage": None, "avg_set_size": None}
    )

    out_df = _costruisci_dataframe_predizioni(
        df=df,
        dep_pred=dep_pred,
        sent_pred=sent_pred,
        dep_proba=dep_proba,
        sent_proba=sent_proba,
        classi_reparto=dep_classes,
        classi_sentiment=sent_classes,
        conf_sets=conf_sets,
    )
    out_df = applica_guardrail_sentiment_df(out_df)
    out_df = applica_reparti_impattati_df(out_df)
    out_df = applica_campi_operativi(
        out_df,
        soglie_reparto=soglie_reparto,
        soglie_sentiment=soglie_sentiment,
        revisione_diagnostica=revisione_diagnostica,
    )

    dep_pred_runtime = out_df["pred_department"].values
    sent_pred_runtime = out_df["pred_sentiment"].values

    result = {
        "rows": int(len(df)),
        "department": _calcola_metriche_task(y_dep, dep_pred_runtime, etichette=dep_classes),
        "sentiment": _calcola_metriche_task(y_sent, sent_pred_runtime, etichette=sent_classes),
        "calibration": {
            "department": {
                "ece": errore_atteso_calibrazione(y_dep, dep_proba, classi=dep_classes, n_bins=15),
                "brier": punteggio_brier_multiclasse(y_dep, dep_proba, classi=dep_classes),
            },
            "sentiment": {
                "ece": errore_atteso_calibrazione(y_sent, sent_proba, classi=sent_classes, n_bins=15),
                "brier": punteggio_brier_multiclasse(y_sent, sent_proba, classi=sent_classes),
            },
        },
        "conformal_department": conf_stats,
        "priority_distribution": out_df["priority"].value_counts(normalize=True).to_dict() if len(out_df) else {},
    }

    if revisione_diagnostica:
        result["coverage"] = float(1.0 - out_df["needs_review_diag"].mean()) if len(out_df) else 0.0
        result["needs_review_rate"] = float(out_df["needs_review_diag"].mean()) if len(out_df) else 0.0
    else:
        result["coverage"] = None
        result["needs_review_rate"] = None

    result["sla_summary"] = simula_sla(out_df).to_dict(orient="records")
    return result, out_df


def _calcola_delta_benchmark_vs_baseline(adv_metrics: dict, baseline_metrics: dict | None) -> dict:
    """Confronta i benchmark advanced con quelli baseline, se disponibili."""
    if not baseline_metrics:
        return {}

    out: dict[str, dict] = {}
    base_bench = baseline_metrics.get("benchmarks", {})
    adv_bench = adv_metrics.get("benchmarks", {})

    for name, adv_row in adv_bench.items():
        base_row = base_bench.get(name)
        if not base_row:
            continue
        out[name] = {
            "department_f1_delta": float(adv_row["department"]["f1_macro"] - base_row["department"]["f1_macro"]),
            "sentiment_f1_delta": float(adv_row["sentiment"]["f1_macro"] - base_row["sentiment"]["f1_macro"]),
            "department_ece_delta": float(
                adv_row["calibration"]["department"]["ece"] - base_row.get("calibration", {}).get("department", {}).get("ece", 0.0)
            ),
            "sentiment_ece_delta": float(
                adv_row["calibration"]["sentiment"]["ece"] - base_row.get("calibration", {}).get("sentiment", {}).get("ece", 0.0)
            ),
        }

    return out


def esegui_addestramento_avanzato() -> None:
    """Addestra la pipeline advanced, salva artefatti e misura i benchmark."""
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
    parser.add_argument("--quota_meta", "--meta_size", dest="quota_meta", type=float, default=0.25, help="Quota del pool di addestramento usata come meta/calibrazione")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha_conformal", "--conformal_alpha", dest="alpha_conformal", type=float, default=0.10)
    parser.add_argument("--metodo_conformal", "--conformal_method", dest="metodo_conformal", type=str, default="aps", choices=["aps", "threshold"])
    parser.add_argument("--quantile_soglia", "--threshold_quantile", dest="quantile_soglia", type=float, default=0.08)
    parser.add_argument("--minimo_soglia", "--threshold_floor", dest="minimo_soglia", type=float, default=0.45)
    parser.add_argument("--revisione_diagnostica", "--diagnostic_review", dest="revisione_diagnostica", action="store_true")
    parser.add_argument("--cartella_modelli", "--models_dir", dest="cartella_modelli", type=str, default="models/advanced")
    parser.add_argument("--cartella_output", "--outputs_dir", dest="cartella_output", type=str, default="outputs/advanced")
    args = parser.parse_args()

    garantisci_cartella(args.cartella_modelli)
    garantisci_cartella(args.cartella_output)

    df = _prepara_dataframe(pd.read_csv(args.percorso_dati))

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

    x_pool, x_test, y_dep_pool, y_dep_test, y_sent_pool, y_sent_test, y_joint_pool, _ = train_test_split(
        x,
        y_dep,
        y_sent,
        y_joint,
        test_size=args.quota_test,
        random_state=args.seed,
        stratify=y_joint,
    )

    x_train, x_meta, y_dep_train, y_dep_meta, y_sent_train, y_sent_meta, _, _ = train_test_split(
        x_pool,
        y_dep_pool,
        y_sent_pool,
        y_joint_pool,
        test_size=args.quota_meta,
        random_state=args.seed + 1,
        stratify=y_joint_pool,
    )

    print("[INFO] Sonda di calibrazione (reparto): sigmoidale vs isotonica")
    dep_cal_method, dep_probe = sonda_calibrazione_logistica_word(
        compito="department",
        x_train=x_train,
        y_train=np.array(y_dep_train),
        x_val=x_meta,
        y_val=np.array(y_dep_meta),
        seed=args.seed,
    )

    print("[INFO] Sonda di calibrazione (sentiment): sigmoidale vs isotonica")
    sent_cal_method, sent_probe = sonda_calibrazione_logistica_word(
        compito="sentiment",
        x_train=x_train,
        y_train=np.array(y_sent_train),
        x_val=x_meta,
        y_val=np.array(y_sent_meta),
        seed=args.seed,
    )

    print(f"[INFO] Metodi di calibrazione scelti -> reparto={dep_cal_method} | sentiment={sent_cal_method}")

    dep_model = StackedTextEnsemble(compito="department", metodo_calibrazione=dep_cal_method, seed=args.seed)
    sent_model = StackedTextEnsemble(compito="sentiment", metodo_calibrazione=sent_cal_method, seed=args.seed)

    print("[INFO] Addestramento ensemble stacked per reparto...")
    dep_model.fit(
        x_train=np.array(x_train),
        y_train=np.array(y_dep_train),
        x_meta=np.array(x_meta),
        y_meta=np.array(y_dep_meta),
    )
    print("[INFO] Addestramento ensemble stacked per sentiment...")
    sent_model.fit(
        x_train=np.array(x_train),
        y_train=np.array(y_sent_train),
        x_meta=np.array(x_meta),
        y_meta=np.array(y_sent_meta),
    )

    dep_classes = [str(c) for c in dep_model.classes_]
    sent_classes = [str(c) for c in sent_model.classes_]

    # Threshold derivation on meta split
    dep_meta_proba = dep_model.predict_proba(np.array(x_meta))
    sent_meta_proba = sent_model.predict_proba(np.array(x_meta))
    dep_thresholds = normalizza_mappa_soglie(
        deriva_soglie_per_classe(
            y_true=np.array(y_dep_meta),
            y_proba=dep_meta_proba,
            classi=dep_classes,
            quantile=args.quantile_soglia,
            floor=args.minimo_soglia,
        ),
        classi=dep_classes,
        fallback=args.minimo_soglia,
    )
    sent_thresholds = normalizza_mappa_soglie(
        deriva_soglie_per_classe(
            y_true=np.array(y_sent_meta),
            y_proba=sent_meta_proba,
            classi=sent_classes,
            quantile=args.quantile_soglia,
            floor=args.minimo_soglia,
        ),
        classi=sent_classes,
        fallback=args.minimo_soglia,
    )

    conformal = SplitConformal(alpha=args.alpha_conformal, method=args.metodo_conformal).fit(
        y_true=np.array(y_dep_meta),
        y_proba=dep_meta_proba,
        classi=dep_classes,
    )

    test_df = pd.DataFrame(
        {
            "text": x_test,
            "department": y_dep_test,
            "sentiment": y_sent_test,
        }
    )
    test_result, test_pred_df = _valuta_split(
        df=test_df,
        modello_reparto=dep_model,
        modello_sentiment=sent_model,
        soglie_reparto=dep_thresholds,
        soglie_sentiment=sent_thresholds,
        modello_conformal=conformal,
        revisione_diagnostica=args.revisione_diagnostica,
    )

    dep_cm = confusion_matrix(
        test_pred_df["department"].values,
        test_pred_df["pred_department"].values,
        labels=dep_classes,
    )
    sent_cm = confusion_matrix(
        test_pred_df["sentiment"].values,
        test_pred_df["pred_sentiment"].values,
        labels=sent_classes,
    )
    _salva_matrice_confusione(
        dep_cm,
        dep_classes,
        f"{args.cartella_output}/confusion_department_advanced.png",
        "Matrice di confusione avanzata - Reparto",
    )
    _salva_matrice_confusione(
        sent_cm,
        sent_classes,
        f"{args.cartella_output}/confusion_sentiment_advanced.png",
        "Matrice di confusione avanzata - Sentiment",
    )

    # benchmark sets
    bench_results: dict[str, dict] = {}
    bench_dir = Path(args.cartella_benchmark)
    if bench_dir.exists():
        for csv_path in sorted(bench_dir.glob("*.csv")):
            bdf = _prepara_dataframe(pd.read_csv(csv_path))
            bres, _ = _valuta_split(
                df=bdf[["text", "department", "sentiment"]].copy(),
                modello_reparto=dep_model,
                modello_sentiment=sent_model,
                soglie_reparto=dep_thresholds,
                soglie_sentiment=sent_thresholds,
                modello_conformal=conformal,
                revisione_diagnostica=args.revisione_diagnostica,
            )
            bench_results[csv_path.name] = bres

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    test_pred_path = f"{args.cartella_output}/test_predictions_advanced_{ts}.csv"
    garantisci_cartella_padre(test_pred_path)
    test_pred_df.to_csv(test_pred_path, index=False)

    joblib.dump(dep_model, f"{args.cartella_modelli}/department_ensemble_advanced.joblib")
    joblib.dump(sent_model, f"{args.cartella_modelli}/sentiment_ensemble_advanced.joblib")
    joblib.dump(conformal, f"{args.cartella_modelli}/department_conformal_advanced.joblib")

    baseline_metrics = carica_json("outputs/metrics.json", predefinito=None)
    baseline_summary = {
        "test_department_f1": baseline_metrics.get("test_split", {}).get("department", {}).get("f1_macro")
        if baseline_metrics
        else None,
        "test_sentiment_f1": baseline_metrics.get("test_split", {}).get("sentiment", {}).get("f1_macro")
        if baseline_metrics
        else None,
    }

    payload = {
        "timestamp": ts,
        "mode": "advanced_lightweight_ensemble",
        "dataset": {
            "rows": int(len(df)),
            "test_size": float(args.quota_test),
            "meta_size_on_train_pool": float(args.quota_meta),
            "seed": int(args.seed),
            "augmentation": augmentation_info,
        },
        "components": {
            "base_models": ["tfidf_char_logreg_calibrated", "tfidf_word_linearsvc_calibrated", "fasttext_supervised"],
            "stacking_meta": "logistic_regression",
            "sentiment_hardening": {
                "class_weight": {"neg": 1.35, "pos": 1.0},
                "contrast_negation_tokens": True,
            },
            "conformal_prediction": {
                "enabled": True,
                "method": str(args.metodo_conformal),
                "alpha": float(args.alpha_conformal),
                "threshold": float(conformal.threshold) if conformal.threshold is not None else None,
                "qhat": float(conformal.qhat if conformal.qhat is not None else 0.0),
            },
        },
        "calibration_probe": {
            "department": {"chosen": dep_cal_method, "candidates": dep_probe},
            "sentiment": {"chosen": sent_cal_method, "candidates": sent_probe},
        },
        "thresholds": {
            "department": dep_thresholds,
            "sentiment": sent_thresholds,
            "quantile": float(args.quantile_soglia),
            "floor": float(args.minimo_soglia),
            "note": "Solo diagnostica; la modalità automatica non blocca le predizioni",
        },
        "test_split": test_result,
        "benchmarks": bench_results,
        "comparison_vs_baseline": {
            "baseline_reference": baseline_summary,
            "test_f1_delta": {
                "department": float(test_result["department"]["f1_macro"] - baseline_summary["test_department_f1"])
                if baseline_summary["test_department_f1"] is not None
                else None,
                "sentiment": float(test_result["sentiment"]["f1_macro"] - baseline_summary["test_sentiment_f1"])
                if baseline_summary["test_sentiment_f1"] is not None
                else None,
            },
            "benchmark_delta": _calcola_delta_benchmark_vs_baseline(
                adv_metrics={"benchmarks": bench_results},
                baseline_metrics=baseline_metrics,
            ),
        },
        "artifacts": {
            "department_model": f"{args.cartella_modelli}/department_ensemble_advanced.joblib",
            "sentiment_model": f"{args.cartella_modelli}/sentiment_ensemble_advanced.joblib",
            "conformal_model": f"{args.cartella_modelli}/department_conformal_advanced.joblib",
            "test_predictions": test_pred_path,
            "confusion_department": f"{args.cartella_output}/confusion_department_advanced.png",
            "confusion_sentiment": f"{args.cartella_output}/confusion_sentiment_advanced.png",
        },
    }

    metrics_path = f"{args.cartella_output}/metrics_advanced.json"
    thresholds_path = f"{args.cartella_output}/thresholds_advanced.json"
    salva_json(payload, metrics_path)
    salva_json(
        {
            "timestamp": ts,
            "department": dep_thresholds,
            "sentiment": sent_thresholds,
            "diagnostic_only": True,
        },
        thresholds_path,
    )

    print("[OK] Addestramento avanzato completato.")
    print(
        f"[METRICHE] Reparto accuratezza={test_result['department']['accuracy']:.4f} | "
        f"F1 macro={test_result['department']['f1_macro']:.4f}"
    )
    print(
        f"[METRICHE] Sentiment accuratezza={test_result['sentiment']['accuracy']:.4f} | "
        f"F1 macro={test_result['sentiment']['f1_macro']:.4f}"
    )
    print(
        "[CALIBRAZIONE] ECE reparto/sentiment="
        f"{test_result['calibration']['department']['ece']:.4f}/"
        f"{test_result['calibration']['sentiment']['ece']:.4f}"
    )
    print(
        "[CONFORMAL] copertura/dimensione_media_set="
        f"{test_result['conformal_department']['coverage']:.4f}/"
        f"{test_result['conformal_department']['avg_set_size']:.4f}"
    )
    print(f"[SALVATO] {metrics_path}")
    print(f"[SALVATO] {thresholds_path}")
    print(f"[SALVATO] {test_pred_path}")


if __name__ == "__main__":
    esegui_addestramento_avanzato()
