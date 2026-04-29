import json
import importlib.util
import io
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from utils_ops import (
    DEFAULT_THRESHOLDS,
    applica_campi_operativi,
    applica_guardrail_sentiment_df,
    applica_reparti_impattati_df,
    carica_json,
    contributi_principali_token,
    normalizza_mappa_soglie,
    simula_sla,
)
from utils_text import normalizza_testo, unisci_titolo_corpo

st.set_page_config(page_title="Smistamento Recensioni Hotel", layout="wide")

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
FILE_REQUISITI_BASE = "requirements.txt"
FILE_REQUISITI_TRANSFORMER = "requisiti-transformer.txt"

DIPENDENZE_ADVANCED = {
    "fasttext": "fasttext-wheel",
}

DIPENDENZE_TRANSFORMER = {
    "torch": "torch",
    "transformers": "transformers",
    "sentence_transformers": "sentence-transformers",
}

PROFILI_ACTIVE_LEARNING = {
    "active_learning_oracle": "Oracle",
    "active_learning_v2_no_replay": "V2 no replay",
    "active_learning_v2_replay": "V2 replay",
}


def percorso_root(percorso_relativo: str) -> Path:
    """Restituisce un percorso assoluto relativo alla radice del progetto."""
    return ROOT_DIR / percorso_relativo


def moduli_mancanti(moduli: dict[str, str]) -> list[str]:
    """Restituisce i pacchetti richiesti ma non importabili nell'ambiente corrente."""
    mancanti: list[str] = []
    for nome_modulo, nome_pacchetto in moduli.items():
        if importlib.util.find_spec(nome_modulo) is None:
            mancanti.append(nome_pacchetto)
    return mancanti


def percorsi_profilo(nome_profilo: str) -> tuple[str, str, str, str | None]:
    """Restituisce i percorsi di modelli e soglie associati a un profilo."""
    if nome_profilo in {"baseline_pure", "baseline_hardened"}:
        suffisso = "pure" if nome_profilo.endswith("_pure") else "hardened"
        return (
            str(percorso_root(f"models/baseline_{suffisso}/department_model.joblib")),
            str(percorso_root(f"models/baseline_{suffisso}/sentiment_model.joblib")),
            str(percorso_root(f"outputs/baseline_{suffisso}/thresholds.json")),
            None,
        )
    if nome_profilo in {"advanced_aps_pure", "advanced_aps_hardened"}:
        suffisso = "pure" if nome_profilo.endswith("_pure") else "hardened"
        return (
            str(percorso_root(f"models/advanced_aps_{suffisso}/department_ensemble_advanced.joblib")),
            str(percorso_root(f"models/advanced_aps_{suffisso}/sentiment_ensemble_advanced.joblib")),
            str(percorso_root(f"outputs/advanced_aps_{suffisso}/thresholds_advanced.json")),
            str(percorso_root(f"models/advanced_aps_{suffisso}/department_conformal_advanced.joblib")),
        )
    if nome_profilo == "advanced_aps":
        return (
            str(percorso_root("models/advanced_aps/department_ensemble_advanced.joblib")),
            str(percorso_root("models/advanced_aps/sentiment_ensemble_advanced.joblib")),
            str(percorso_root("outputs/advanced_aps/thresholds_advanced.json")),
            str(percorso_root("models/advanced_aps/department_conformal_advanced.joblib")),
        )
    if nome_profilo in PROFILI_ACTIVE_LEARNING:
        return (
            str(percorso_root(f"models/{nome_profilo}/department_model.joblib")),
            str(percorso_root(f"models/{nome_profilo}/sentiment_model.joblib")),
            str(percorso_root(f"outputs/{nome_profilo}/thresholds.json")),
            None,
        )
    return (
        str(percorso_root("models/department_model.joblib")),
        str(percorso_root("models/sentiment_model.joblib")),
        str(percorso_root("outputs/thresholds.json")),
        None,
    )


def profili_disponibili() -> list[str]:
    """Elenca i profili modello realmente disponibili sul filesystem."""
    out: list[str] = []
    profili_espliciti = [
        "baseline_pure",
        "baseline_hardened",
        "advanced_aps_pure",
        "advanced_aps_hardened",
        *PROFILI_ACTIVE_LEARNING.keys(),
    ]
    for p in profili_espliciti:
        dep_path, sent_path, _, _ = percorsi_profilo(p)
        if Path(dep_path).exists() and Path(sent_path).exists():
            out.append(p)
    if out:
        return out
    for p in ["baseline", "advanced_aps"]:
        dep_path, sent_path, _, _ = percorsi_profilo(p)
        if Path(dep_path).exists() and Path(sent_path).exists():
            out.append(p)
    return out


def descrizione_profilo(nome_profilo: str | None) -> str:
    """Descrive in modo breve la configurazione del profilo selezionato."""
    descrizioni = {
        "baseline_pure": "Modello base pulito: casi critici usati solo per valutazione, senza aggiunte al training.",
        "baseline_hardened": "Modello base irrobustito: usa esempi critici nel training con repeat=2.",
        "advanced_aps_pure": "Modello avanzato APS pulito: ensemble e insiemi di reparti plausibili, senza aggiunta di casi critici nel training.",
        "advanced_aps_hardened": "Modello avanzato APS irrobustito: ensemble, APS e aggiunta di casi critici con repeat=2.",
        "active_learning_oracle": "Active learning oracle: riaddestramento da coda etichettata ideale.",
        "active_learning_v2_no_replay": "Active learning senza replay: riaddestramento solo sui nuovi casi etichettati.",
        "active_learning_v2_replay": "Active learning con replay: riaddestramento con nuovi casi e dataset base.",
        "baseline": "Profilo storico: nel pacchetto aggiornato equivale al modello base irrobustito.",
        "advanced_aps": "Profilo storico: nel pacchetto aggiornato equivale al modello avanzato APS irrobustito.",
    }
    return descrizioni.get(str(nome_profilo), "")


def etichetta_profilo(nome_profilo: str | None) -> str:
    """Restituisce un'etichetta leggibile per i profili tecnici salvati su disco."""
    etichette = {
        "baseline_pure": "Base pulito",
        "baseline_hardened": "Base irrobustito",
        "advanced_aps_pure": "Avanzato APS pulito",
        "advanced_aps_hardened": "Avanzato APS irrobustito",
        "active_learning_oracle": "Active learning oracle",
        "active_learning_v2_no_replay": "Active learning senza replay",
        "active_learning_v2_replay": "Active learning con replay",
        "baseline": "Base storico",
        "advanced_aps": "Avanzato APS storico",
    }
    return etichette.get(str(nome_profilo), str(nome_profilo))


def etichetta_priorita(priorita: str | None) -> str:
    """Traduce le priorità operative tecniche in etichette leggibili in italiano."""
    etichette = {
        "LOW": "BASSA",
        "MEDIUM": "MEDIA",
        "HIGH": "ALTA",
        "URGENT": "URGENTE",
    }
    return etichette.get(str(priorita), str(priorita))


def etichetta_booleano(valore) -> str:
    """Mostra i flag tecnici come SÌ/NO nelle tabelle della dashboard."""
    if pd.isna(valore):
        return "NO"
    if isinstance(valore, str):
        return "SÌ" if valore.strip().lower() in {"true", "1", "si", "sì", "yes"} else "NO"
    return "SÌ" if bool(valore) else "NO"


def rinomina_colonne_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    """Mapping dei nomi delle colonne in dashboard."""
    out = df.copy()
    if "priority" in out.columns:
        out["priority"] = out["priority"].map(etichetta_priorita)
    for colonna_bool in [
        "cross_department_signal",
        "sentiment_guardrail_applied",
        "sentiment_hazard_flag",
        "needs_review_diag",
    ]:
        if colonna_bool in out.columns:
            out[colonna_bool] = out[colonna_bool].map(etichetta_booleano)
    for colonna_lista in ["impacted_departments", "department_conformal_set"]:
        if colonna_lista in out.columns:
            out[colonna_lista] = out[colonna_lista].astype(str).str.replace("|", ", ", regex=False)
    if "impacted_departments_reason" in out.columns:
        out["impacted_departments_reason"] = (
            out["impacted_departments_reason"]
            .astype(str)
            .str.replace(":model", ":modello", regex=False)
            .str.replace(";", " | ", regex=False)
        )
    mappa = {
        "id": "ID",
        "title": "Titolo",
        "body": "Testo recensione",
        "text": "Testo completo",
        "pred_department": "Reparto previsto",
        "department_confidence": "Confidenza reparto",
        "pred_sentiment": "Sentiment previsto",
        "sentiment_confidence": "Confidenza sentiment",
        "priority": "Priorità",
        "risk_score": "Punteggio di rischio",
        "impacted_departments": "Reparti coinvolti",
        "impacted_departments_count": "Numero reparti coinvolti",
        "impacted_departments_reason": "Motivo reparti coinvolti",
        "cross_department_signal": "Segnale multi-reparto",
        "sentiment_guardrail_applied": "Correzione applicata",
        "sentiment_guardrail_score": "Punteggio segnali negativi",
        "sentiment_hazard_score": "Punteggio igienico-sanitario",
        "sentiment_hazard_flag": "Criticità igienico-sanitaria",
        "sentiment_guardrail_reason": "Motivo correzione",
        "needs_review_diag": "Controllo umano consigliato",
        "department_conformal_set": "Reparti plausibili APS",
        "department_conformal_size": "Numero reparti plausibili",
        "explain_dep_top_tokens": "Parole influenti reparto",
        "explain_sent_top_tokens": "Parole influenti sentiment",
        "department": "Reparto",
        "volume": "Volume",
        "window_hours": "Finestra ore",
        "capacity_tickets": "Capacità ticket",
        "backlog_estimated": "Backlog stimato",
        "sla_target_hours": "Target SLA ore",
        "estimated_avg_response_hours": "Risposta media stimata ore",
        "estimated_sla_hit_rate": "Rispetto SLA stimato",
    }
    colonne = {}
    for colonna in out.columns:
        if colonna in mappa:
            colonne[colonna] = mappa[colonna]
        elif colonna.startswith("proba_department_"):
            colonne[colonna] = f"Probabilità reparto {colonna.replace('proba_department_', '')}"
        elif colonna.startswith("proba_sentiment_"):
            colonne[colonna] = f"Probabilità sentiment {colonna.replace('proba_sentiment_', '')}"
    return out.rename(columns=colonne)


@st.cache_resource(show_spinner="Caricamento del profilo modello in corso, potrebbe volerci qualche istante...")
def carica_modelli_e_soglie(nome_profilo: str):
    """Carica modelli, soglie e artefatti conformal per il profilo richiesto."""
    dep_path, sent_path, thr_path, conformal_path = percorsi_profilo(nome_profilo)

    if not Path(dep_path).exists() or not Path(sent_path).exists():
        raise FileNotFoundError(
            f"Modelli non trovati per profilo={nome_profilo}. "
            f"Attesi: {dep_path} e {sent_path}"
        )

    dep_model = joblib.load(dep_path)
    sent_model = joblib.load(sent_path)
    conformal_model = joblib.load(conformal_path) if conformal_path and Path(conformal_path).exists() else None

    payload = carica_json(thr_path, predefinito=DEFAULT_THRESHOLDS)

    dep_classes = ottieni_classi_modello(dep_model)
    sent_classes = ottieni_classi_modello(sent_model)

    dep_thr = normalizza_mappa_soglie(payload.get("department", {}), dep_classes, fallback=0.55)
    sent_thr = normalizza_mappa_soglie(payload.get("sentiment", {}), sent_classes, fallback=0.55)

    return dep_model, sent_model, dep_thr, sent_thr, dep_classes, sent_classes, conformal_model


def ottieni_classi_modello(modello):
    """Recupera le classi esposte da un modello sklearn o da una pipeline."""
    model = modello
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        clf = model.named_steps["clf"]
        if hasattr(clf, "classes_"):
            return [str(c) for c in clf.classes_]
    if hasattr(model, "classes_"):
        return [str(c) for c in model.classes_]
    raise ValueError("Impossibile recuperare le classi dal modello.")


def supporta_spiegabilita_per_token(modello) -> bool:
    """Indica se il modello espone i componenti necessari per la spiegabilità token-level."""
    return hasattr(modello, "named_steps") and "tfidf" in modello.named_steps and "clf" in modello.named_steps


def predici_recensione_singola(
    modello_reparto,
    modello_sentiment,
    soglie_reparto,
    soglie_sentiment,
    classi_reparto,
    classi_sentiment,
    modello_conformal,
    titolo: str,
    corpo: str,
    top_k: int,
    revisione_diagnostica: bool,
):
    """Esegue la predizione completa su una singola recensione."""
    text = normalizza_testo(unisci_titolo_corpo(titolo, corpo))

    dep_pred = modello_reparto.predict([text])[0]
    dep_proba = modello_reparto.predict_proba([text])[0]
    dep_conf = float(dep_proba.max())
    dep_probs = {classi_reparto[i]: float(dep_proba[i]) for i in range(len(classi_reparto))}

    sent_pred = modello_sentiment.predict([text])[0]
    sent_proba = modello_sentiment.predict_proba([text])[0]
    sent_conf = float(sent_proba.max())
    sent_probs = {classi_sentiment[i]: float(sent_proba[i]) for i in range(len(classi_sentiment))}

    df = pd.DataFrame(
        [
            {
                "text": text,
                "pred_department": dep_pred,
                "pred_sentiment": sent_pred,
                "department_confidence": dep_conf,
                "sentiment_confidence": sent_conf,
                **{f"proba_department_{k}": v for k, v in dep_probs.items()},
                **{f"proba_sentiment_{k}": v for k, v in sent_probs.items()},
            }
        ]
    )

    df = applica_guardrail_sentiment_df(df)
    df = applica_reparti_impattati_df(df)

    df = applica_campi_operativi(
        df,
        soglie_reparto=soglie_reparto,
        soglie_sentiment=soglie_sentiment,
        revisione_diagnostica=revisione_diagnostica,
    )

    row = df.iloc[0]
    dep_pred = str(row["pred_department"])
    sent_pred = str(row["pred_sentiment"])
    dep_conf = float(row["department_confidence"])
    sent_conf = float(row["sentiment_confidence"])
    dep_probs = {
        classi_reparto[i]: float(row[f"proba_department_{classi_reparto[i]}"])
        for i in range(len(classi_reparto))
    }
    sent_probs = {
        classi_sentiment[i]: float(row[f"proba_sentiment_{classi_sentiment[i]}"])
        for i in range(len(classi_sentiment))
    }

    dep_exp = (
        contributi_principali_token(modello_reparto, text, top_k=top_k)
        if supporta_spiegabilita_per_token(modello_reparto)
        else []
    )
    sent_exp = (
        contributi_principali_token(modello_sentiment, text, top_k=top_k)
        if supporta_spiegabilita_per_token(modello_sentiment)
        else []
    )

    out = {
        "department": dep_pred,
        "department_confidence": dep_conf,
        "department_probs": dep_probs,
        "sentiment": sent_pred,
        "sentiment_confidence": sent_conf,
        "sentiment_probs": sent_probs,
        "priority": str(df.iloc[0]["priority"]),
        "risk_score": float(df.iloc[0]["risk_score"]),
        "dep_explain": dep_exp,
        "sent_explain": sent_exp,
        "sentiment_guardrail_applied": bool(row.get("sentiment_guardrail_applied", False)),
        "sentiment_guardrail_score": int(row.get("sentiment_guardrail_score", 0)),
        "sentiment_hazard_score": int(row.get("sentiment_hazard_score", 0)),
        "sentiment_hazard_flag": bool(row.get("sentiment_hazard_flag", False)),
        "sentiment_guardrail_reason": str(row.get("sentiment_guardrail_reason", "")),
        "impacted_departments": str(row.get("impacted_departments", dep_pred)).split("|"),
        "impacted_departments_count": int(row.get("impacted_departments_count", 1)),
        "cross_department_signal": bool(row.get("cross_department_signal", False)),
    }

    if revisione_diagnostica:
        out["needs_review_diag"] = bool(df.iloc[0]["needs_review_diag"])

    if modello_conformal is not None:
        pred_set = modello_conformal.predict_sets(dep_proba.reshape(1, -1))[0]
        out["department_conformal_set"] = pred_set
        out["department_conformal_size"] = len(pred_set)

    return out


def predici_recensioni_lotto(
    modello_reparto,
    modello_sentiment,
    soglie_reparto,
    soglie_sentiment,
    classi_reparto,
    classi_sentiment,
    modello_conformal,
    df: pd.DataFrame,
    top_k: int,
    revisione_diagnostica: bool,
):
    """Esegue la predizione completa su un CSV di recensioni."""
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
    out_df["department_confidence"] = dep_proba.max(axis=1)

    out_df["pred_sentiment"] = sent_pred
    out_df["sentiment_confidence"] = sent_proba.max(axis=1)

    for i, c in enumerate(classi_reparto):
        out_df[f"proba_department_{c}"] = dep_proba[:, i]

    for i, c in enumerate(classi_sentiment):
        out_df[f"proba_sentiment_{c}"] = sent_proba[:, i]

    out_df = applica_guardrail_sentiment_df(out_df)
    out_df = applica_reparti_impattati_df(out_df)

    out_df = applica_campi_operativi(
        out_df,
        soglie_reparto=soglie_reparto,
        soglie_sentiment=soglie_sentiment,
        revisione_diagnostica=revisione_diagnostica,
    )

    if supporta_spiegabilita_per_token(modello_reparto) and supporta_spiegabilita_per_token(modello_sentiment):
        dep_tokens = []
        sent_tokens = []
        for txt in out_df["text"].values:
            dep_exp = contributi_principali_token(modello_reparto, txt, top_k=top_k)
            sent_exp = contributi_principali_token(modello_sentiment, txt, top_k=top_k)
            dep_tokens.append(", ".join([tok for tok, _ in dep_exp]))
            sent_tokens.append(", ".join([tok for tok, _ in sent_exp]))

        out_df["explain_dep_top_tokens"] = dep_tokens
        out_df["explain_sent_top_tokens"] = sent_tokens

    if modello_conformal is not None:
        dep_sets = modello_conformal.predict_sets(dep_proba)
        out_df["department_conformal_set"] = ["|".join(s) for s in dep_sets]
        out_df["department_conformal_size"] = [len(s) for s in dep_sets]

    if not revisione_diagnostica and "needs_review_diag" in out_df.columns:
        out_df = out_df.drop(columns=["needs_review_diag"])

    return out_df


def esegui_passi_guidati(
    chiave_risultato: str,
    titolo: str,
    passi: list[dict],
    refresh_models: bool = False,
) -> None:
    """Lancia una sequenza di comandi shell e salva i log in sessione Streamlit."""
    logs: list[dict] = []
    success = True

    for step in passi:
        cmd = step["cmd"]
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in step.get("env", {}).items()})
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            env=env,
        )
        logs.append(
            {
                "label": step["label"],
                "cmd": shlex.join(cmd),
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )
        if proc.returncode != 0:
            success = False
            break

    st.session_state[chiave_risultato] = {"title": titolo, "success": success, "logs": logs}
    if refresh_models and success:
        carica_modelli_e_soglie.clear()
    st.rerun()


def mostra_prompt_installazione_dipendenze(
    *,
    chiave_prompt: str,
    chiave_risultato: str,
    titolo: str,
    messaggio: str,
    file_requisiti: str,
    moduli: dict[str, str],
    refresh_models: bool = False,
) -> bool:
    """Mostra un prompt Si/No per installare dipendenze opzionali mancanti."""
    mancanti = moduli_mancanti(moduli)
    if not mancanti:
        return False

    requisiti_path = percorso_root(file_requisiti)
    elenco = ", ".join(sorted(set(mancanti)))

    st.warning(f"{messaggio} Dipendenze mancanti: {elenco}.")
    st.caption(
        f"Vuoi installarle ora da `{file_requisiti}`? L'operazione richiede internet e può durare alcuni minuti. "
        "Al termine la pagina verrà ricaricata automaticamente."
    )

    col_si, col_no = st.columns(2)
    with col_si:
        if st.button("Sì, installa", key=f"{chiave_prompt}_si"):
            esegui_passi_guidati(
                chiave_risultato,
                titolo,
                [
                    {
                        "label": f"Installazione dipendenze da {file_requisiti}",
                        "cmd": [sys.executable, "-m", "pip", "install", "-r", str(requisiti_path)],
                        "env": {"PIP_CACHE_DIR": str(percorso_root(".cache/pip"))},
                    }
                ],
                refresh_models=refresh_models,
            )
    with col_no:
        st.button("No, non ora", key=f"{chiave_prompt}_no")

    mostra_risultato_strumento(chiave_risultato)
    return True


def mostra_risultato_strumento(chiave_risultato: str) -> None:
    """Mostra l'esito di uno strumento guidato con i log di esecuzione."""
    payload = st.session_state.get(chiave_risultato)
    if not payload:
        return

    if payload["success"]:
        st.success(f"{payload['title']}: completato correttamente.")
    else:
        failed = next((log for log in payload["logs"] if log["returncode"] != 0), None)
        suffix = f" Step fallito: {failed['label']}." if failed else ""
        st.error(f"{payload['title']}: esecuzione non completata.{suffix}")

    riepilogo = [
        {
            "step": idx,
            "label": log["label"],
            "exit_code": log["returncode"],
            "comando": log["cmd"],
        }
        for idx, log in enumerate(payload["logs"], start=1)
    ]
    st.dataframe(pd.DataFrame(riepilogo), use_container_width=True, hide_index=True)

    if not payload["success"]:
        for idx, log in enumerate(payload["logs"], start=1):
            st.markdown(f"**Dettagli step {idx}: {log['label']}**")
            if log["stdout"]:
                st.code(log["stdout"], language="text")
            if log["stderr"]:
                st.code(log["stderr"], language="text")


def mostra_confronto_safety() -> None:
    """Mostra in dashboard il report JSON del confronto safety before/after."""
    report_path = percorso_root("outputs/safety_exp/safety_delta_report_r2.json")
    report_md_path = percorso_root("outputs/safety_exp/safety_delta_report_r2.md")
    if not report_path.exists():
        st.info("Report delta casi critici non ancora presente. Usa il pulsante sopra per generarlo.")
        return

    report = carica_json(str(report_path), predefinito={})
    sezioni = report.get("sections", {})
    if not isinstance(sezioni, dict) or not sezioni:
        st.warning("Report delta casi critici presente, ma senza sezioni leggibili.")
        return

    label_sezioni = {
        "test_split": "Test split",
        "reviews_in_distribution.csv": "In distribution",
        "reviews_ambiguous.csv": "Ambiguo",
        "reviews_noisy.csv": "Rumoroso",
        "reviews_colloquial.csv": "Colloquiale",
        "reviews_safety_critical.csv": "Casi critici",
    }

    righe = []
    for nome, dati in sezioni.items():
        prima = dati.get("before", {})
        dopo = dati.get("after", {})
        delta = dati.get("delta_after_minus_before", {})

        def delta_pp(nome_metrica: str) -> float:
            valore = delta.get(nome_metrica, 0.0)
            if valore is None:
                return 0.0
            return 100 * float(valore)

        righe.append(
            {
                "benchmark": label_sezioni.get(nome, nome),
                "department_f1_prima": prima.get("department_f1"),
                "department_f1_dopo": dopo.get("department_f1"),
                "department_f1_delta_pp": delta_pp("department_f1"),
                "sentiment_f1_prima": prima.get("sentiment_f1"),
                "sentiment_f1_dopo": dopo.get("sentiment_f1"),
                "sentiment_f1_delta_pp": delta_pp("sentiment_f1"),
                "recall_neg_delta_pp": delta_pp("sentiment_recall_neg"),
                "coverage_delta_pp": delta_pp("coverage"),
                "needs_review_delta_pp": delta_pp("needs_review_rate"),
            }
        )

    df_delta = pd.DataFrame(righe)
    focus = df_delta[df_delta["benchmark"] == "Casi critici"]
    if not focus.empty:
        r = focus.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Delta reparto F1", f"{r['department_f1_delta_pp']:+.1f} pp")
        c2.metric("Delta sentiment F1", f"{r['sentiment_f1_delta_pp']:+.1f} pp")
        c3.metric("Delta recall neg", f"{r['recall_neg_delta_pp']:+.1f} pp")
        c4.metric("Delta gestione automatica", f"{r['coverage_delta_pp']:+.1f} pp")

    st.caption("Delta calcolati come run irrobustito - run base. Valori positivi indicano una crescita dopo l'irrobustimento.")
    df_delta_display = df_delta.rename(
        columns={
            "benchmark": "Benchmark",
            "department_f1_prima": "Reparto F1 prima",
            "department_f1_dopo": "Reparto F1 dopo",
            "department_f1_delta_pp": "Delta reparto pp",
            "sentiment_f1_prima": "Sentiment F1 prima",
            "sentiment_f1_dopo": "Sentiment F1 dopo",
            "sentiment_f1_delta_pp": "Delta sentiment pp",
            "recall_neg_delta_pp": "Delta recall neg pp",
            "coverage_delta_pp": "Delta gestione automatica pp",
            "needs_review_delta_pp": "Delta controllo umano pp",
        }
    )
    st.dataframe(df_delta_display.round(4), use_container_width=True, hide_index=True)

    chart_df = df_delta.set_index("benchmark")[
        ["department_f1_delta_pp", "sentiment_f1_delta_pp", "coverage_delta_pp"]
    ]
    st.bar_chart(chart_df)

    if report_md_path.exists() and st.checkbox("Mostra un report markdown generato", key="show_safety_report_md"):
        st.code(report_md_path.read_text(encoding="utf-8"), language="markdown")


def mostra_metriche_active_learning() -> None:
    """Mostra una sintesi dei run active learning rigenerabili."""
    righe = []
    for profilo_al, etichetta in PROFILI_ACTIVE_LEARNING.items():
        dep_path, sent_path, _, _ = percorsi_profilo(profilo_al)
        metrics_path = percorso_root(f"outputs/{profilo_al}/metrics.json")
        metriche = carica_json(str(metrics_path), predefinito={}) if metrics_path.exists() else {}
        test = metriche.get("test_split", {})
        safety = metriche.get("benchmarks", {}).get("reviews_safety_critical.csv", {})
        righe.append(
            {
                "profilo": etichetta_profilo(profilo_al),
                "run": etichetta,
                "modelli presenti": Path(dep_path).exists() and Path(sent_path).exists(),
                "test reparto F1": test.get("department", {}).get("f1_macro"),
                "test sentiment F1": test.get("sentiment", {}).get("f1_macro"),
                "casi critici reparto F1": safety.get("department", {}).get("f1_macro"),
                "casi critici sentiment F1": safety.get("sentiment", {}).get("f1_macro"),
                "file metriche": str(metrics_path.relative_to(ROOT_DIR)),
            }
        )

    df_al = pd.DataFrame(righe)
    st.markdown("### Metriche run active learning")
    st.caption("I modelli dopo essere stati ricreati compaiono nel menu laterale `Profilo modello`.")
    st.dataframe(df_al.round(4), use_container_width=True, hide_index=True, height=180)


def mostra_risultati_minilm() -> None:
    """Mostra l'ultimo confronto baseline vs MiniLM salvato negli output."""
    paths = sorted(percorso_root("outputs").glob("transformer_comparison_*.json"))
    if not paths:
        st.info("Nessun risultato MiniLM presente. Usa il pulsante sopra per generarlo.")
        return

    path = paths[-1]
    report = carica_json(str(path), predefinito={})
    test_split = report.get("test_split", {})
    benchmark = report.get("benchmarks", {})

    st.markdown("### Ultimo risultato MiniLM")
    st.caption(
        f"Fonte: `{path.relative_to(ROOT_DIR)}`. MiniLM produce embedding del testo; le classi non vengono decise dal transformer, "
        "ma da due LogisticRegression addestrate sugli embedding, una per reparto e una per sentiment."
    )

    righe = []
    for sezione, dati in [("Test split", test_split), *[(nome, valore) for nome, valore in benchmark.items()]]:
        base = dati.get("baseline", {})
        mini = dati.get("transformer", {})
        delta = dati.get("delta_f1_macro", {})
        righe.append(
            {
                "sezione": sezione.replace("reviews_", "").replace(".csv", ""),
                "baseline_reparto_f1": base.get("department", {}).get("f1_macro"),
                "minilm_reparto_f1": mini.get("department", {}).get("f1_macro"),
                "delta_reparto": delta.get("department"),
                "baseline_sentiment_f1": base.get("sentiment", {}).get("f1_macro"),
                "minilm_sentiment_f1": mini.get("sentiment", {}).get("f1_macro"),
                "delta_sentiment": delta.get("sentiment"),
            }
        )

    df_minilm = pd.DataFrame(righe)
    if not df_minilm.empty:
        c1, c2, c3, c4 = st.columns(4)
        test = df_minilm.iloc[0]
        c1.metric("Base reparto F1", f"{test['baseline_reparto_f1']:.4f}")
        c2.metric("MiniLM reparto F1", f"{test['minilm_reparto_f1']:.4f}")
        c3.metric("Base sentiment F1", f"{test['baseline_sentiment_f1']:.4f}")
        c4.metric("MiniLM sentiment F1", f"{test['minilm_sentiment_f1']:.4f}")
        df_minilm_display = df_minilm.rename(
            columns={
                "sezione": "Sezione",
                "baseline_reparto_f1": "Base reparto F1",
                "minilm_reparto_f1": "MiniLM reparto F1",
                "delta_reparto": "Delta reparto",
                "baseline_sentiment_f1": "Base sentiment F1",
                "minilm_sentiment_f1": "MiniLM sentiment F1",
                "delta_sentiment": "Delta sentiment",
            }
        )
        st.dataframe(df_minilm_display.round(4), use_container_width=True, hide_index=True, height=250)


def mostra_stato_artefatti() -> None:
    """Visualizza lo stato degli artefatti principali necessari alla demo."""
    rows = [
        {"Risorsa": "Base pulito reparto", "Presente": percorso_root("models/baseline_pure/department_model.joblib").exists(), "Percorso": "models/baseline_pure/department_model.joblib"},
        {"Risorsa": "Base pulito sentiment", "Presente": percorso_root("models/baseline_pure/sentiment_model.joblib").exists(), "Percorso": "models/baseline_pure/sentiment_model.joblib"},
        {"Risorsa": "Base irrobustito reparto", "Presente": percorso_root("models/baseline_hardened/department_model.joblib").exists(), "Percorso": "models/baseline_hardened/department_model.joblib"},
        {"Risorsa": "Base irrobustito sentiment", "Presente": percorso_root("models/baseline_hardened/sentiment_model.joblib").exists(), "Percorso": "models/baseline_hardened/sentiment_model.joblib"},
        {"Risorsa": "Avanzato APS pulito reparto", "Presente": percorso_root("models/advanced_aps_pure/department_ensemble_advanced.joblib").exists(), "Percorso": "models/advanced_aps_pure/department_ensemble_advanced.joblib"},
        {"Risorsa": "Avanzato APS pulito sentiment", "Presente": percorso_root("models/advanced_aps_pure/sentiment_ensemble_advanced.joblib").exists(), "Percorso": "models/advanced_aps_pure/sentiment_ensemble_advanced.joblib"},
        {"Risorsa": "Avanzato APS irrobustito reparto", "Presente": percorso_root("models/advanced_aps_hardened/department_ensemble_advanced.joblib").exists(), "Percorso": "models/advanced_aps_hardened/department_ensemble_advanced.joblib"},
        {"Risorsa": "Avanzato APS irrobustito sentiment", "Presente": percorso_root("models/advanced_aps_hardened/sentiment_ensemble_advanced.joblib").exists(), "Percorso": "models/advanced_aps_hardened/sentiment_ensemble_advanced.joblib"},
        {"Risorsa": "Active learning oracle", "Presente": percorso_root("models/active_learning_oracle/department_model.joblib").exists(), "Percorso": "models/active_learning_oracle/department_model.joblib"},
        {"Risorsa": "Active learning v2 no replay", "Presente": percorso_root("models/active_learning_v2_no_replay/department_model.joblib").exists(), "Percorso": "models/active_learning_v2_no_replay/department_model.joblib"},
        {"Risorsa": "Active learning v2 replay", "Presente": percorso_root("models/active_learning_v2_replay/department_model.joblib").exists(), "Percorso": "models/active_learning_v2_replay/department_model.joblib"},
        {"Risorsa": "Coda AL oracle", "Presente": percorso_root("data/active_learning/active_learning_queue_oracle_labeled.csv").exists(), "Percorso": "data/active_learning/active_learning_queue_oracle_labeled.csv"},
        {"Risorsa": "Coda AL v2", "Presente": percorso_root("data/active_learning/active_learning_v2_queue_labeled.csv").exists(), "Percorso": "data/active_learning/active_learning_v2_queue_labeled.csv"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def materiali_scaricabili() -> list[dict[str, str]]:
    """Elenca i file utili da esporre in dashboard."""
    return [
        {"Categoria": "Dataset", "Nome": "Dataset principale", "Percorso": "data/reviews_synth.csv"},
        {"Categoria": "Dataset", "Nome": "CSV demo per batch", "Percorso": "data/demo/reviews_trappola_demo.csv"},
        {"Categoria": "Dataset", "Nome": "Benchmark in distribuzione", "Percorso": "data/benchmarks/reviews_in_distribution.csv"},
        {"Categoria": "Dataset", "Nome": "Benchmark ambiguo", "Percorso": "data/benchmarks/reviews_ambiguous.csv"},
        {"Categoria": "Dataset", "Nome": "Benchmark rumoroso", "Percorso": "data/benchmarks/reviews_noisy.csv"},
        {"Categoria": "Dataset", "Nome": "Benchmark colloquiale", "Percorso": "data/benchmarks/reviews_colloquial.csv"},
        {"Categoria": "Dataset", "Nome": "Benchmark casi critici", "Percorso": "data/benchmarks/reviews_safety_critical.csv"},
        {"Categoria": "Output demo", "Nome": "Predizioni demo già generate", "Percorso": "outputs/predictions_demo_refresh.csv"},
        {"Categoria": "Output demo", "Nome": "Riepilogo SLA demo", "Percorso": "outputs/sla_batch_summary_demo.json"},
        {"Categoria": "Metriche JSON", "Nome": "Metriche base pulito", "Percorso": "outputs/baseline_pure/metrics.json"},
        {"Categoria": "Metriche JSON", "Nome": "Metriche base irrobustito", "Percorso": "outputs/baseline_hardened/metrics.json"},
        {"Categoria": "Metriche JSON", "Nome": "Metriche avanzato APS pulito", "Percorso": "outputs/advanced_aps_pure/metrics_advanced.json"},
        {"Categoria": "Metriche JSON", "Nome": "Metriche avanzato APS irrobustito", "Percorso": "outputs/advanced_aps_hardened/metrics_advanced.json"},
        {"Categoria": "Metriche JSON", "Nome": "Sintesi progetto avanzato", "Percorso": "outputs/advanced_project_summary.json"},
        {"Categoria": "Metriche JSON", "Nome": "Confronto MiniLM", "Percorso": "outputs/transformer_comparison_20260428_005027.json"},
        {"Categoria": "Metriche JSON", "Nome": "Report casi critici prima/dopo", "Percorso": "outputs/safety_exp/safety_delta_report_r2.json"},
        {"Categoria": "Report", "Nome": "Report casi critici markdown", "Percorso": "outputs/safety_exp/safety_delta_report_r2.md"},
        {"Categoria": "Active learning", "Nome": "Coda oracle etichettata", "Percorso": "data/active_learning/active_learning_queue_oracle_labeled.csv"},
        {"Categoria": "Active learning", "Nome": "Coda v2 etichettata", "Percorso": "data/active_learning/active_learning_v2_queue_labeled.csv"},
    ]


def mime_file(percorso: Path) -> str:
    """Restituisce il MIME type usato nei download della dashboard."""
    suffisso = percorso.suffix.lower()
    if suffisso == ".csv":
        return "text/csv"
    if suffisso == ".json":
        return "application/json"
    if suffisso == ".png":
        return "image/png"
    if suffisso == ".md":
        return "text/markdown"
    return "application/octet-stream"


def mostra_materiali_scaricabili() -> None:
    """Mostra una panoramica dei principali artefatti scaricabili."""
    materiali = materiali_scaricabili()
    righe = []
    for item in materiali:
        percorso = percorso_root(item["Percorso"])
        righe.append(
            {
                "Categoria": item["Categoria"],
                "Materiale": item["Nome"],
                "Presente": percorso.exists(),
                "Percorso": item["Percorso"],
                "Dimensione KB": round(percorso.stat().st_size / 1024, 1) if percorso.exists() else None,
            }
        )
    st.dataframe(pd.DataFrame(righe), use_container_width=True, hide_index=True, height=280)
    st.caption("Questa sezione espone solo file già presenti su disco. Se manca qualcosa, rigeneralo dagli strumenti sopra o da terminale.")

    for categoria in dict.fromkeys(item["Categoria"] for item in materiali):
        presenti = [item for item in materiali if item["Categoria"] == categoria and percorso_root(item["Percorso"]).exists()]
        if not presenti:
            continue
        st.markdown(f"#### {categoria}")
        colonne = st.columns(3)
        for indice_download, item in enumerate(presenti):
            percorso = percorso_root(item["Percorso"])
            with colonne[indice_download % 3]:
                st.download_button(
                    label=f"Scarica {item['Nome']}",
                    data=percorso.read_bytes(),
                    file_name=percorso.name,
                    mime=mime_file(percorso),
                    key=f"download_materiale_{categoria}_{indice_download}_{percorso.name}",
                    use_container_width=True,
                )


def avvia_dashboard() -> None:
    """Avvia la dashboard Streamlit del project work."""
    st.title("Da recensione a decisione operativa: prototipo ML per la gestione dei feedback alberghieri")
    st.caption("Santoro_Martino_0312300507")
    st.caption("Modalità predefinita: risposta automatica. Controllo umano opzionale sui casi incerti.")

    profili = profili_disponibili()
    dep_model = sent_model = dep_thr = sent_thr = dep_classes = sent_classes = conformal_model = None
    profilo = None
    load_error = None

    supporta_token_exp = False

    if profili:
        indice_default = profili.index("baseline_pure") if "baseline_pure" in profili else 0
        profilo_default = profili[indice_default]
        profilo = st.sidebar.selectbox(
            "Profilo modello",
            options=profili,
            index=indice_default,
            format_func=etichetta_profilo,
            help="base pulita: casi critici solo in valutazione | irrobustita: casi critici usati anche nel training",
        )
        if profilo_default == "baseline_pure":
            st.sidebar.caption("Profilo predefinito: modello base pulito, cioè il riferimento principale del progetto.")
        else:
            st.sidebar.caption(
                f"Il modello base pulito non è presente su disco; uso il primo profilo disponibile: {etichetta_profilo(profilo_default)}."
            )
        st.sidebar.caption(descrizione_profilo(profilo))
        if not any(p.startswith("advanced_aps") for p in profili):
            st.sidebar.caption("Profili avanzati APS non disponibili in questo ambiente (modelli non presenti).")

        try:
            dep_model, sent_model, dep_thr, sent_thr, dep_classes, sent_classes, conformal_model = carica_modelli_e_soglie(
                profilo
            )
        except Exception as exc:
            load_error = str(exc)
            st.sidebar.error(f"Errore caricamento profilo {profilo}: {exc}")
        else:
            supporta_token_exp = supporta_spiegabilita_per_token(dep_model) and supporta_spiegabilita_per_token(sent_model)
    else:
        st.sidebar.warning("Nessun profilo modello trovato. Usa la tab Strumenti per rigenerare modello base e dataset.")

    if str(profilo).startswith("advanced_aps") and load_error and moduli_mancanti(DIPENDENZE_ADVANCED):
        mostra_prompt_installazione_dipendenze(
            chiave_prompt="prompt_fasttext_profile",
            chiave_risultato="tool_install_fasttext_profile",
            titolo="Installazione dipendenze base",
            messaggio="Il profilo `advanced_aps` è presente ma non può essere caricato in questo ambiente.",
            file_requisiti=FILE_REQUISITI_BASE,
            moduli=DIPENDENZE_ADVANCED,
            refresh_models=True,
        )

    if str(profilo).startswith("advanced_aps") and dep_model is not None:
        st.info(f"Profilo attivo: {etichetta_profilo(profilo)} (ensemble + APS).")
        st.caption(
            "Il primo caricamento di questo profilo può richiedere 2-4 minuti: i modelli advanced occupano "
            "oltre 1.2 GB complessivi e poi restano in cache nella sessione."
        )
        if not supporta_token_exp:
            st.caption("Nota: con questo profilo non è disponibile la spiegazione per parole.")
    elif profilo and dep_model is not None:
        st.info(f"Profilo attivo: {etichetta_profilo(profilo)}. {descrizione_profilo(profilo)}")

    tab1, tab2, tab3 = st.tabs(["Predizione singola", "Predizione batch (CSV)", "Strumenti e rigenerazione"])

    with tab1:
        if dep_model is None or sent_model is None:
            st.subheader("Predizione non disponibile")
            if load_error:
                st.error(f"Il profilo selezionato non è stato caricato correttamente: {load_error}")
            else:
                st.info("Non ci sono modelli caricati. Usa la tab Strumenti e rigenerazione per creare almeno il modello base.")
        else:
            st.subheader("Incolla una recensione")
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Titolo", value="Bagno con muffa e asciugamani sporchi")
            with col2:
                top_k = st.slider(
                    "Parole influenti da mostrare",
                    3,
                    10,
                    5,
                    1,
                    disabled=not supporta_token_exp,
                    help=(
                        "Mostra parole o frammenti che hanno pesato di più nella decisione del modello lineare. "
                        "La barra cambia solo quante voci visualizzare nelle tabelle finali: non modifica la predizione."
                    ),
                )
                if supporta_token_exp:
                    st.caption("La barra regola solo quante parole mostrare nell'analisi finale: non cambia reparto, sentiment o rischio.")
                else:
                    st.caption("Con questo profilo non è disponibile la spiegazione per parole, quindi la barra resta disattivata.")

            diagnostic_single = st.toggle(
                "Segnala se serve controllo umano",
                value=False,
                help="Aggiunge un avviso quando il caso è incerto o delicato e meriterebbe una verifica manuale.",
            )

            body = st.text_area(
                "Testo recensione",
                value=(
                    "Il check-in è stato rapido e il personale gentile, però in camera il bagno aveva muffa "
                    "nella doccia, asciugamani sporchi e un forte odore di umidità. Abbiamo segnalato il "
                    "problema alla reception, ma non è stato risolto durante il soggiorno."
                ),
                height=180,
            )
            with st.expander("Cosa sono le parole più influenti?"):
                st.write(
                    "È una lettura del modello: elenca parole o frammenti che hanno spinto maggiormente verso reparto e sentiment previsti."
                )
                st.write(
                    "Non è una seconda predizione e non cambia il risultato. Aumentare la barra da 3 a 10 mostra solo più parole nella tabella finale."
                )
                st.write(
                    "Nel profilo base i contributi arrivano dal modello TF-IDF lineare; nei profili avanzati questa vista non è disponibile."
                )

            if st.button("Predici"):
                res = predici_recensione_singola(
                    modello_reparto=dep_model,
                    modello_sentiment=sent_model,
                    soglie_reparto=dep_thr,
                    soglie_sentiment=sent_thr,
                    classi_reparto=dep_classes,
                    classi_sentiment=sent_classes,
                    modello_conformal=conformal_model,
                    titolo=title,
                    corpo=body,
                    top_k=top_k,
                    revisione_diagnostica=diagnostic_single,
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Reparto consigliato", res["department"], f"confidenza {res['department_confidence']:.2f}")
                c2.metric("Sentiment stimato", res["sentiment"], f"confidenza {res['sentiment_confidence']:.2f}")
                c3.metric("Priorità operativa", etichetta_priorita(res["priority"]))
                c4.metric("Rischio operativo", f"{res['risk_score']:.2f}", "scala 0-1")

                st.markdown("### Lettura rapida")
                lettura_singola = [
                    f"La recensione viene indirizzata a **{res['department']}** con confidenza **{res['department_confidence']:.2f}**.",
                    f"Il sentiment stimato è **{res['sentiment']}** con confidenza **{res['sentiment_confidence']:.2f}**.",
                    f"La priorità operativa è **{etichetta_priorita(res['priority'])}** e il rischio è **{res['risk_score']:.2f}** su scala 0-1.",
                ]
                if res.get("sentiment_guardrail_applied"):
                    lettura_singola.append(
                        "È stata applicata una correzione del sentiment: il testo contiene segnali negativi forti."
                    )
                if res.get("sentiment_hazard_flag"):
                    lettura_singola.append(
                        "Sono presenti segnali igienico-sanitari critici: il caso va trattato come criticità operativa, non come semplice feedback generico."
                    )
                if diagnostic_single:
                    if res["needs_review_diag"]:
                        lettura_singola.append(
                            "Il sistema consiglia un controllo umano: il caso è abbastanza incerto o delicato da meritare verifica."
                        )
                    else:
                        lettura_singola.append("Non emergono segnali tali da richiedere controllo umano: il caso può restare in gestione automatica.")
                if res.get("cross_department_signal"):
                    lettura_singola.append(
                        "Il testo coinvolge più reparti: il reparto mostrato è quello principale, ma la tabella sotto evidenzia anche gli impatti secondari."
                    )
                st.info("\n\n".join(lettura_singola))

                if res.get("sentiment_guardrail_applied"):
                    st.warning(
                        "Correzione del sentiment attivata: testo con segnali negativi forti "
                        f"(punteggio={res.get('sentiment_guardrail_score', 0)})."
                    )
                if res.get("sentiment_hazard_flag"):
                    st.error(f"Segnale igienico-sanitario critico rilevato (punteggio={res.get('sentiment_hazard_score', 0)}).")

                if diagnostic_single:
                    st.info(f"Controllo umano consigliato: {'SÌ' if res['needs_review_diag'] else 'NO'}")

                if "department_conformal_set" in res:
                    st.markdown("### Conformal set reparto (APS)")
                    st.write(", ".join(res["department_conformal_set"]))
                    st.caption(f"Dimensione set: {res['department_conformal_size']}")

                st.markdown("### Impatto multi-reparto")
                st.write(", ".join(res["impacted_departments"]))
                if res.get("cross_department_signal"):
                    st.caption("Segnale multi-topic: coinvolgimento di più reparti nello stesso testo.")

                st.markdown("### Probabilità reparto")
                st.caption("Distribuzione delle probabilità del modello per il routing reparto.")
                st.dataframe(pd.DataFrame([res["department_probs"]]), use_container_width=True, hide_index=True)

                st.markdown("### Probabilità sentiment")
                st.caption("Distribuzione delle probabilità del modello tra sentiment positivo e negativo.")
                st.dataframe(pd.DataFrame([res["sentiment_probs"]]), use_container_width=True, hide_index=True)

                if res["dep_explain"] or res["sent_explain"]:
                    st.markdown("### Parole più influenti")
                    exp_col1, exp_col2 = st.columns(2)
                    with exp_col1:
                        st.write("Reparto")
                        st.caption("Parole o frammenti che hanno pesato di più nella scelta del reparto.")
                        st.dataframe(pd.DataFrame(res["dep_explain"], columns=["parola/frammento", "contributo"]), hide_index=True)
                    with exp_col2:
                        st.write("Sentiment")
                        st.caption("Parole o frammenti che hanno pesato di più nella scelta del sentiment.")
                        st.dataframe(pd.DataFrame(res["sent_explain"], columns=["parola/frammento", "contributo"]), hide_index=True)

    with tab2:
        if dep_model is None or sent_model is None:
            st.subheader("Predizione batch non disponibile")
            st.info("Per usare il batch devi prima avere almeno il profilo base disponibile. Puoi crearlo dalla tab Strumenti.")
        else:
            st.subheader("Predizione batch")
            st.write("Puoi usare il CSV demo incluso oppure caricare un file personale con colonne: `title`, `body` e `id` opzionale.")

            df_batch = None
            origine_batch = None
            demo_path = percorso_root("data/demo/reviews_trappola_demo.csv")
            if demo_path.exists():
                demo_df = pd.read_csv(demo_path)
                st.markdown("### CSV demo pronto all'uso")
                st.caption("Contiene recensioni miste, ambigue e con criticità operative, utili per provare subito la dashboard.")
                demo_col1, demo_col2, demo_col3 = st.columns(3)
                with demo_col1:
                    if st.button("Mostra CSV demo", key="btn_show_demo_batch"):
                        st.session_state["show_demo_batch_csv"] = not st.session_state.get("show_demo_batch_csv", False)
                with demo_col2:
                    if st.button("Esegui demo batch", key="btn_run_demo_batch"):
                        st.session_state["run_demo_batch_csv"] = True
                        st.session_state["run_uploaded_batch_csv"] = False
                        st.session_state["show_demo_batch_csv"] = True
                with demo_col3:
                    st.download_button(
                        label="Scarica CSV demo",
                        data=demo_df.to_csv(index=False).encode("utf-8"),
                        file_name="reviews_trappola_demo.csv",
                        mime="text/csv",
                    )

                if st.session_state.get("show_demo_batch_csv"):
                    st.dataframe(demo_df, use_container_width=True, hide_index=True)
                if st.session_state.get("run_demo_batch_csv"):
                    df_batch = demo_df.copy()
                    origine_batch = "CSV demo incluso: data/demo/reviews_trappola_demo.csv"
                    st.info("Demo batch attiva. Per usare un file personale, carica un CSV sotto.")
            else:
                st.warning("CSV demo non trovato: `data/demo/reviews_trappola_demo.csv`.")

            st.markdown("### CSV personale")
            up = st.file_uploader("Carica CSV personale", type=["csv"])
            top_k_batch = st.slider(
                "Parole influenti nel batch",
                3,
                10,
                5,
                1,
                key="topk_batch",
                disabled=not supporta_token_exp,
                help=(
                    "Controlla quante parole o frammenti salvare nelle colonne di spiegazione del batch. "
                    "Non modifica le predizioni, solo il dettaglio mostrato o esportato."
                ),
            )
            if supporta_token_exp:
                st.caption("Nel batch la barra cambia solo quante parole o frammenti salvare per ogni riga.")
            else:
                st.caption("Nel profilo corrente la spiegazione per parole non è disponibile, quindi il controllo batch è disattivato.")
            diagnostic_batch = st.toggle(
                "Segnala righe da controllare",
                value=False,
                help="Aggiunge una colonna che indica quali righe sono incerte o delicate e meritano verifica manuale.",
            )

            if up is not None:
                uploaded_bytes = up.getvalue()
                uploaded_signature = f"{up.name}:{len(uploaded_bytes)}"
                if st.session_state.get("uploaded_batch_signature") != uploaded_signature:
                    st.session_state["run_uploaded_batch_csv"] = False
                    st.session_state["uploaded_batch_signature"] = uploaded_signature
                uploaded_df = pd.read_csv(io.BytesIO(uploaded_bytes))
                st.session_state["uploaded_batch_name"] = up.name
                st.session_state["uploaded_batch_df"] = uploaded_df
                st.session_state["run_demo_batch_csv"] = False
                st.caption(f"CSV caricato: {up.name}")
                st.dataframe(uploaded_df, use_container_width=True, hide_index=True, height=240)
                if st.button("Esegui batch su CSV caricato", key="btn_run_uploaded_batch"):
                    st.session_state["run_uploaded_batch_csv"] = True

            if st.session_state.get("run_uploaded_batch_csv") and st.session_state.get("uploaded_batch_df") is not None:
                df_batch = st.session_state["uploaded_batch_df"].copy()
                origine_batch = f"CSV caricato: {st.session_state.get('uploaded_batch_name', 'file utente')}"

            if df_batch is not None:
                st.caption(f"Origine batch: {origine_batch}")
                if "title" not in df_batch.columns or "body" not in df_batch.columns:
                    st.error("Il CSV deve contenere le colonne 'title' e 'body'.")
                else:
                    out_df = predici_recensioni_lotto(
                        modello_reparto=dep_model,
                        modello_sentiment=sent_model,
                        soglie_reparto=dep_thr,
                        soglie_sentiment=sent_thr,
                        classi_reparto=dep_classes,
                        classi_sentiment=sent_classes,
                        modello_conformal=conformal_model,
                        df=df_batch,
                        top_k=top_k_batch,
                        revisione_diagnostica=diagnostic_batch,
                    )

                    st.markdown("### KPI operativi")
                    totale_righe = int(len(out_df))
                    alta_priorita = int(out_df["priority"].isin(["HIGH", "URGENT"]).sum())
                    risk_medio = float(out_df["risk_score"].mean())
                    guardrail_count = (
                        int(out_df["sentiment_guardrail_applied"].sum())
                        if "sentiment_guardrail_applied" in out_df.columns
                        else 0
                    )
                    hazard_count = (
                        int(out_df["sentiment_hazard_flag"].sum())
                        if "sentiment_hazard_flag" in out_df.columns
                        else 0
                    )
                    needs_review_count = (
                        int(out_df["needs_review_diag"].sum())
                        if diagnostic_batch and "needs_review_diag" in out_df.columns
                        else 0
                    )

                    k1, k2, k3 = st.columns(3)
                    k1.metric("Righe analizzate", totale_righe)
                    k2.metric(
                        "Priorità alta",
                        f"{alta_priorita / max(totale_righe, 1):.2%}",
                        f"{alta_priorita}/{totale_righe} ALTA o URGENTE",
                    )
                    k3.metric("Rischio medio", f"{risk_medio:.2f}", "scala 0-1")
                    g1, g2, g3 = st.columns(3)
                    if "sentiment_guardrail_applied" in out_df.columns:
                        g1.metric(
                            "Correzioni prudenziali",
                            f"{guardrail_count / max(totale_righe, 1):.2%}",
                            f"{guardrail_count}/{totale_righe} casi corretti in prudenza",
                        )
                    if "sentiment_hazard_flag" in out_df.columns:
                        g2.metric(
                            "Segnali igienico-sanitari critici",
                            f"{hazard_count / max(totale_righe, 1):.2%}",
                            f"{hazard_count}/{totale_righe} casi con segnali critici",
                        )

                    if diagnostic_batch and "needs_review_diag" in out_df.columns:
                        g3.metric(
                            "Da controllare",
                            f"{needs_review_count / max(totale_righe, 1):.2%}",
                            f"{needs_review_count}/{totale_righe} righe da verificare",
                        )
                    if "department_conformal_size" in out_df.columns:
                        st.metric("Conformal set medio", f"{out_df['department_conformal_size'].mean():.2f}")

                    st.markdown("### Lettura rapida del lotto")
                    lettura = [
                        f"Il file contiene **{totale_righe} recensioni**.",
                        f"**{alta_priorita}** recensioni sono in priorità alta (`ALTA` o `URGENTE`), quindi richiedono attenzione operativa prima delle altre.",
                        f"Il punteggio di rischio medio è **{risk_medio:.2f}** su scala 0-1: più si avvicina a 1, più il lotto è critico.",
                    ]
                    if "sentiment_guardrail_applied" in out_df.columns:
                        lettura.append(
                            f"Il guardrail sentiment è intervenuto su **{guardrail_count}** righe: sono casi in cui parole o segnali negativi forti rendono prudente non fidarsi solo della probabilità del modello."
                        )
                    if "sentiment_hazard_flag" in out_df.columns:
                        lettura.append(
                            f"I segnali igienico-sanitari critici compaiono in **{hazard_count}** righe: muffa, odori forti, allergie o segnali simili vengono evidenziati come criticità operative."
                        )
                    if diagnostic_batch and "needs_review_diag" in out_df.columns:
                        lettura.append(
                            f"Il controllo umano viene consigliato per **{needs_review_count}** righe: non blocca il sistema, ma segnala i casi meno affidabili o più delicati."
                        )
                    st.info("\n\n".join(lettura))

                    st.markdown("### Anteprima risultati")
                    st.dataframe(rinomina_colonne_dashboard(out_df), use_container_width=True, hide_index=True, height=460)

                    sla_df = simula_sla(out_df)
                    st.markdown("### Simulazione SLA")
                    st.dataframe(rinomina_colonne_dashboard(sla_df), use_container_width=True, hide_index=True, height=260)

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    export_name = f"predictions_{ts}.csv"
                    st.download_button(
                        label="Scarica risultati CSV",
                        data=out_df.to_csv(index=False).encode("utf-8"),
                        file_name=export_name,
                        mime="text/csv",
                    )

                    st.download_button(
                        label="Scarica SLA summary JSON",
                        data=json.dumps({"rows": sla_df.to_dict(orient="records")}, ensure_ascii=False, indent=2).encode(
                            "utf-8"
                        ),
                        file_name=f"sla_summary_{ts}.json",
                        mime="application/json",
                    )

    with tab3:
        st.subheader("Strumenti guidati")
        st.caption(
            "Questa sezione serve ai docenti per ricreare dataset, modelli ed esperimenti senza usare il terminale. "
            "Ogni pulsante lancia gli script Python del progetto e mostra il log in italiano."
        )

        top_col1, top_col2 = st.columns([3, 1])
        with top_col1:
            st.markdown("### Stato rapido degli artefatti")
        with top_col2:
            if st.button("Aggiorna stato", help="Svuota la cache dei modelli e rilegge i file presenti su disco."):
                carica_modelli_e_soglie.clear()
                st.rerun()

        mostra_stato_artefatti()

        with st.expander("Materiali scaricabili per verifica", expanded=False):
            st.write(
                "Qui raccolgo i principali file già prodotti dal progetto, così possono "
                "essere scaricati senza dover essere cercati a mano nelle cartelle."
            )
            mostra_materiali_scaricabili()

        st.markdown("### Preparazione del progetto")
        with st.expander("Dataset e benchmark", expanded=False):
            st.write("Usa questo pulsante se mancano `data/reviews_synth.csv` o i file in `data/benchmarks/`, oppure se vuoi ricrearli da zero con lo stesso seed.")
            if st.button("Rigenera dataset e benchmark", key="btn_regen_data"):
                esegui_passi_guidati(
                    "tool_regen_data",
                    "Rigenerazione dataset e benchmark",
                    [
                        {
                            "label": "Generazione dataset sintetico e benchmark",
                            "cmd": [
                                sys.executable,
                                "src/01_generate_dataset.py",
                                "--n",
                                "350",
                                "--seed",
                                "42",
                                "--percorso_output",
                                "data/reviews_synth.csv",
                                "--cartella_benchmark",
                                "data/benchmarks",
                                "--n_benchmark_colloquiali",
                                "220",
                            ],
                        }
                    ],
                )
            mostra_risultato_strumento("tool_regen_data")

        with st.expander("Modelli base", expanded=False):
            st.write("Ricrea separatamente il modello base pulito e quello irrobustito.")
            st.caption("Pulito: i casi critici restano solo valutazione. Irrobustito: i casi critici vengono aggiunti al training con repeat=2.")
            col_base_pure, col_base_hardened = st.columns(2)
            with col_base_pure:
                if st.button("Ricrea base pulito", key="btn_regen_baseline_pure"):
                    esegui_passi_guidati(
                        "tool_regen_baseline_pure",
                        "Rigenerazione modello base pulito",
                        [
                            {
                                "label": "Training modello base pulito",
                                "cmd": [
                                    sys.executable,
                                    "src/02_train_evaluate.py",
                                    "--dati",
                                    "data/reviews_synth.csv",
                                    "--cartella_benchmark",
                                    "data/benchmarks",
                                    "--revisione_diagnostica",
                                    "--cartella_modelli",
                                    "models/baseline_pure",
                                    "--cartella_output",
                                    "outputs/baseline_pure",
                                ],
                            }
                        ],
                        refresh_models=True,
                    )
            with col_base_hardened:
                if st.button("Ricrea base irrobustito", key="btn_regen_baseline_hardened"):
                    esegui_passi_guidati(
                        "tool_regen_baseline_hardened",
                        "Rigenerazione modello base irrobustito",
                        [
                            {
                                "label": "Training modello base irrobustito",
                                "cmd": [
                                    sys.executable,
                                    "src/02_train_evaluate.py",
                                    "--dati",
                                    "data/reviews_synth.csv",
                                    "--cartella_benchmark",
                                    "data/benchmarks",
                                    "--percorso_augment_sicurezza",
                                    "data/benchmarks/reviews_safety_critical.csv",
                                    "--ripetizioni_augment_sicurezza",
                                    "2",
                                    "--revisione_diagnostica",
                                    "--cartella_modelli",
                                    "models/baseline_hardened",
                                    "--cartella_output",
                                    "outputs/baseline_hardened",
                                ],
                            }
                        ],
                        refresh_models=True,
                    )
            mostra_risultato_strumento("tool_regen_baseline_pure")
            mostra_risultato_strumento("tool_regen_baseline_hardened")

        st.markdown("### Profilo avanzato")
        with st.expander("Modelli avanzati APS", expanded=False):
            st.write("Addestra separatamente il profilo avanzato pulito e quello irrobustito.")
            st.caption("Il primo caricamento dei profili avanzati può essere lento per la dimensione dei file `.joblib`.")
            blocca_advanced = mostra_prompt_installazione_dipendenze(
                chiave_prompt="prompt_fasttext_advanced",
                chiave_risultato="tool_install_fasttext_advanced",
                titolo="Installazione dipendenze base",
                messaggio="Per addestrare o caricare `advanced_aps` serve `fasttext`, incluso nelle dipendenze base.",
                file_requisiti=FILE_REQUISITI_BASE,
                moduli=DIPENDENZE_ADVANCED,
                refresh_models=True,
            )
            if not blocca_advanced:
                col_adv_pure, col_adv_hardened = st.columns(2)
                with col_adv_pure:
                    if st.button("Rigenera avanzato pulito", key="btn_regen_advanced_pure"):
                        esegui_passi_guidati(
                            "tool_regen_advanced_pure",
                            "Rigenerazione avanzato APS pulito",
                            [
                                {
                                    "label": "Training avanzato APS pulito",
                                    "cmd": [
                                        sys.executable,
                                        "src/07_train_advanced.py",
                                        "--dati",
                                        "data/reviews_synth.csv",
                                        "--cartella_benchmark",
                                        "data/benchmarks",
                                        "--revisione_diagnostica",
                                        "--metodo_conformal",
                                        "aps",
                                        "--alpha_conformal",
                                        "0.10",
                                        "--cartella_modelli",
                                        "models/advanced_aps_pure",
                                        "--cartella_output",
                                        "outputs/advanced_aps_pure",
                                    ],
                                }
                            ],
                            refresh_models=True,
                        )
                with col_adv_hardened:
                    if st.button("Rigenera avanzato irrobustito", key="btn_regen_advanced_hardened"):
                        esegui_passi_guidati(
                            "tool_regen_advanced_hardened",
                            "Rigenerazione avanzato APS irrobustito",
                            [
                                {
                                    "label": "Training avanzato APS irrobustito",
                                    "cmd": [
                                        sys.executable,
                                        "src/07_train_advanced.py",
                                        "--dati",
                                        "data/reviews_synth.csv",
                                        "--cartella_benchmark",
                                        "data/benchmarks",
                                        "--percorso_augment_sicurezza",
                                        "data/benchmarks/reviews_safety_critical.csv",
                                        "--ripetizioni_augment_sicurezza",
                                        "2",
                                        "--revisione_diagnostica",
                                        "--metodo_conformal",
                                        "aps",
                                        "--alpha_conformal",
                                        "0.10",
                                        "--cartella_modelli",
                                        "models/advanced_aps_hardened",
                                        "--cartella_output",
                                        "outputs/advanced_aps_hardened",
                                    ],
                                }
                            ],
                            refresh_models=True,
                        )
            mostra_risultato_strumento("tool_regen_advanced_pure")
            mostra_risultato_strumento("tool_regen_advanced_hardened")

        st.markdown("### Esperimenti")
        with st.expander("Nuovo ciclo di active learning", expanded=False):
            st.write("Genera una nuova coda di campioni difficili a partire dal benchmark `reviews_noisy.csv`.")
            st.caption("Questo non addestra modelli nuovi: costruisce la coda da far revisionare a un umano.")
            if st.button("Simula nuovo ciclo AL", key="btn_simulate_al"):
                esegui_passi_guidati(
                    "tool_simulate_al",
                    "Simulazione nuovo ciclo active learning",
                    [
                        {
                            "label": "Generazione coda AL",
                            "cmd": [
                                sys.executable,
                                "src/05_active_learning_cycle.py",
                                "--dataset",
                                "data/reviews_synth.csv",
                                "--pool",
                                "data/benchmarks/reviews_noisy.csv",
                                "--numero_top",
                                "40",
                                "--solo_revisione_diagnostica",
                                "--strategia",
                                "hybrid_v2",
                                "--peso_incertezza",
                                "0.50",
                                "--peso_diversita",
                                "0.30",
                                "--peso_operativo",
                                "0.20",
                                "--percorso_coda_output",
                                "outputs/active_learning_queue_latest.csv",
                            ],
                        }
                    ],
                )
            mostra_risultato_strumento("tool_simulate_al")

        with st.expander("Confronto casi critici prima/dopo", expanded=False):
            st.write("Rigenera il confronto controllato tra modello base pulito e modello base irrobustito sui casi critici.")
            st.caption(
                "Confronto sperimentale: addestra due modelli base in `models/safety_exp/*`, "
                "può durare alcuni minuti, produce `outputs/safety_exp/safety_delta_report_r2.md` "
                "e serve a misurare l'effetto dell'irrobustimento senza cambiare il profilo usato per inferenza."
            )
            if st.button("Rigenera confronto casi critici", key="btn_safety_delta"):
                esegui_passi_guidati(
                    "tool_safety_delta",
                    "Rigenerazione confronto casi critici prima/dopo",
                    [
                        {
                            "label": "Training modello base pulito safety_exp",
                            "cmd": [
                                sys.executable,
                                "src/02_train_evaluate.py",
                                "--dati",
                                "data/reviews_synth.csv",
                                "--cartella_benchmark",
                                "data/benchmarks",
                                "--revisione_diagnostica",
                                "--cartella_modelli",
                                "models/safety_exp/base",
                                "--cartella_output",
                                "outputs/safety_exp/base",
                                "--seed",
                                "42",
                            ],
                        },
                        {
                            "label": "Training modello base irrobustito safety_exp",
                            "cmd": [
                                sys.executable,
                                "src/02_train_evaluate.py",
                                "--dati",
                                "data/reviews_synth.csv",
                                "--cartella_benchmark",
                                "data/benchmarks",
                                "--percorso_augment_sicurezza",
                                "data/benchmarks/reviews_safety_critical.csv",
                                "--ripetizioni_augment_sicurezza",
                                "2",
                                "--revisione_diagnostica",
                                "--cartella_modelli",
                                "models/safety_exp/hardened_r2",
                                "--cartella_output",
                                "outputs/safety_exp/hardened_r2",
                                "--seed",
                                "42",
                            ],
                        },
                        {
                            "label": "Creazione report delta casi critici",
                            "cmd": [
                                sys.executable,
                                "src/08_safety_delta_report.py",
                                "--before",
                                "outputs/safety_exp/base/metrics.json",
                                "--after",
                                "outputs/safety_exp/hardened_r2/metrics.json",
                                "--safety_benchmark",
                                "reviews_safety_critical.csv",
                                "--out_json",
                                "outputs/safety_exp/safety_delta_report_r2.json",
                                "--out_md",
                                "outputs/safety_exp/safety_delta_report_r2.md",
                            ],
                        },
                    ],
                )
            mostra_risultato_strumento("tool_safety_delta")
            mostra_confronto_safety()

        with st.expander("Active learning archiviato", expanded=False):
            st.write("Rigenera i tre profili active learning a partire dalle code etichettate incluse nel repository.")
            st.caption(
                "Ogni run crea prima il dataset aggiornato, poi addestra i modelli nella cartella dedicata. "
                "Quando il run termina correttamente, il profilo compare nel menu laterale."
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Oracle", key="btn_al_oracle"):
                    esegui_passi_guidati(
                        "tool_al_oracle",
                        "Riproduzione run active learning oracle",
                        [
                            {
                                "label": "Costruzione dataset oracle",
                                "cmd": [
                                    sys.executable,
                                    "src/05_active_learning_cycle.py",
                                    "--dataset",
                                    "data/reviews_synth.csv",
                                    "--pool",
                                    "data/benchmarks/reviews_noisy.csv",
                                    "--coda_etichettata",
                                    "data/active_learning/active_learning_queue_oracle_labeled.csv",
                                    "--percorso_dataset_output",
                                    "data/reviews_synth_active_oracle.csv",
                                    "--percorso_coda_output",
                                    "outputs/active_learning_queue_oracle_rebuilt.csv",
                                ],
                            },
                            {
                                "label": "Training modelli oracle",
                                "cmd": [
                                    sys.executable,
                                    "src/02_train_evaluate.py",
                                    "--dati",
                                    "data/reviews_synth_active_oracle.csv",
                                    "--cartella_benchmark",
                                    "data/benchmarks",
                                    "--percorso_augment_sicurezza",
                                    "data/benchmarks/reviews_safety_critical.csv",
                                    "--ripetizioni_augment_sicurezza",
                                    "2",
                                    "--revisione_diagnostica",
                                    "--cartella_modelli",
                                    "models/active_learning_oracle",
                                    "--cartella_output",
                                    "outputs/active_learning_oracle",
                                    "--seed",
                                    "42",
                                ],
                            },
                        ],
                        refresh_models=True,
                    )
            with c2:
                if st.button("V2 no replay", key="btn_al_v2_noreplay"):
                    esegui_passi_guidati(
                        "tool_al_v2_noreplay",
                        "Riproduzione run active learning v2 no replay",
                        [
                            {
                                "label": "Costruzione dataset v2 no replay",
                                "cmd": [
                                    sys.executable,
                                    "src/05_active_learning_cycle.py",
                                    "--dataset",
                                    "data/reviews_synth.csv",
                                    "--pool",
                                    "data/benchmarks/reviews_noisy.csv",
                                    "--coda_etichettata",
                                    "data/active_learning/active_learning_v2_queue_labeled.csv",
                                    "--percorso_dataset_output",
                                    "data/reviews_synth_active_v2.csv",
                                    "--percorso_coda_output",
                                    "outputs/active_learning_v2_queue_rebuilt.csv",
                                ],
                            },
                            {
                                "label": "Training modelli v2 no replay",
                                "cmd": [
                                    sys.executable,
                                    "src/02_train_evaluate.py",
                                    "--dati",
                                    "data/reviews_synth_active_v2.csv",
                                    "--cartella_benchmark",
                                    "data/benchmarks",
                                    "--percorso_augment_sicurezza",
                                    "data/benchmarks/reviews_safety_critical.csv",
                                    "--ripetizioni_augment_sicurezza",
                                    "2",
                                    "--revisione_diagnostica",
                                    "--cartella_modelli",
                                    "models/active_learning_v2_no_replay",
                                    "--cartella_output",
                                    "outputs/active_learning_v2_no_replay",
                                    "--seed",
                                    "42",
                                ],
                            },
                        ],
                        refresh_models=True,
                    )
            with c3:
                if st.button("V2 replay", key="btn_al_v2_replay"):
                    esegui_passi_guidati(
                        "tool_al_v2_replay",
                        "Riproduzione run active learning v2 replay",
                        [
                            {
                                "label": "Costruzione dataset v2 replay",
                                "cmd": [
                                    sys.executable,
                                    "src/05_active_learning_cycle.py",
                                    "--dataset",
                                    "data/reviews_synth.csv",
                                    "--pool",
                                    "data/benchmarks/reviews_noisy.csv",
                                    "--coda_etichettata",
                                    "data/active_learning/active_learning_v2_queue_labeled.csv",
                                    "--percorso_dataset_output",
                                    "data/reviews_synth_active_v2_replay.csv",
                                    "--percorso_coda_output",
                                    "outputs/active_learning_v2_queue_rebuilt_replay.csv",
                                    "--dimensione_replay",
                                    "120",
                                    "--seed_replay",
                                    "42",
                                ],
                            },
                            {
                                "label": "Training modelli v2 replay",
                                "cmd": [
                                    sys.executable,
                                    "src/02_train_evaluate.py",
                                    "--dati",
                                    "data/reviews_synth_active_v2_replay.csv",
                                    "--cartella_benchmark",
                                    "data/benchmarks",
                                    "--percorso_augment_sicurezza",
                                    "data/benchmarks/reviews_safety_critical.csv",
                                    "--ripetizioni_augment_sicurezza",
                                    "2",
                                    "--revisione_diagnostica",
                                    "--cartella_modelli",
                                    "models/active_learning_v2_replay",
                                    "--cartella_output",
                                    "outputs/active_learning_v2_replay",
                                    "--seed",
                                    "42",
                                ],
                            },
                        ],
                        refresh_models=True,
                    )
            mostra_risultato_strumento("tool_al_oracle")
            mostra_risultato_strumento("tool_al_v2_noreplay")
            mostra_risultato_strumento("tool_al_v2_replay")
            mostra_metriche_active_learning()

        with st.expander("Confronto con MiniLM", expanded=False):
            st.write("Rilancia il benchmark comparativo tra modello base e `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.")
            st.caption("La dashboard usa `HF_HOME=.hf_cache/huggingface`, i pesi MiniLM restano nella cache locale del progetto, così da poterli cancellare facilmente.")
            st.info(
                "MiniLM non è il classificatore, trasforma ogni recensione in embedding "
                "numerici, poi due LogisticRegression imparano le classi \"department\" e \"sentiment\" usando le etichette del dataset."
            )
            blocca_transformer = mostra_prompt_installazione_dipendenze(
                chiave_prompt="prompt_transformer_minilm",
                chiave_risultato="tool_install_transformer",
                titolo="Installazione dipendenze confronto MiniLM",
                messaggio="Per eseguire il confronto MiniLM servono librerie aggiuntive non incluse nel setup base.",
                file_requisiti=FILE_REQUISITI_TRANSFORMER,
                moduli=DIPENDENZE_TRANSFORMER,
            )
            if not blocca_transformer and st.button("Esegui confronto MiniLM", key="btn_transformer_cmp"):
                esegui_passi_guidati(
                    "tool_transformer_cmp",
                    "Confronto con MiniLM",
                    [
                        {
                            "label": "Benchmark modello base vs MiniLM",
                            "cmd": [
                                sys.executable,
                                "src/06_compare_transformer.py",
                                "--dati",
                                "data/reviews_synth.csv",
                                "--cartella_benchmark",
                                "data/benchmarks",
                                "--modello_reparto_baseline",
                                "models/baseline_pure/department_model.joblib",
                                "--modello_sentiment_baseline",
                                "models/baseline_pure/sentiment_model.joblib",
                                "--seed",
                                "42",
                            ],
                            "env": {"HF_HOME": str(percorso_root(".hf_cache/huggingface"))},
                        }
                    ],
                )
            mostra_risultato_strumento("tool_transformer_cmp")
            mostra_risultati_minilm()


if __name__ == "__main__":
    avvia_dashboard()
