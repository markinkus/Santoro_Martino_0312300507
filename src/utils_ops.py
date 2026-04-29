from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = {
    "department": {"default": 0.55},
    "sentiment": {"default": 0.55},
}

DEFAULT_SLA_HOURS = {
    "Reception": 2.0,
    "Housekeeping": 4.0,
    "F&B": 6.0,
}

DEFAULT_CAPACITY_PER_HOUR = {
    "Reception": 14.0,
    "Housekeeping": 10.0,
    "F&B": 8.0,
}

SENTIMENT_NEG_CUES = {
    "sporco",
    "sporca",
    "sporchi",
    "sporcizia",
    "orrendo",
    "orrenda",
    "orrendi",
    "orrende",
    "orribile",
    "orribili",
    "terribile",
    "terribili",
    "inaccettabile",
    "vergognoso",
    "vergognosa",
    "puzzolente",
    "puzza",
    "puzzava",
    "muffa",
    "insalubre",
    "nauseante",
    "scarafaggi",
    "scarafaggio",
    "blatte",
    "cimici",
    "acari",
    "morso",
    "morsi",
    "infestazione",
    "infestata",
    "infestato",
    "disgustoso",
    "disgustosa",
    "schifo",
    "deludente",
    "pessimo",
    "pessima",
    "problema",
    "problemi",
    "intoppi",
    "guasto",
    "guasti",
    "rotto",
    "rotta",
}

SENTIMENT_NEG_BIGRAMS = {
    "odore forte",
    "bagno sporco",
    "camera sporca",
    "servizio pessimo",
    "attesa lunga",
    "scarafaggi in cucina",
    "cimici da letto",
    "peli sul camice",
    "odore nauseante",
    "igiene pessima",
    "peli delle ascelle",
}

SENTIMENT_POS_CUES = {
    "ottimo",
    "ottima",
    "ottimi",
    "ottime",
    "spettacolo",
    "top",
    "promosso",
    "promossa",
    "contento",
    "contenta",
    "contenti",
    "contente",
    "gentile",
    "gentili",
    "buono",
    "buona",
    "buoni",
    "buone",
    "pulito",
    "pulita",
    "puliti",
    "pulite",
    "comodo",
    "comoda",
    "comodi",
    "comode",
    "consigliato",
    "consigliata",
    "torneremo",
    "tornerei",
}

SENTIMENT_POS_BIGRAMS = {
    "si mangia bene",
    "molto pulito",
    "davvero pulito",
    "staff gentile",
    "staff gentilissimo",
    "posto spettacolo",
    "ci siamo trovati bene",
    "rimasta contenta",
    "rimasto contento",
    "colazione ottima",
    "camera pulita",
}

SENTIMENT_HAZARD_CUES = {
    "scarafaggi",
    "scarafaggio",
    "blatte",
    "cimici",
    "acari",
    "muffa",
    "insalubre",
    "infestazione",
    "puzzolente",
    "puzzava",
    "nauseante",
    "morso",
    "morsi",
}

SENTIMENT_HAZARD_BIGRAMS = {
    "scarafaggi in cucina",
    "cimici da letto",
    "acari che ci hanno morso",
    "peli sul camice",
    "peli delle ascelle",
    "odore nauseante",
    "igiene pessima",
}

SENTIMENT_NEGATORS = {"non", "nessun", "nessuna", "nessuno", "senza", "mai", "niente"}

DEPARTMENT_IMPACT_CUES = {
    "Housekeeping": {
        "camera",
        "stanza",
        "bagno",
        "doccia",
        "lenzuola",
        "asciugamani",
        "materasso",
        "acari",
        "cimici",
        "muffa",
        "pulizia",
        "riordino",
        "aria condizionata",
        "letto",
        "cuscini",
        "polvere",
        "odore",
        "peli",
        "wc",
        "bidet",
        "scarafaggi",
    },
    "Reception": {
        "check in",
        "check out",
        "prenotazione",
        "fattura",
        "pagamento",
        "reception",
        "accoglienza",
        "attesa",
        "documenti",
        "desk",
        "coda",
        "garage",
        "navetta",
        "chiave",
        "chiavi",
        "parcheggio",
        "camera pronta",
        "late check out",
        "desk",
    },
    "F&B": {
        "colazione",
        "buffet",
        "ristorante",
        "cucina",
        "menu",
        "piatti",
        "cuoco",
        "bar",
        "bevande",
        "glutine",
        "lattosio",
        "allergia",
        "allergeni",
        "cena",
        "pranzo",
        "mangia",
        "mangiato",
        "cameriere",
        "tavolo",
        "torta",
        "compleanno",
        "cornetti",
        "cappuccino",
        "aperitivo",
        "conto",
    },
}
def garantisci_cartella(percorso: str | Path) -> None:
    """Crea una cartella se non esiste già.

    Parametri:
    - percorso: directory da creare.
    """
    Path(percorso).mkdir(parents=True, exist_ok=True)


def garantisci_cartella_padre(percorso_file: str | Path) -> None:
    """Crea la cartella padre di un file se non esiste già.

    Parametri:
    - percorso_file: file di destinazione di cui garantire la directory padre.
    """
    Path(percorso_file).parent.mkdir(parents=True, exist_ok=True)


def salva_json(dati: dict[str, Any], percorso_output: str | Path) -> None:
    """Salva un dizionario JSON in UTF-8 con indentazione leggibile.

    Parametri:
    - dati: contenuto da serializzare.
    - percorso_output: file di destinazione.
    """
    garantisci_cartella_padre(percorso_output)
    with open(percorso_output, "w", encoding="utf-8") as file_output:
        json.dump(dati, file_output, ensure_ascii=False, indent=2)


def carica_json(
    percorso_input: str | Path,
    predefinito: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Carica un file JSON restituendo un fallback se il file manca.

    Parametri:
    - percorso_input: file da leggere.
    - predefinito: valore restituito se il file non esiste.
    """
    percorso = Path(percorso_input)
    if not percorso.exists():
        return predefinito or {}
    with open(percorso, "r", encoding="utf-8") as file_input:
        return json.load(file_input)


def normalizza_mappa_soglie(
    mappa_soglie: dict[str, float] | None,
    classi: list[str],
    fallback: float = 0.55,
) -> dict[str, float]:
    """Normalizza una mappa di soglie e assicura una soglia per ogni classe.

    Parametri:
    - mappa_soglie: mappa letta da file o calcolata in training.
    - classi: classi previste dal modello.
    - fallback: soglia da usare se mancano valori validi.
    """
    soglie_normalizzate = {"default": float(fallback)}
    if mappa_soglie:
        for chiave, valore in mappa_soglie.items():
            try:
                soglie_normalizzate[str(chiave)] = float(valore)
            except (TypeError, ValueError):
                continue
    for classe in classi:
        soglie_normalizzate.setdefault(classe, soglie_normalizzate.get("default", fallback))
    return soglie_normalizzate


def deriva_soglie_per_classe(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classi: list[str],
    quantile: float = 0.08,
    floor: float = 0.45,
    ceil: float = 0.90,
) -> dict[str, float]:
    """Deriva soglie per classe a partire dalle probabilità corrette osservate.

    Parametri:
    - y_true: etichette reali del set di calibrazione.
    - y_proba: probabilità prodotte dal modello.
    - classi: ordine delle classi nelle colonne di `y_proba`.
    - quantile: quantile usato per stimare una soglia prudenziale.
    - floor: limite inferiore della soglia.
    - ceil: limite superiore della soglia.
    """
    soglie: dict[str, float] = {}
    if y_true.size == 0:
        return {"default": floor}

    for indice, classe in enumerate(classi):
        maschera = y_true == classe
        if not np.any(maschera):
            soglie[classe] = floor
            continue

        probabilita = y_proba[maschera, indice]
        valore_quantile = float(np.quantile(probabilita, quantile))
        soglie[classe] = float(np.clip(valore_quantile, floor, ceil))

    valore_default = float(np.clip(np.mean(list(soglie.values())), floor, ceil))
    soglie["default"] = valore_default
    return soglie


def calcola_revisione_necessaria(
    etichette_predette: np.ndarray,
    confidenze: np.ndarray,
    mappa_soglie: dict[str, float],
    fallback: float = 0.55,
) -> np.ndarray:
    """Marca i campioni che richiedono revisione diagnostica.

    Parametri:
    - etichette_predette: classi previste dal modello.
    - confidenze: confidenza associata a ogni predizione.
    - mappa_soglie: soglie per classe.
    - fallback: soglia di sicurezza da usare se manca la classe.
    """
    revisioni = []
    soglia_default = float(mappa_soglie.get("default", fallback))
    for etichetta, confidenza in zip(etichette_predette, confidenze):
        soglia = float(mappa_soglie.get(str(etichetta), soglia_default))
        revisioni.append(float(confidenza) < soglia)
    return np.array(revisioni, dtype=bool)


def calcola_priorita_e_rischio(
    probabilita_negativa: float,
    confidenza_reparto: float,
    confidenza_sentiment: float,
    necessita_revisione: bool,
    punteggio_hazard: float = 0.0,
    multi_reparto: bool = False,
) -> tuple[str, float]:
    """Calcola priorità e rischio operativo per una recensione.

    Parametri:
    - probabilita_negativa: probabilità stimata del sentiment negativo.
    - confidenza_reparto: confidenza del task reparto.
    - confidenza_sentiment: confidenza del task sentiment.
    - necessita_revisione: flag diagnostico di incertezza.
    - punteggio_hazard: severità lessicale di pattern safety.
    - multi_reparto: indica se il testo coinvolge più reparti.
    """
    rischio = (
        0.62 * float(probabilita_negativa)
        + 0.24 * (1.0 - float(confidenza_reparto))
        + 0.14 * (1.0 - float(confidenza_sentiment))
    )
    if necessita_revisione:
        rischio += 0.05
    if multi_reparto:
        rischio += 0.04

    hazard = float(max(0.0, punteggio_hazard))
    rischio += min(0.24, 0.06 * hazard)

    rischio = float(np.clip(rischio, 0.0, 1.0))

    # Indizi igienico-sanitari gravi non devono finire in priorità basse.
    if hazard >= 2.0:
        rischio = max(rischio, 0.58)

    if rischio >= 0.78:
        return "URGENT", rischio
    if rischio >= 0.58:
        return "HIGH", rischio
    if rischio >= 0.38:
        return "MEDIUM", rischio
    return "LOW", rischio


def estrai_classi_e_coefficienti(classificatore) -> tuple[list[str], np.ndarray | None]:
    """Recupera classi e coefficienti da un classificatore lineare calibrato o no."""
    classi = list(getattr(classificatore, "classes_", []))

    if hasattr(classificatore, "coef_"):
        return classi, classificatore.coef_

    # Caso CalibratedClassifierCV: ricostruisco la media dei coefficienti dei modelli interni.
    calibrati = getattr(classificatore, "calibrated_classifiers_", None)
    if calibrati:
        lista_coefficienti = []
        for calibrato in calibrati:
            stimatore = getattr(calibrato, "estimator", None)
            if stimatore is not None and hasattr(stimatore, "coef_"):
                lista_coefficienti.append(stimatore.coef_)
        if lista_coefficienti:
            return classi, np.mean(np.stack(lista_coefficienti, axis=0), axis=0)

    return classi, None


def contributi_principali_token(modello, testo: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Restituisce i token che contribuiscono di più alla decisione del modello."""
    vettorizzatore = modello.named_steps["tfidf"]
    classificatore = modello.named_steps["clf"]

    classi, coefficienti = estrai_classi_e_coefficienti(classificatore)
    if coefficienti is None:
        return []

    matrice_testo = vettorizzatore.transform([testo])
    predizione = modello.predict([testo])[0]

    nomi_feature = vettorizzatore.get_feature_names_out()

    if coefficienti.shape[0] == 1:
        if len(classi) < 2:
            return []
        classe_positiva = classi[1]
        vettore_coefficienti = coefficienti[0] if predizione == classe_positiva else -coefficienti[0]
    else:
        if predizione not in classi:
            return []
        indice_classe = classi.index(predizione)
        vettore_coefficienti = coefficienti[indice_classe]

    riga = matrice_testo[0]
    indici = riga.indices
    valori = riga.data
    if len(indici) == 0:
        return []

    punteggi = valori * vettore_coefficienti[indici]
    coppie = [(nomi_feature[i], float(p)) for i, p in zip(indici, punteggi) if p > 0]

    if not coppie:
        coppie = [(nomi_feature[i], float(abs(p))) for i, p in zip(indici, punteggi)]

    coppie.sort(key=lambda elemento: elemento[1], reverse=True)
    return coppie[:top_k]


def applica_campi_operativi(
    df: pd.DataFrame,
    soglie_reparto: dict[str, float] | None = None,
    soglie_sentiment: dict[str, float] | None = None,
    revisione_diagnostica: bool = False,
) -> pd.DataFrame:
    """Aggiunge colonne operative come review diagnostica, priorità e rischio."""
    out = df.copy()

    if revisione_diagnostica and soglie_reparto and soglie_sentiment:
        revisione_reparto = calcola_revisione_necessaria(
            etichette_predette=out["pred_department"].values,
            confidenze=out["department_confidence"].values,
            mappa_soglie=soglie_reparto,
        )
        revisione_sentiment = calcola_revisione_necessaria(
            etichette_predette=out["pred_sentiment"].values,
            confidenze=out["sentiment_confidence"].values,
            mappa_soglie=soglie_sentiment,
        )
        out["needs_review_diag"] = revisione_reparto | revisione_sentiment
    else:
        out["needs_review_diag"] = False

    if "proba_sentiment_neg" in out.columns:
        probabilita_negative = out["proba_sentiment_neg"].astype(float).values
    else:
        probabilita_negative = np.where(out["pred_sentiment"].values == "neg", 0.70, 0.30)

    priorita = []
    rischi = []
    for indice in range(len(out)):
        punteggio_hazard = (
            float(out.iloc[indice]["sentiment_hazard_score"]) if "sentiment_hazard_score" in out.columns else 0.0
        )
        if "impacted_departments_count" in out.columns:
            multi_reparto = int(out.iloc[indice]["impacted_departments_count"]) > 1
        else:
            multi_reparto = bool(out.iloc[indice].get("cross_department_signal", False))

        priorita_corrente, rischio_corrente = calcola_priorita_e_rischio(
            probabilita_negativa=float(probabilita_negative[indice]),
            confidenza_reparto=float(out.iloc[indice]["department_confidence"]),
            confidenza_sentiment=float(out.iloc[indice]["sentiment_confidence"]),
            necessita_revisione=bool(out.iloc[indice]["needs_review_diag"]),
            punteggio_hazard=punteggio_hazard,
            multi_reparto=multi_reparto,
        )
        priorita.append(priorita_corrente)
        rischi.append(rischio_corrente)

    out["priority"] = priorita
    out["risk_score"] = rischi
    return out


def normalizza_per_matching(testo: str) -> tuple[str, list[str]]:
    """Normalizza un testo per il matching lessicale del runtime."""
    testo_normalizzato = str(testo or "").strip().lower()
    if not testo_normalizzato:
        return "", []
    testo_normalizzato = re.sub(r"[^a-z0-9àèìòù\s]", " ", testo_normalizzato)
    testo_normalizzato = re.sub(r"\s+", " ", testo_normalizzato).strip()
    return testo_normalizzato, testo_normalizzato.split()


def ha_negazione_locale(token: list[str], indice: int, finestra: int = 2) -> bool:
    """Controlla se vicino al token target è presente una negazione locale."""
    precedenti = token[max(0, indice - finestra) : indice]
    return any(parola in SENTIMENT_NEGATORS for parola in precedenti)


def punteggio_indizi_sentiment(testo: str) -> tuple[int, list[str]]:
    """Calcola un punteggio lessicale di negatività sul testo."""
    testo_normalizzato, token = normalizza_per_matching(testo)
    if not testo_normalizzato:
        return 0, []

    punteggio = 0
    motivi: list[str] = []

    for bigramma in SENTIMENT_NEG_BIGRAMS:
        if bigramma in testo_normalizzato:
            punteggio += 2
            motivi.append(f"bigram:{bigramma}")

    for indice, parola in enumerate(token):
        if parola not in SENTIMENT_NEG_CUES:
            continue
        if ha_negazione_locale(token, indice):
            continue
        punteggio += 1
        motivi.append(f"cue:{parola}")

    return punteggio, motivi


def punteggio_hazard_sentiment(testo: str) -> tuple[int, list[str]]:
    """Calcola un punteggio di severità safety sul testo."""
    testo_normalizzato, token = normalizza_per_matching(testo)
    if not testo_normalizzato:
        return 0, []

    punteggio = 0
    motivi: list[str] = []

    for bigramma in SENTIMENT_HAZARD_BIGRAMS:
        if bigramma in testo_normalizzato:
            punteggio += 2
            motivi.append(f"hazard_bigram:{bigramma}")

    for indice, parola in enumerate(token):
        if parola not in SENTIMENT_HAZARD_CUES:
            continue
        if ha_negazione_locale(token, indice):
            continue
        punteggio += 1
        motivi.append(f"hazard_cue:{parola}")

    return punteggio, motivi


def punteggio_positivita_sentiment(testo: str) -> tuple[int, list[str]]:
    """Calcola un punteggio lessicale di positività sul testo."""
    testo_normalizzato, token = normalizza_per_matching(testo)
    if not testo_normalizzato:
        return 0, []

    punteggio = 0
    motivi: list[str] = []

    for bigramma in SENTIMENT_POS_BIGRAMS:
        if bigramma in testo_normalizzato:
            punteggio += 2
            motivi.append(f"pos_bigram:{bigramma}")

    for indice, parola in enumerate(token):
        if parola not in SENTIMENT_POS_CUES:
            continue
        if ha_negazione_locale(token, indice):
            continue
        punteggio += 1
        motivi.append(f"pos_cue:{parola}")

    return punteggio, motivi


def profilo_indizi_reparto(testo: str) -> tuple[dict[str, int], dict[str, list[str]]]:
    """Costruisce profili di indizi lessicali per ciascun reparto."""
    testo_normalizzato, token = normalizza_per_matching(testo)
    insieme_token = set(token)
    punteggi = {reparto: 0 for reparto in DEPARTMENT_IMPACT_CUES}
    corrispondenze = {reparto: [] for reparto in DEPARTMENT_IMPACT_CUES}
    if not testo_normalizzato:
        return punteggi, corrispondenze

    for reparto, indizi in DEPARTMENT_IMPACT_CUES.items():
        for indizio in indizi:
            if " " in indizio:
                if indizio in testo_normalizzato:
                    punteggi[reparto] += 2
                    corrispondenze[reparto].append(indizio)
            elif indizio in insieme_token:
                punteggi[reparto] += 1
                corrispondenze[reparto].append(indizio)
    return punteggi, corrispondenze


def applica_reparti_impattati_df(
    df: pd.DataFrame,
    colonna_testo: str = "text",
    colonna_predizione: str = "pred_department",
    prefisso_prob_reparto: str = "proba_department_",
    soglia_indizi: int = 2,
    soglia_probabilita: float = 0.28,
) -> pd.DataFrame:
    """Aggiunge la ricostruzione dei reparti coinvolti oltre al reparto principale."""
    out = df.copy()
    if colonna_testo not in out.columns or colonna_predizione not in out.columns:
        return out

    colonne_prob = [colonna for colonna in out.columns if colonna.startswith(prefisso_prob_reparto)]
    classi_reparto = (
        [colonna[len(prefisso_prob_reparto) :] for colonna in colonne_prob]
        if colonne_prob
        else list(DEPARTMENT_IMPACT_CUES.keys())
    )

    valori_reparti: list[str] = []
    conteggi_reparti: list[int] = []
    motivi_reparti: list[str] = []
    segnali_cross: list[bool] = []

    for indice in range(len(out)):
        riga = out.iloc[indice]
        reparto_predetto = str(riga[colonna_predizione]) if colonna_predizione in out.columns else ""
        testo = str(riga[colonna_testo])

        punteggi_indizi, match_indizi = profilo_indizi_reparto(testo)
        selezionati = set()
        if reparto_predetto:
            selezionati.add(reparto_predetto)

        for reparto in classi_reparto:
            colonna_prob = f"{prefisso_prob_reparto}{reparto}"
            if colonna_prob in out.columns and float(riga[colonna_prob]) >= soglia_probabilita:
                selezionati.add(reparto)
            if punteggi_indizi.get(reparto, 0) >= int(soglia_indizi):
                selezionati.add(reparto)

        ranking = sorted(
            classi_reparto,
            key=lambda reparto: (
                int(punteggi_indizi.get(reparto, 0)),
                float(riga.get(f"{prefisso_prob_reparto}{reparto}", 0.0)),
                reparto,
            ),
            reverse=True,
        )
        reparti_coinvolti = [reparto for reparto in ranking if reparto in selezionati]
        if not reparti_coinvolti and reparto_predetto:
            reparti_coinvolti = [reparto_predetto]

        motivi = []
        for reparto in reparti_coinvolti[:3]:
            punteggio = punteggi_indizi.get(reparto, 0)
            if punteggio > 0:
                indizi = ",".join(match_indizi.get(reparto, [])[:3])
                motivi.append(f"{reparto}:{punteggio}[{indizi}]")
            else:
                motivi.append(f"{reparto}:model")

        valori_reparti.append("|".join(reparti_coinvolti))
        conteggi_reparti.append(len(reparti_coinvolti))
        motivi_reparti.append(";".join(motivi))
        segnali_cross.append(len(reparti_coinvolti) > 1)

    out["impacted_departments"] = valori_reparti
    out["impacted_departments_count"] = conteggi_reparti
    out["impacted_departments_reason"] = motivi_reparti
    out["cross_department_signal"] = segnali_cross
    return out


def applica_guardrail_sentiment_riga(
    testo: str,
    classi_sentiment: list[str],
    riga_probabilita_sentiment: np.ndarray,
    sentiment_predetto: str,
) -> tuple[str, np.ndarray, bool, int, int, str]:
    """Applica il guardrail safety a una singola riga."""
    classi = [str(classe) for classe in classi_sentiment]
    if "neg" not in classi or "pos" not in classi:
        return sentiment_predetto, riga_probabilita_sentiment, False, 0, 0, ""

    indice_neg = classi.index("neg")
    indice_pos = classi.index("pos")
    probabilita = riga_probabilita_sentiment.astype(float).copy()

    punteggio_indizi, motivi_indizi = punteggio_indizi_sentiment(testo)
    punteggio_hazard, motivi_hazard = punteggio_hazard_sentiment(testo)
    punteggio_positivita, motivi_positivi = punteggio_positivita_sentiment(testo)
    attivato = False

    # Il layer hazard ha precedenza: anomalie igienico-sanitarie non devono passare per positive.
    if str(sentiment_predetto) == "pos" and punteggio_hazard >= 1:
        target_neg = 0.78 if punteggio_hazard == 1 else 0.90 if punteggio_hazard == 2 else 0.95
        if probabilita[indice_neg] < target_neg:
            probabilita[indice_neg] = target_neg
            probabilita[indice_pos] = max(0.0, 1.0 - target_neg)
            somma = float(np.sum(probabilita))
            if somma > 0:
                probabilita = probabilita / somma
            attivato = True
    elif str(sentiment_predetto) == "pos" and punteggio_indizi >= 2:
        target_neg = 0.68 if punteggio_indizi == 2 else 0.82
        if probabilita[indice_neg] < target_neg:
            probabilita[indice_neg] = target_neg
            probabilita[indice_pos] = max(0.0, 1.0 - target_neg)
            somma = float(np.sum(probabilita))
            if somma > 0:
                probabilita = probabilita / somma
            attivato = True
    elif (
        str(sentiment_predetto) == "neg"
        and punteggio_hazard == 0
        and punteggio_indizi == 0
        and punteggio_positivita >= 3
    ):
        target_pos = 0.74 if punteggio_positivita == 3 else 0.86
        if probabilita[indice_pos] < target_pos:
            probabilita[indice_pos] = target_pos
            probabilita[indice_neg] = max(0.0, 1.0 - target_pos)
            somma = float(np.sum(probabilita))
            if somma > 0:
                probabilita = probabilita / somma
            attivato = True

    predizione = classi[int(np.argmax(probabilita))]
    tutti_i_motivi = motivi_hazard[:3] + motivi_indizi[:3] + motivi_positivi[:3]
    return predizione, probabilita, attivato, punteggio_indizi, punteggio_hazard, ";".join(tutti_i_motivi[:5])


def applica_guardrail_sentiment_df(
    df: pd.DataFrame,
    colonna_testo: str = "text",
    colonna_predizione: str = "pred_sentiment",
    prefisso_probabilita: str = "proba_sentiment_",
) -> pd.DataFrame:
    """Applica il guardrail sentiment a un intero DataFrame di predizioni."""
    out = df.copy()

    if colonna_testo not in out.columns or colonna_predizione not in out.columns:
        return out

    colonne_prob = [colonna for colonna in out.columns if colonna.startswith(prefisso_probabilita)]
    if not colonne_prob:
        return out

    classi_sentiment = [colonna[len(prefisso_probabilita) :] for colonna in colonne_prob]

    guardrail_attivo: list[bool] = []
    punteggi: list[int] = []
    punteggi_hazard: list[int] = []
    motivi: list[str] = []

    for indice in range(len(out)):
        riga = out.iloc[indice]
        riga_probabilita = np.array([float(riga[colonna]) for colonna in colonne_prob], dtype=float)

        (
            nuova_predizione,
            nuove_probabilita,
            attivato,
            punteggio_indizi,
            punteggio_hazard,
            motivo,
        ) = applica_guardrail_sentiment_riga(
            testo=str(riga[colonna_testo]),
            classi_sentiment=classi_sentiment,
            riga_probabilita_sentiment=riga_probabilita,
            sentiment_predetto=str(riga[colonna_predizione]),
        )

        out.at[indice, colonna_predizione] = nuova_predizione
        for posizione, colonna in enumerate(colonne_prob):
            out.at[indice, colonna] = float(nuove_probabilita[posizione])

        if "sentiment_confidence" in out.columns:
            out.at[indice, "sentiment_confidence"] = float(np.max(nuove_probabilita))

        guardrail_attivo.append(bool(attivato))
        punteggi.append(int(punteggio_indizi))
        punteggi_hazard.append(int(punteggio_hazard))
        motivi.append(motivo)

    out["sentiment_guardrail_applied"] = guardrail_attivo
    out["sentiment_guardrail_score"] = punteggi
    out["sentiment_hazard_score"] = punteggi_hazard
    out["sentiment_hazard_flag"] = [punteggio >= 2 for punteggio in punteggi_hazard]
    out["sentiment_guardrail_reason"] = motivi
    return out


def simula_sla(
    df: pd.DataFrame,
    finestra_ore: int = 8,
    ore_sla: dict[str, float] | None = None,
    capacita_per_ora: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Simula backlog e tenuta SLA per reparto su una finestra temporale."""
    mappa_sla = ore_sla or DEFAULT_SLA_HOURS
    mappa_capacita = capacita_per_ora or DEFAULT_CAPACITY_PER_HOUR

    righe: list[dict[str, Any]] = []

    for reparto in ["Reception", "Housekeeping", "F&B"]:
        df_reparto = df[df["pred_department"] == reparto]
        volume = int(len(df_reparto))
        target_sla = float(mappa_sla.get(reparto, 4.0))
        capacita = float(mappa_capacita.get(reparto, 8.0)) * float(finestra_ore)

        backlog = max(0, volume - int(capacita))
        rapporto_carico = (volume / capacita) if capacita > 0 else 1.0

        risposta_media_ore = target_sla * (1.0 + max(0.0, rapporto_carico - 1.0) * 1.75)

        if volume == 0:
            quota_urgenti = 0.0
        else:
            quota_urgenti = float((df_reparto["priority"].isin(["URGENT", "HIGH"]).sum()) / volume)

        rispetto_sla = float(np.clip(1.0 - max(0.0, rapporto_carico - 1.0) * 0.55 - quota_urgenti * 0.12, 0.0, 1.0))

        righe.append(
            {
                "department": reparto,
                "volume": volume,
                "window_hours": int(finestra_ore),
                "capacity_tickets": int(capacita),
                "backlog_estimated": int(backlog),
                "sla_target_hours": target_sla,
                "estimated_avg_response_hours": round(float(risposta_media_ore), 2),
                "estimated_sla_hit_rate": round(float(rispetto_sla), 3),
            }
        )

    return pd.DataFrame(righe)
