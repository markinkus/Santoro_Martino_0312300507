import argparse
import random
from pathlib import Path

import pandas as pd

from utils_ops import garantisci_cartella_padre

DEPARTMENTS = ["Housekeeping", "Reception", "F&B"] # reparti
SENTIMENTS = ["pos", "neg"] # sentimenti

DEPARTMENT_TOPICS = { # temi ricorrenti per reparto
    "Housekeeping": {
        "core": [
            "camera",
            "stanza",
            "bagno",
            "doccia",
            "lenzuola",
            "asciugamani",
            "materasso",
            "letto",
            "cuscini",
            "polvere",
            "odore",
            "aria condizionata",
            "manutenzione",
            "riordino",
        ],
        "service": ["pulizia", "comfort", "igiene", "silenzio", "intervento tecnico", "rapidità del riordino"],
    },
    "Reception": {
        "core": [
            "check in",
            "check out",
            "prenotazione",
            "fattura",
            "pagamento",
            "accoglienza",
            "attesa",
            "documenti",
            "garage",
            "navetta",
            "chiave",
            "camera pronta",
            "late check out",
            "parcheggio",
        ],
        "service": ["assistenza", "rapidità", "chiarezza", "disponibilità", "organizzazione", "gestione reclami"],
    },
    "F&B": {
        "core": [
            "colazione",
            "buffet",
            "ristorante",
            "menu",
            "piatti",
            "porzioni",
            "bar",
            "bevande",
            "glutine",
            "lattosio",
            "cena",
            "pranzo",
            "cameriere",
            "tavolo",
            "cornetti",
            "cappuccino",
            "torta",
            "aperitivo",
        ],
        "service": ["qualità", "varietà", "servizio", "temperature", "gestione allergie", "tempi di sala"],
    },
}

STARTERS = [ # frasi di apertura
    "Siamo stati in struttura due notti.",
    "Weekend veloce, recensione scritta a caldo.",
    "Viaggio di lavoro, quindi esperienza abbastanza pratica.",
    "Ci siamo fermati solo per un fine settimana.",
    "Scrivo il feedback subito dopo il checkout.",
    "Soggiorno breve ma intenso.",
]

STARTERS_COLLOQUIAL = [ # frasi di apertura con toni colloquiali
    "Lo scrivo adesso perchè è ancora fresco.",
    "Feedback al volo dopo il rientro.",
    "Recensione sincera, senza giri di parole.",
    "Parlo da ospite normale, non da cliente che cerca il pelo nell'uovo.",
    "Eravamo lì per un'occasione di famiglia e quindi ci siamo accorti di tutto.",
    "Viaggio corto ma abbastanza per capire come gira il posto.",
]

POS_SENTENCES = [ # frasi positive standard esplicite
    "Nel complesso mi sono trovato bene.",
    "Esperienza positiva e abbastanza lineare.",
    "Direi promosso, anche oltre le aspettative.",
    "In generale il servizio ha funzionato.",
    "Bilancio finale buono, tornerei.",
]

NEG_SENTENCES = [ # frasi negative standard esplicite
    "Nel complesso esperienza sotto le aspettative.",
    "Onestamente ci sono stati troppi intoppi.",
    "Servizio da rivedere su piu punti.",
    "Bilancio finale deludente.",
    "Mi aspettavo una gestione migliore.",
]

POS_SENTENCES_COLLOQUIAL = [ # frasi positive con toni colloquiali espliciti
    "Alla fine ci siamo trovati proprio bene.",
    "Esperienza bella piena e senza brutte sorprese.",
    "Per noi è andata bene davvero.",
    "Onestamente lo consiglierei senza problemi.",
    "Siamo usciti contenti e si vede.",
]

NEG_SENTENCES_COLLOQUIAL = [ # frasi negative con toni colloquiali esplicite
    "Alla fine ci è rimasta una brutta sensazione.",
    "Ci sono stati troppi problemi per far finta di niente.",
    "Non è andata bene, punto.",
    "Per quello che abbiamo vissuto non posso parlarne bene.",
    "Sinceramente così non ci siamo.",
]

SOFT_NEG = [ # frasi negative con toni standard per smorzare l'impatto e creare ambiguità
    "peccato per alcuni dettagli da sistemare",
    "in certe fasce orarie il servizio rallenta",
    "qualche passaggio è stato poco chiaro",
    "restano margini di miglioramento",
]

SOFT_POS = [ # frasi positive con toni standard per smorzare l'entusiasmo e creare ambiguità
    "va detto che lo staff è stato gentile",
    "comunque su altri aspetti si sono mossi bene",
    "almeno il personale ha provato a rimediare",
    "una parte del servizio è stata curata",
]

SOFT_NEG_COLLOQUIAL = [ # frasi negative con toni colloquiali per smorzare l'impatto e creare ambiguità
    "qualche inciampo c'è stato",
    "non tutto ha girato liscio",
    "si poteva fare meglio in alcuni punti",
    "non è stato perfetto fino in fondo",
]

SOFT_POS_COLLOQUIAL = [ # frasi positive con toni colloquiali per smorzare l'entusiasmo e creare ambiguità
    "per onestà qualcosa di buono c'era",
    "una parte del team ha provato davvero a metterci una pezza",
    "non tutto era da buttare",
    "su un paio di cose si sono comportati bene",
]

TITLE_POS = [ # titoli positivi
    "Soggiorno piacevole",
    "Meglio del previsto",
    "Buona esperienza",
    "Nel complesso promosso",
    "Weekend positivo",
]

TITLE_NEG = [ # titoli negativi
    "Poteva andare meglio",
    "Qualche problema di troppo",
    "Sotto le aspettative",
    "Esperienza da rivedere",
    "Non proprio convincente",
]

TITLE_NEUTRAL = [ # titoli neutrali
    "Feedback sul soggiorno",
    "Recensione post viaggio",
    "Commento rapido",
    "Appunti sul soggiorno",
    "Valutazione generale",
]

TITLE_POS_COLLOQUIAL = [ # titoli positivi con toni più colloquiali
    "Ci siamo trovati bene",
    "Posto promosso sul serio",
    "Esperienza bella piena",
    "Torneremmo volentieri",
    "Piacevole davvero",
]

TITLE_NEG_COLLOQUIAL = [ # titoli negativi con toni più colloquiali
    "Così non va",
    "Esperienza storta",
    "Troppi problemi tutti insieme",
    "Non ci siamo proprio",
    "Male più del previsto",
]

TITLE_NEUTRAL_COLLOQUIAL = [ # titoli neutrali con toni più colloquiali
    "Due righe sincere",
    "Com'è andata davvero",
    "Feedback senza filtri",
    "Recensione schietta",
    "Com'era il soggiorno",
]

DEPARTMENT_TITLE_HINTS = { # frasi da inserire nei titoli per suggerire il reparto coinvolto, aumentando la coerenza testo-etichetta
    "Housekeeping": ["in camera", "sulla pulizia", "sulla stanza", "su bagno e riordino"],
    "Reception": ["alla reception", "su check in/out", "sull'accoglienza", "su arrivo e partenza"],
    "F&B": ["su colazione e ristorante", "in ristorazione", "sul buffet", "su cena e sala"],
}

POS_DEPT_TEMPLATES = [ # template positivi per descrivere esperienze legate ai reparti
    "Per {t1} e {t2} davvero bene, con {sv} gestito in modo ordinato.",
    "Su {t1} e {t2} nessun problema, bene anche il lato {sv}.",
    "Ho apprezzato {t1} e {t2}; sul fronte {sv} il team e stato efficace.",
    "La parte {t1}/{t2} mi ha convinto, soprattutto per {sv}.",
    "Niente da dire su {t1} e {t2}, anche {sv} in linea.",
]

NEG_DEPT_TEMPLATES = [ # template negative per descrivere esperienze legate ai reparti
    "Su {t1} e {t2} ci sono stati problemi, e il lato {sv} e da rivedere.",
    "La parte {t1}/{t2} non ha funzionato bene, soprattutto su {sv}.",
    "Ho trovato criticita su {t1} e {t2}, con gestione {sv} poco efficace.",
    "Per {t1} e {t2} mi aspettavo molto di piu, male anche su {sv}.",
    "Diversi intoppi su {t1} e {t2}; il fronte {sv} e rimasto debole.",
]

CROSS_DEPT_TEMPLATES = [ # template standard per inserire temi di altri reparti
    "Tra l'altro e uscito anche un tema su {x1} e {x2}, quindi il caso tocca piu reparti.",
    "A margine segnalo anche {x1} e {x2}, non solo l'area principale.",
    "Si sono incrociati pure aspetti su {x1} e {x2}, da coordinare internamente.",
    "Oltre a questo, ho notato problemi anche su {x1} e {x2}.",
]

CROSS_DEPT_TEMPLATES_COLLOQUIAL = [ # template colloquiali per inserire temi di altri reparti
    "Tra una cosa e l'altra sono saltati fuori pure {x1} e {x2}, quindi non è solo colpa di un reparto.",
    "Non c'era solo il tema principale: si sono messi in mezzo anche {x1} e {x2}.",
    "A cascata ho visto pure questioni su {x1} e {x2}.",
    "Il punto è che oltre al resto si sono incrociati pure {x1} e {x2}.",
]

CLOSINGS_POS = [ # frasi di chiusura positive
    "Se mantengono questo livello, ci tornerei.",
    "Per me esperienza complessivamente valida.",
    "Direi che il soggiorno è stato centrato.",
]

CLOSINGS_NEG = [ # frasi di chiusura negative
    "Spero migliorino perché la struttura ha potenziale.",
    "Non escludo un ritorno, ma solo con miglioramenti concreti.",
    "Al momento non mi sento di consigliarlo senza riserve.",
]

CLOSINGS_POS_COLLOQUIAL = [ # frasi di chiusura positive con toni colloquiali
    "Per noi è promosso e ci tornerei volentieri.",
    "Se resta così, lo consiglio sereno.",
    "Alla fine ci siamo alzati dal tavolo e dalla stanza contenti.",
]

CLOSINGS_NEG_COLLOQUIAL = [ # frasi di chiusura negative con toni colloquiali
    "Peccato, perché il posto avrebbe pure del potenziale.",
    "Così faccio fatica a consigliarlo ad amici o parenti.",
    "Ad oggi non me la sento di parlarne bene senza riserve.",
]

DEPARTMENT_STORIES = { # frasi che descrivono in modo articolato esperienze legate al reparto
    "Housekeeping": {
        "pos": [
            "Camera pulita sul serio, letto comodo e bagno sistemato come si deve già dal primo pomeriggio.",
            "Appena entrati si sentiva odore di pulito, asciugamani in ordine e stanza curata senza cose lasciate a metà.",
            "Riordino fatto bene ogni giorno, niente polvere sugli appoggi e doccia finalmente in ordine.",
            "Si dormiva bene perché stanza, cuscini e aria condizionata erano messi a posto come ci si aspetta.",
            "Bagno e camera erano davvero a livello: tutto pulito, niente odori strani e manutenzione fatta bene.",
            "Per la parte camera non ho da lamentarmi: letto rifatto bene, bagno pulito e ambiente curato.",
        ],
        "neg": [
            "In stanza abbiamo trovato polvere, bagno sistemato male e un odore che non passava nemmeno con la finestra aperta.",
            "La camera sembrava tirata via: lenzuola discutibili, asciugamani contati e doccia da rivedere.",
            "Riordino fatto male, letto sistemato in fretta e bagno lasciato con dettagli che non dovrebbero esserci.",
            "La notte è stata pesante per via di rumore, aria condizionata ballerina e sensazione generale di poca cura.",
            "Pulizia sotto tono: polvere sugli appoggi, bagno poco convincente e manutenzione lenta.",
            "Per la parte housekeeping male: stanza non davvero pronta, odore fastidioso e comfort lontano da quello promesso.",
        ],
    },
    "Reception": {
        "pos": [
            "Alla reception ci hanno preso in carico subito, con spiegazioni chiare e zero confusione.",
            "Check in rapido, chiave pronta e staff sveglio anche quando siamo arrivati tardi.",
            "Su prenotazione, documenti e parcheggio e filato tutto liscio senza code inutili.",
            "Check out veloce e fattura corretta al primo colpo, che non è scontato.",
            "Per arrivo e partenza davvero bene: reception presente, disponibile e chiara.",
            "Ci hanno gestito bene dal primo minuto, anche con una richiesta extra sul late check out.",
        ],
        "neg": [
            "Alla reception c'era confusione, attesa lunga e informazioni cambiate più di una volta.",
            "Check in lento, camera non pronta e prenotazione gestita con troppa approssimazione.",
            "Per una semplice fattura ci hanno fatto tornare più volte al desk senza risolvere subito.",
            "Staff cortese a parole ma poco concreto su chiave, documenti e richiesta di supporto.",
            "Arrivo e partenza gestiti male: coda, poca chiarezza e tempo perso inutilmente.",
            "Sul fronte reception male soprattutto per l'attesa e per come hanno gestito la prenotazione.",
        ],
    },
    "F&B": {
        "pos": [
            "A colazione e a cena si mangiava bene davvero, con porzioni giuste e servizio presente.",
            "Abbiamo mangiato assai e bene: piatti curati, sala tranquilla e tempi sensati.",
            "Buffet ricco il giusto, cornetti buoni, cappuccino fatto bene e personale di sala sul pezzo.",
            "Per il compleanno hanno gestito torta e tavolo senza fare confusione, e tutti sono rimasti contenti.",
            "Ristorante promosso: menu chiaro, piatti saporiti e cameriere attento senza essere invadente.",
            "Si mangia bene sul serio, dalla colazione all'aperitivo, con buona attenzione anche alle esigenze alimentari.",
        ],
        "neg": [
            "A tavola male: piatti tiepidi, attese lunghe e servizio che si perdeva tra un tavolo e l'altro.",
            "Colazione deludente, buffet poco curato e nessuno che chiarisse bene le opzioni per allergie o intolleranze.",
            "Abbiamo mangiato male perché tra porzioni sbagliate, tempi lunghi e sala confusa non ci siamo proprio.",
            "Cena da rivedere: cameriere in affanno, piatti arrivati freddi e conto gestito male.",
            "Sul lato ristorazione esperienza fiacca, con menu poco chiaro e qualità sotto tono.",
            "Per pranzo e cena ci aspettavamo altro: tavolo dimenticato, bevande lente e servizio troppo approssimativo.",
        ],
    },
}

REAL_LIFE_CONTEXTS = { # frasi che descrivono situazioni di vita reale
    "Housekeeping": {
        "pos": [
            "Eravamo con un bimbo piccolo e poter entrare in una stanza già davvero in ordine ha fatto la differenza.",
            "Siamo rientrati tardi e trovare camera e bagno ancora messi bene è stato un sollievo.",
            "Con il caldo di questi giorni avere aria condizionata e letto in ordine ci ha rimesso al mondo.",
        ],
        "neg": [
            "Con una bambina piccola una stanza così poco curata si nota subito e pesa parecchio.",
            "Dopo una giornata intera fuori volevamo solo una camera a posto, e invece abbiamo trovato disordine e odori.",
            "Arrivando stanchi la sera, trovare il bagno in quelle condizioni ha peggiorato tutto il soggiorno.",
        ],
    },
    "Reception": {
        "pos": [
            "Siamo arrivati stanchi e in ritardo, ma ci hanno sbrigato tutto senza farci perdere altro tempo.",
            "Avevamo un treno presto e il check out rapido ci ha salvato la mattinata.",
            "Con varie richieste al volo hanno risposto bene e senza mandarci avanti e indietro.",
        ],
        "neg": [
            "Dopo il viaggio non avevamo energie per fare fila al desk, ma è andata proprio così.",
            "Con le valigie e la macchina da sistemare l'attesa in reception è pesata il doppio.",
            "Quando hai una prenotazione già confermata, dover ridire tutto tre volte stanca subito.",
        ],
    },
    "F&B": {
        "pos": [
            "Eravamo a festeggiare un compleanno e la nonna è rimasta contenta di torta, tavolo e atmosfera.",
            "A pranzo ci siamo seduti affamati e ci siamo alzati davvero soddisfatti.",
            "Con amici e famiglia ci siamo trovati bene perché si mangiava bene e la sala girava liscia.",
        ],
        "neg": [
            "Eravamo a tavola per stare tranquilli e invece tra attese e piatti sbagliati si è rovinato il momento.",
            "Per un pranzo di famiglia ci aspettavamo più cura, non certo tempi morti e portate tiepide.",
            "Con una persona anziana al tavolo il servizio lento e la confusione si sono sentiti ancora di più.",
        ],
    },
}

SAFETY_NEG_TITLES = [ # frasi che indicano chiaramente problemi di sicurezza, per creare casi critici
    "Segnalazione grave igiene",
    "Situazione inaccettabile",
    "Reclamo urgente",
    "Esperienza sanitaria critica",
]

SAFETY_POS_NEGATED_TITLES = [ # frasi che negano esplicitamente problemi di sicurezza, per creare casi ambigui
    "Controllo igiene ok",
    "Nessuna criticità sanitaria",
    "Verifica condizioni positive",
]

SAFETY_NEG_TEMPLATES = { # frasi che descrivono situazioni di sicurezza gravi, per creare casi critici
    "Housekeeping": [
        "In camera c'erano acari e in bagno ho trovato muffa vicino alla doccia. La notte siamo stati morsi e la pulizia è stata pessima.",
        "Materasso pieno di acari, odore forte in stanza e cimici da letto. Situazione insalubre e inaccettabile.",
        "Asciugamani sporchi, bagno con muffa e odore nauseante. Igiene pessima per tutta la notte.",
    ],
    "Reception": [
        "Check in rapido ma alla reception non hanno gestito una segnalazione di scarafaggi in camera. Attesa lunga e assistenza deludente.",
        "Alla reception abbiamo segnalato odore forte e muffa in stanza, ma non è stato fatto nulla. Gestione problema insufficiente.",
        "Reception cortese ma inefficace: nonostante le foto di cimici da letto, nessun intervento e nessun cambio camera.",
    ],
    "F&B": [
        "In cucina ho visto scarafaggi in giro e il cuoco aveva peli sul camice. Odore nauseante e condizioni igieniche pessime.",
        "Durante la colazione c'era puzza forte, piatti non puliti e blatte vicino al buffet. Esperienza disgustosa.",
        "Al ristorante situazione grave: scarafaggi in cucina, personale trascurato e igiene insalubre.",
    ],
}

SAFETY_POS_NEGATED_TEMPLATES = { # frasi che negano esplicitamente problemi di sicurezza, per creare casi ambigui
    "Housekeeping": [
        "Camera e bagno molto puliti, nessuna muffa e nessun odore forte. Materasso comodo e notte tranquilla.",
        "Controllo accurato: nessun acaro visibile, nessuna cimice e ottima igiene in stanza.",
    ],
    "Reception": [
        "Reception efficiente: segnalazioni gestite subito, nessuna criticità su igiene camera e supporto rapido.",
        "Check in e assistenza ottimi, nessun problema sanitario riscontrato e comunicazione chiara.",
    ],
    "F&B": [
        "Colazione ben organizzata, nessun odore forte e cucina visibilmente pulita.",
        "Ristorante curato: nessuno scarafaggio, nessun problema igienico e personale professionale.",
    ],
}

REALISTIC_TYPOS_MULTI = [ # errori di battitura comuni su parole chiave
    ("check in", "checkin"),
    ("check out", "checkout"),
    ("aria condizionata", "aria condizionta"),
]

REALISTIC_TYPOS_WORD = { # singoli errori di battitura
    "servizio": "servizo",
    "camera": "camra",
    "colazione": "colazzione",
    "prenotazione": "prenotazone",
    "personale": "personle",
    "fattura": "fatura",
    "pulizia": "puliza",
    "ristorante": "ristornte",
    "asciugamani": "asciugamni",
    "problemi": "problemii",
}


def _parametri_profilo(profilo: str) -> dict[str, float]:
    """Restituisce i parametri di generazione per il profilo richiesto."""
    if profilo == "train":
        return {
            "ambiguity_prob": 0.24,
            "mixed_prob": 0.35,
            "noise_prob": 0.00,
            "neutral_title_prob": 0.52,
            "title_hint_prob": 0.22,
            "colloquial_prob": 0.48,
            "story_prob": 0.40,
            "context_prob": 0.34,
            "dep_flip": 0.03,
            "sent_flip": 0.06,
        }
    if profilo == "ambiguous":
        return {
            "ambiguity_prob": 0.62,
            "mixed_prob": 0.70,
            "noise_prob": 0.03,
            "neutral_title_prob": 0.70,
            "title_hint_prob": 0.16,
            "colloquial_prob": 0.58,
            "story_prob": 0.48,
            "context_prob": 0.44,
            "dep_flip": 0.10,
            "sent_flip": 0.13,
        }
    if profilo == "colloquial":
        return {
            "ambiguity_prob": 0.32,
            "mixed_prob": 0.28,
            "noise_prob": 0.04,
            "neutral_title_prob": 0.78,
            "title_hint_prob": 0.10,
            "colloquial_prob": 0.88,
            "story_prob": 0.76,
            "context_prob": 0.68,
            "dep_flip": 0.00,
            "sent_flip": 0.00,
        }
    return {
        "ambiguity_prob": 0.46,
        "mixed_prob": 0.55,
        "noise_prob": 0.18,
        "neutral_title_prob": 0.62,
        "title_hint_prob": 0.18,
        "colloquial_prob": 0.54,
        "story_prob": 0.44,
        "context_prob": 0.38,
        "dep_flip": 0.10,
        "sent_flip": 0.18,
    }


def _inietta_rumore_realistico(testo: str, generatore: random.Random, livello: float = 0.12) -> str:
    """Inserisce typo e piccole anomalie lessicali per simulare rumore reale."""
    out = testo

    for src, dst in REALISTIC_TYPOS_MULTI:
        if src in out and generatore.random() < livello:
            out = out.replace(src, dst, 1)

    words = out.split()
    for i, w in enumerate(words):
        base = w.strip(".,;:!?").lower()
        if base in REALISTIC_TYPOS_WORD and generatore.random() < livello:
            trailing = ""
            if w and w[-1] in ".,;:!?":
                trailing = w[-1]
            words[i] = REALISTIC_TYPOS_WORD[base] + trailing

    out = " ".join(words)

    if generatore.random() < livello:
        out = out.replace(",", "", 1)
    if generatore.random() < livello * 0.7:
        out = out.replace("  ", " ")

    return out


def _scegli_titolo(sentiment: str, reparto: str, profilo: str, generatore: random.Random) -> str:
    """Costruisce un titolo coerente con sentiment, reparto e profilo di generazione."""
    parametri = _parametri_profilo(profilo)

    use_colloquial = generatore.random() < parametri["colloquial_prob"]
    if generatore.random() < parametri["neutral_title_prob"]:
        base_pool = TITLE_NEUTRAL_COLLOQUIAL if use_colloquial else TITLE_NEUTRAL
    else:
        if sentiment == "pos":
            base_pool = TITLE_POS_COLLOQUIAL if use_colloquial else TITLE_POS
        else:
            base_pool = TITLE_NEG_COLLOQUIAL if use_colloquial else TITLE_NEG

    base = generatore.choice(base_pool)
    if generatore.random() < parametri["title_hint_prob"]:
        return f"{base} - {generatore.choice(DEPARTMENT_TITLE_HINTS[reparto])}"
    return base


def _costruisci_frase_reparto(reparto: str, sentiment: str, profilo: str, generatore: random.Random) -> str:
    """Genera la frase principale legata al reparto target."""
    parametri = _parametri_profilo(profilo)
    if generatore.random() < parametri["story_prob"]:
        return generatore.choice(DEPARTMENT_STORIES[reparto][sentiment])

    topic = DEPARTMENT_TOPICS[reparto]
    t1, t2 = generatore.sample(topic["core"], k=2)
    sv = generatore.choice(topic["service"])

    if sentiment == "pos":
        tpl = generatore.choice(POS_DEPT_TEMPLATES)
    else:
        tpl = generatore.choice(NEG_DEPT_TEMPLATES)

    return tpl.format(t1=t1, t2=t2, sv=sv)


def _costruisci_frase_contesto(reparto: str, sentiment: str, profilo: str, generatore: random.Random) -> str:
    """Aggiunge un contesto realistico di uso dell'hotel."""
    parametri = _parametri_profilo(profilo)
    if generatore.random() >= parametri["context_prob"]:
        return ""
    return generatore.choice(REAL_LIFE_CONTEXTS[reparto][sentiment])


def _costruisci_frase_sentiment(sentiment: str, profilo: str, generatore: random.Random) -> str:
    """Aggiunge una frase che esplicita il tono generale della recensione."""
    parametri = _parametri_profilo(profilo)
    if generatore.random() < parametri["colloquial_prob"]:
        pool = POS_SENTENCES_COLLOQUIAL if sentiment == "pos" else NEG_SENTENCES_COLLOQUIAL
    else:
        pool = POS_SENTENCES if sentiment == "pos" else NEG_SENTENCES
    return generatore.choice(pool)


def _costruisci_frase_mista(sentiment: str, profilo: str, generatore: random.Random) -> str:
    """Inserisce un contrasto semantico per rendere il testo meno lineare."""
    parametri = _parametri_profilo(profilo)
    use_colloquial = generatore.random() < parametri["colloquial_prob"]
    if sentiment == "pos":
        pool = SOFT_NEG_COLLOQUIAL if use_colloquial else SOFT_NEG
        return f"Detto questo, {generatore.choice(pool)}."
    pool = SOFT_POS_COLLOQUIAL if use_colloquial else SOFT_POS
    return f"Per correttezza, {generatore.choice(pool)}."


def _costruisci_frase_interreparto(reparto_principale: str, profilo: str, generatore: random.Random) -> str:
    """Aggiunge un tema secondario di un altro reparto per creare ambiguità controllata."""
    altro_reparto = generatore.choice([reparto for reparto in DEPARTMENTS if reparto != reparto_principale])
    x1, x2 = generatore.sample(DEPARTMENT_TOPICS[altro_reparto]["core"], k=2)
    parametri = _parametri_profilo(profilo)
    templates = CROSS_DEPT_TEMPLATES_COLLOQUIAL if generatore.random() < parametri["colloquial_prob"] else CROSS_DEPT_TEMPLATES
    return generatore.choice(templates).format(x1=x1, x2=x2)


def _costruisci_chiusura(sentiment: str, profilo: str, generatore: random.Random) -> str:
    """Chiude la recensione con una frase coerente con la polarità complessiva."""
    parametri = _parametri_profilo(profilo)
    if generatore.random() < parametri["colloquial_prob"]:
        pool = CLOSINGS_POS_COLLOQUIAL if sentiment == "pos" else CLOSINGS_NEG_COLLOQUIAL
    else:
        pool = CLOSINGS_POS if sentiment == "pos" else CLOSINGS_NEG
    return generatore.choice(pool)


def _costruisci_corpo(reparto: str, sentiment: str, profilo: str, generatore: random.Random) -> str:
    """Compone il corpo della recensione usando segnali principali, contesto, ambiguità e rumore."""
    parametri = _parametri_profilo(profilo)

    chunks = []
    starters_pool = STARTERS_COLLOQUIAL if generatore.random() < parametri["colloquial_prob"] else STARTERS
    if generatore.random() < 0.78:
        chunks.append(generatore.choice(starters_pool))

    chunks.append(_costruisci_frase_reparto(reparto=reparto, sentiment=sentiment, profilo=profilo, generatore=generatore))

    contesto = _costruisci_frase_contesto(reparto=reparto, sentiment=sentiment, profilo=profilo, generatore=generatore)
    if contesto:
        chunks.append(contesto)

    chunks.append(_costruisci_frase_sentiment(sentiment=sentiment, profilo=profilo, generatore=generatore))

    if generatore.random() < parametri["mixed_prob"]:
        chunks.append(_costruisci_frase_mista(sentiment=sentiment, profilo=profilo, generatore=generatore))

    if generatore.random() < parametri["ambiguity_prob"]:
        chunks.append(_costruisci_frase_interreparto(reparto_principale=reparto, profilo=profilo, generatore=generatore))

    if generatore.random() < 0.62:
        chunks.append(_costruisci_chiusura(sentiment=sentiment, profilo=profilo, generatore=generatore))

    body = " ".join([chunk for chunk in chunks if chunk])

    if parametri["noise_prob"] > 0 and generatore.random() < parametri["noise_prob"]:
        body = _inietta_rumore_realistico(body, generatore, livello=0.13)

    return body


def _introduci_rumore_etichetta(
    reparto: str,
    sent: str,
    profilo: str,
    generatore: random.Random,
) -> tuple[str, str]:
    """Introduce label noise controllato in base al profilo del dataset."""
    parametri = _parametri_profilo(profilo)
    dep_flip = float(parametri["dep_flip"])
    sent_flip = float(parametri["sent_flip"])

    out_dep = reparto
    out_sent = sent

    if generatore.random() < dep_flip:
        out_dep = generatore.choice([d for d in DEPARTMENTS if d != reparto])
    if generatore.random() < sent_flip:
        out_sent = "neg" if sent == "pos" else "pos"

    return out_dep, out_sent


def genera_dataset(n: int, seed: int, profilo: str) -> pd.DataFrame:
    """Genera il dataset sintetico principale o i benchmark derivati."""
    rng = random.Random(seed)

    rows = []
    buckets = [(dep, sent) for dep in DEPARTMENTS for sent in SENTIMENTS]
    per_bucket = n // len(buckets)
    remainder = n % len(buckets)

    sample_plan = []
    for dep, sent in buckets:
        sample_plan.extend([(dep, sent)] * per_bucket)

    sample_plan.extend(rng.choices(buckets, k=remainder))
    rng.shuffle(sample_plan)

    for idx, (dep, sent) in enumerate(sample_plan, start=1):
        label_dep, label_sent = _introduci_rumore_etichetta(dep, sent, profilo=profilo, generatore=rng)

        rows.append(
            {
                "id": idx,
                "title": _scegli_titolo(sentiment=sent, reparto=dep, profilo=profilo, generatore=rng),
                "body": _costruisci_corpo(reparto=dep, sentiment=sent, profilo=profilo, generatore=rng),
                "department": label_dep,
                "sentiment": label_sent,
            }
        )

    return pd.DataFrame(rows)


def genera_dataset_safety_critico(n: int, seed: int) -> pd.DataFrame:
    """Genera il benchmark dedicato ai casi safety più severi."""
    rng = random.Random(seed)
    rows = []

    buckets = DEPARTMENTS.copy()
    per_bucket = n // len(buckets)
    remainder = n % len(buckets)

    plan = []
    for dep in buckets:
        plan.extend([dep] * per_bucket)
    plan.extend(rng.choices(buckets, k=remainder))
    rng.shuffle(plan)

    for idx, dep in enumerate(plan, start=1):
        severe_case = rng.random() < 0.80
        if severe_case:
            title = rng.choice(SAFETY_NEG_TITLES) + f" - {dep}"
            body = rng.choice(SAFETY_NEG_TEMPLATES[dep])
            sentiment = "neg"
        else:
            title = rng.choice(SAFETY_POS_NEGATED_TITLES) + f" - {dep}"
            body = rng.choice(SAFETY_POS_NEGATED_TEMPLATES[dep])
            sentiment = "pos"

        rows.append(
            {
                "id": idx,
                "title": title,
                "body": body,
                "department": dep,
                "sentiment": sentiment,
            }
        )

    return pd.DataFrame(rows)


def _stampa_riassunto(df: pd.DataFrame, etichetta: str) -> None:
    """Stampa un riepilogo compatto della distribuzione del dataset."""
    dep_dist = df["department"].value_counts().to_dict()
    sent_dist = df["sentiment"].value_counts().to_dict()
    print(f"[INFO] {etichetta}: righe={len(df)} | reparto={dep_dist} | sentiment={sent_dist}")


def esegui_generazione_dataset() -> None:
    """Legge gli argomenti CLI e genera dataset principale più benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=350, help="Numero recensioni dataset principale")
    parser.add_argument("--percorso_output", "--out", dest="percorso_output", type=str, default="data/reviews_synth.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cartella_benchmark", "--benchmarks_dir", dest="cartella_benchmark", type=str, default="data/benchmarks")
    parser.add_argument("--n_benchmark_id", "--bench_n_id", dest="n_benchmark_id", type=int, default=250)
    parser.add_argument("--n_benchmark_ambigui", "--bench_n_ambiguous", dest="n_benchmark_ambigui", type=int, default=220)
    parser.add_argument("--n_benchmark_rumorosi", "--bench_n_noisy", dest="n_benchmark_rumorosi", type=int, default=220)
    parser.add_argument("--n_benchmark_sicurezza", "--bench_n_safety", dest="n_benchmark_sicurezza", type=int, default=180)
    parser.add_argument("--n_benchmark_colloquiali", "--bench_n_colloquial", dest="n_benchmark_colloquiali", type=int, default=220)
    parser.add_argument("--salta_benchmark", "--skip_benchmarks", dest="salta_benchmark", action="store_true")
    args = parser.parse_args()

    if args.n < 200 or args.n > 600:
        print("[AVVISO] Pegaso consiglia 200-600 record per il dataset principale.")

    main_df = genera_dataset(n=args.n, seed=args.seed, profilo="train")

    garantisci_cartella_padre(args.percorso_output)
    main_df.to_csv(args.percorso_output, index=False)
    print(f"[OK] Dataset principale salvato: {args.percorso_output}")
    _stampa_riassunto(main_df, "main")

    if args.salta_benchmark:
        return

    bench_dir = Path(args.cartella_benchmark)
    bench_dir.mkdir(parents=True, exist_ok=True)

    id_df = genera_dataset(n=args.n_benchmark_id, seed=args.seed + 101, profilo="train")
    amb_df = genera_dataset(n=args.n_benchmark_ambigui, seed=args.seed + 202, profilo="ambiguous")
    noisy_df = genera_dataset(n=args.n_benchmark_rumorosi, seed=args.seed + 303, profilo="noisy")
    safety_df = genera_dataset_safety_critico(n=args.n_benchmark_sicurezza, seed=args.seed + 404)
    colloquial_df = genera_dataset(n=args.n_benchmark_colloquiali, seed=args.seed + 505, profilo="colloquial")

    id_path = bench_dir / "reviews_in_distribution.csv"
    amb_path = bench_dir / "reviews_ambiguous.csv"
    noisy_path = bench_dir / "reviews_noisy.csv"
    safety_path = bench_dir / "reviews_safety_critical.csv"
    colloquial_path = bench_dir / "reviews_colloquial.csv"

    id_df.to_csv(id_path, index=False)
    amb_df.to_csv(amb_path, index=False)
    noisy_df.to_csv(noisy_path, index=False)
    safety_df.to_csv(safety_path, index=False)
    colloquial_df.to_csv(colloquial_path, index=False)

    print(f"[OK] Suddivisioni benchmark salvate in: {bench_dir}")
    _stampa_riassunto(id_df, "benchmark_in_distribution")
    _stampa_riassunto(amb_df, "benchmark_ambiguous")
    _stampa_riassunto(noisy_df, "benchmark_noisy")
    _stampa_riassunto(safety_df, "benchmark_safety_critical")
    _stampa_riassunto(colloquial_df, "benchmark_colloquial")


if __name__ == "__main__":
    esegui_generazione_dataset()
