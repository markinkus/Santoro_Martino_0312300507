import re
import unicodedata

def normalizza_testo(testo: str | None) -> str:
    """Normalizza un testo libero per i task di classificazione.

    Parametri:
    - testo: contenuto testuale da pulire; può essere `None`.

    Restituisce:
    - una stringa in minuscolo, con spazi regolarizzati e caratteri non utili rimossi.
    """
    if testo is None:
        return ""
    testo = testo.strip().lower()
    testo = unicodedata.normalize("NFKC", testo)
    # Rimuovo segni non informativi, mantenendo lettere, numeri e spazi.
    testo = re.sub(r"[^a-z0-9àèìòù\s]", " ", testo)
    testo = re.sub(r"\s+", " ", testo).strip()
    return testo

def unisci_titolo_corpo(titolo: str | None, corpo: str | None, peso_corpo: int = 2) -> str:
    """Unisce titolo e corpo della recensione in un solo testo.

    Parametri:
    - titolo: intestazione o titolo della recensione.
    - corpo: contenuto principale della recensione.
    - peso_corpo: numero di volte in cui il corpo viene replicato nel testo finale.
      Valori > 1 danno priorita al contenuto principale rispetto al titolo.

    Restituisce:
    - una singola stringa pronta per il preprocessing.
    """
    titolo = titolo or ""
    corpo = corpo or ""
    parti = []
    if titolo.strip():
        parti.append(titolo.strip())
    if corpo.strip():
        parti.extend([corpo.strip()] * max(1, int(peso_corpo)))
    return " ".join(parti).strip()
