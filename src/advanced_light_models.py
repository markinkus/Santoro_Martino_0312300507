from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def _testo_sicuro(valore: Any) -> str:
    """Converte un valore arbitrario in una stringa monoriga non vuota."""
    s = str(valore or "")
    s = s.replace("\n", " ").strip()
    return s if s else "vuoto"


def arricchisci_testo_sentiment(testo: str) -> str:
    """Aggiunge token sentinella utili a rafforzare i segnali di sentiment."""
    s = str(testo or "")
    s_low = s.lower()
    tokens = []

    neg_cues = [
        "non",
        "mai",
        "nessun",
        "nessuna",
        "scarso",
        "scarsa",
        "pessimo",
        "deludente",
        "orrendo",
        "schifo",
        "male",
        "freddi",
        "freddo",
    ]
    pos_cues = [
        "ottimo",
        "ottima",
        "spettacolo",
        "top",
        "promosso",
        "promossa",
        "contento",
        "contenta",
        "gentile",
        "gentili",
        "buonissimo",
        "buonissima",
        "consiglio",
        "torneremo",
        "tornerei",
    ]
    contrast_cues = ["ma", "pero", "però", "detto questo", "tuttavia", "comunque"]
    intensifiers = ["molto", "troppo", "davvero", "estremamente", "super"]
    complaint_cues = [
        "problema",
        "criticita",
        "criticità",
        "intoppi",
        "attesa",
        "muffa",
        "allergie",
        "rumore",
        "scarafaggi",
        "freddi",
        "coda",
        "polvere",
    ]
    food_cues = [
        "mangia",
        "mangiato",
        "colazione",
        "cena",
        "pranzo",
        "ristorante",
        "cameriere",
        "torta",
        "compleanno",
        "buffet",
        "cornetti",
        "cappuccino",
    ]

    if any(f" {k} " in f" {s_low} " for k in neg_cues):
        tokens.append("__HAS_NEGATION__")
    if any(f" {k} " in f" {s_low} " for k in pos_cues):
        tokens.append("__HAS_POSITIVE_CUE__")
    if any(k in s_low for k in contrast_cues):
        tokens.append("__HAS_CONTRAST__")
    if any(f" {k} " in f" {s_low} " for k in intensifiers):
        tokens.append("__HAS_INTENSIFIER__")
    if any(k in s_low for k in complaint_cues):
        tokens.append("__HAS_COMPLAINT_CUE__")
    if any(k in s_low for k in food_cues):
        tokens.append("__HAS_FOOD_CONTEXT__")

    if tokens:
        return f"{s} {' '.join(tokens)}"
    return s


def arricchisci_testi_se_necessario(testi: np.ndarray, compito: str) -> np.ndarray:
    """Arricchisce i testi solo per il task sentiment, lasciando invariato il routing."""
    if compito != "sentiment":
        return np.array([_testo_sicuro(x) for x in testi], dtype=object)
    return np.array([arricchisci_testo_sentiment(_testo_sicuro(x)) for x in testi], dtype=object)


def punteggio_brier_multiclasse(y_true: np.ndarray, y_proba: np.ndarray, classi: list[str]) -> float:
    """Calcola il punteggio di Brier in scenario multiclasse."""
    class_to_idx = {c: i for i, c in enumerate(classi)}
    y_onehot = np.zeros((len(y_true), len(classi)), dtype=float)
    for i, y in enumerate(y_true):
        idx = class_to_idx.get(str(y))
        if idx is not None:
            y_onehot[i, idx] = 1.0
    return float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))


def errore_atteso_calibrazione(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classi: list[str],
    n_bins: int = 15,
) -> float:
    """Stima l'Expected Calibration Error sui bin di confidenza."""
    if len(y_true) == 0:
        return 0.0

    pred_idx = np.argmax(y_proba, axis=1)
    pred_labels = np.array([classi[i] for i in pred_idx], dtype=object)
    conf = np.max(y_proba, axis=1)
    correct = (pred_labels == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        lo = bins[b]
        hi = bins[b + 1]
        if b == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        if not np.any(mask):
            continue
        acc_bin = float(np.mean(correct[mask]))
        conf_bin = float(np.mean(conf[mask]))
        w = float(np.sum(mask)) / float(n)
        ece += abs(acc_bin - conf_bin) * w
    return float(ece)


def _ottieni_classi_modello(modello) -> list[str]:
    """Recupera l'elenco classi da un modello base o da una pipeline."""
    if hasattr(modello, "classes_"):
        return [str(c) for c in modello.classes_]
    if hasattr(modello, "named_steps") and "clf" in modello.named_steps:
        clf = modello.named_steps["clf"]
        if hasattr(clf, "classes_"):
            return [str(c) for c in clf.classes_]
    return []


def _allinea_probabilita(probabilita: np.ndarray, classi_origine: list[str], classi_destinazione: list[str]) -> np.ndarray:
    """Riordina una matrice di probabilità secondo l'ordine classi desiderato."""
    idx_map = [classi_origine.index(c) for c in classi_destinazione]
    return probabilita[:, idx_map]


class FastTextProbClassifier:
    def __init__(
        self,
        lr: float = 0.45,
        epoch: int = 35,
        word_ngrams: int = 2,
        dim: int = 80,
        minn: int = 2,
        maxn: int = 5,
        thread: int = 4,
        seed: int = 42,
        verbose: int = 0,
    ) -> None:
        self.lr = lr
        self.epoch = epoch
        self.word_ngrams = word_ngrams
        self.dim = dim
        self.minn = minn
        self.maxn = maxn
        self.thread = thread
        self.seed = seed
        self.verbose = verbose
        self.classes_: np.ndarray | None = None
        self._model = None

    def fit(self, texts: np.ndarray, y: np.ndarray) -> "FastTextProbClassifier":
        try:
            import fasttext  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(
                "fasttext non disponibile. Installa o riallinea le dipendenze base con: "
                "pip install -r requirements.txt"
            ) from exc

        classes = sorted({str(v) for v in y})
        self.classes_ = np.array(classes, dtype=object)

        fd, tmp_path = tempfile.mkstemp(prefix="ft_train_", suffix=".txt")
        os.close(fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for txt, label in zip(texts, y):
                    safe = _testo_sicuro(txt)
                    f.write(f"__label__{label} {safe}\n")

            self._model = fasttext.train_supervised(
                input=tmp_path,
                lr=self.lr,
                epoch=self.epoch,
                wordNgrams=self.word_ngrams,
                dim=self.dim,
                minn=self.minn,
                maxn=self.maxn,
                thread=self.thread,
                verbose=self.verbose,
                seed=self.seed,
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return self

    def predict_proba(self, texts: np.ndarray) -> np.ndarray:
        if self._model is None or self.classes_ is None:
            raise RuntimeError("FastTextProbClassifier non addestrato.")

        classes = [str(c) for c in self.classes_]
        n_classes = len(classes)
        out = np.zeros((len(texts), n_classes), dtype=float)

        safe_texts = [_testo_sicuro(t) for t in texts]
        labels_batch, probs_batch = self._model.predict(safe_texts, k=n_classes)

        class_to_idx = {f"__label__{c}": i for i, c in enumerate(classes)}
        for i, (labels, probs) in enumerate(zip(labels_batch, probs_batch)):
            for lab, prob in zip(labels, probs):
                idx = class_to_idx.get(str(lab))
                if idx is not None:
                    out[i, idx] = float(prob)

            row_sum = float(out[i].sum())
            if row_sum <= 0.0:
                out[i] = 1.0 / float(n_classes)
            else:
                out[i] = out[i] / row_sum
        return out

    def predict(self, texts: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(texts)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.get("_model")
        state["_model"] = None
        state["_model_bytes"] = None
        if model is not None:
            fd, tmp_path = tempfile.mkstemp(prefix="ft_pickle_", suffix=".bin")
            os.close(fd)
            try:
                model.save_model(tmp_path)
                with open(tmp_path, "rb") as f:
                    state["_model_bytes"] = f.read()
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        return state

    def __setstate__(self, state):
        model_bytes = state.pop("_model_bytes", None)
        self.__dict__.update(state)
        self._model = None
        if model_bytes:
            try:
                import fasttext  # type: ignore[import-not-found]
            except Exception as exc:
                raise RuntimeError(
                    "Impossibile deserializzare FastTextProbClassifier: installa prima "
                    "requirements.txt."
                ) from exc

            fd, tmp_path = tempfile.mkstemp(prefix="ft_unpickle_", suffix=".bin")
            os.close(fd)
            try:
                with open(tmp_path, "wb") as f:
                    f.write(model_bytes)
                self._model = fasttext.load_model(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)


def sonda_calibrazione_logistica_word(
    compito: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[str, dict[str, dict[str, float]]]:
    """Confronta sigmoid e isotonic per scegliere la calibrazione meno distorta."""
    if compito == "sentiment":
        base_est = LogisticRegression(
            max_iter=4000,
            solver="liblinear",
            C=0.9,
            class_weight={"neg": 1.35, "pos": 1.0},
            random_state=seed,
        )
    else:
        base_est = LogisticRegression(
            max_iter=4000,
            solver="liblinear",
            C=1.2,
            random_state=seed,
        )

    classi = sorted({str(v) for v in y_train})
    results: dict[str, dict[str, float]] = {}

    xtr = arricchisci_testi_se_necessario(x_train, compito=compito)
    xev = arricchisci_testi_se_necessario(x_val, compito=compito)

    for method in ["sigmoid", "isotonic"]:
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=1,
                        max_df=0.98,
                        lowercase=False,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=base_est,
                        method=method,
                        cv=3,
                    ),
                ),
            ]
        )

        model.fit(xtr, y_train)
        proba = model.predict_proba(xev)
        pred = model.predict(xev)

        acc = float(np.mean(pred == y_val))
        ece = errore_atteso_calibrazione(y_val, proba, classi=classi, n_bins=15)
        brier = punteggio_brier_multiclasse(y_val, proba, classi=classi)
        results[method] = {
            "accuracy": acc,
            "ece": ece,
            "brier": brier,
        }

    chosen = min(
        ["sigmoid", "isotonic"],
        key=lambda m: (results[m]["ece"], results[m]["brier"], -results[m]["accuracy"]),
    )
    return chosen, results


class StackedTextEnsemble:
    def __init__(
        self,
        compito: str,
        metodo_calibrazione: str,
        seed: int = 42,
    ) -> None:
        self.compito = compito
        self.metodo_calibrazione = metodo_calibrazione
        self.seed = seed
        self.classes_: np.ndarray | None = None
        self.base_models: dict[str, Any] = {}
        self.meta_model: LogisticRegression | None = None

    def _build_base_models(self) -> dict[str, Any]:
        if self.compito == "sentiment":
            logreg = LogisticRegression(
                max_iter=4500,
                solver="liblinear",
                C=0.9,
                class_weight={"neg": 1.35, "pos": 1.0},
                random_state=self.seed,
            )
            svc = LinearSVC(
                C=0.95,
                class_weight={"neg": 1.25, "pos": 1.0},
                random_state=self.seed,
            )
            word_cfg = dict(analyzer="word", ngram_range=(1, 2), min_df=1, max_df=0.98)
            char_cfg = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_df=1.0)
        else:
            logreg = LogisticRegression(
                max_iter=4500,
                solver="liblinear",
                C=1.2,
                random_state=self.seed,
            )
            svc = LinearSVC(
                C=1.1,
                random_state=self.seed,
            )
            word_cfg = dict(analyzer="word", ngram_range=(1, 2), min_df=1, max_df=0.98)
            char_cfg = dict(analyzer="char_wb", ngram_range=(4, 6), min_df=1, max_df=1.0)

        char_lr = Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=False, sublinear_tf=True, **char_cfg)),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=logreg,
                        method=self.metodo_calibrazione,
                        cv=3,
                    ),
                ),
            ]
        )

        word_svc = Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=False, sublinear_tf=True, **word_cfg)),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=svc,
                        method=self.metodo_calibrazione,
                        cv=3,
                    ),
                ),
            ]
        )

        fast = FastTextProbClassifier(seed=self.seed)
        return {
            "char_lr": char_lr,
            "word_svc": word_svc,
            "fasttext": fast,
        }

    def _stack_features(self, testi: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Modello non addestrato.")
        target_classes = [str(c) for c in self.classes_]

        blocks = []
        for _, model in self.base_models.items():
            proba = model.predict_proba(testi)
            src_classes = _ottieni_classi_modello(model)
            proba = _allinea_probabilita(proba, src_classes, target_classes)
            blocks.append(proba)
        return np.hstack(blocks)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_meta: np.ndarray,
        y_meta: np.ndarray,
    ) -> "StackedTextEnsemble":
        self.classes_ = np.array(sorted({str(v) for v in y_train}), dtype=object)
        self.base_models = self._build_base_models()

        xtr = arricchisci_testi_se_necessario(x_train, compito=self.compito)
        xmeta = arricchisci_testi_se_necessario(x_meta, compito=self.compito)

        for model in self.base_models.values():
            model.fit(xtr, y_train)

        meta_x = self._stack_features(xmeta)
        self.meta_model = LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            random_state=self.seed,
            multi_class="auto",
        )
        self.meta_model.fit(meta_x, y_meta)
        return self

    def predict_proba(self, testi: np.ndarray) -> np.ndarray:
        if self.meta_model is None:
            raise RuntimeError("Modello non addestrato.")
        xt = arricchisci_testi_se_necessario(testi, compito=self.compito)
        meta_x = self._stack_features(xt)
        proba = self.meta_model.predict_proba(meta_x)

        meta_classes = [str(c) for c in self.meta_model.classes_]
        target_classes = [str(c) for c in self.classes_]
        return _allinea_probabilita(proba, meta_classes, target_classes)

    def predict(self, testi: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(testi)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


@dataclass
class SplitConformal:
    alpha: float = 0.10
    method: str = "aps"  # "aps" | "threshold"
    classes_: list[str] | None = None
    qhat: float | None = None
    threshold: float | None = None

    def fit(self, y_true: np.ndarray, y_proba: np.ndarray, classi: list[str]) -> "SplitConformal":
        self.classes_ = [str(c) for c in classi]
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        scores = []
        if self.method == "aps":
            for y, row in zip(y_true, y_proba):
                idx_true = class_to_idx[str(y)]
                order = np.argsort(-row)
                row_sorted = row[order]
                cum = np.cumsum(row_sorted)
                rank = int(np.where(order == idx_true)[0][0])
                scores.append(float(cum[rank]))
        else:
            for y, row in zip(y_true, y_proba):
                idx = class_to_idx[str(y)]
                scores.append(1.0 - float(row[idx]))
        scores_np = np.array(scores, dtype=float)

        n = len(scores_np)
        q_level = np.ceil((n + 1) * (1.0 - self.alpha)) / max(1, n)
        q_level = float(np.clip(q_level, 0.0, 1.0))
        self.qhat = float(np.quantile(scores_np, q_level, method="higher"))
        if self.method == "aps":
            self.threshold = None
        else:
            self.threshold = float(1.0 - self.qhat)
        return self

    def predict_sets(self, y_proba: np.ndarray) -> list[list[str]]:
        if self.classes_ is None or self.qhat is None:
            raise RuntimeError("SplitConformal non fitted.")

        out: list[list[str]] = []
        if self.method == "aps":
            for row in y_proba:
                order = np.argsort(-row)
                row_sorted = row[order]
                cum = np.cumsum(row_sorted)
                keep = int(np.searchsorted(cum, self.qhat, side="left")) + 1
                keep = max(1, min(keep, len(order)))
                labels = [self.classes_[i] for i in order[:keep]]
                out.append(labels)
        else:
            if self.threshold is None:
                raise RuntimeError("SplitConformal threshold non impostata.")
            for row in y_proba:
                labels = [self.classes_[i] for i, p in enumerate(row) if float(p) >= self.threshold]
                if not labels:
                    labels = [self.classes_[int(np.argmax(row))]]
                out.append(labels)
        return out

    def evaluate(self, y_true: np.ndarray, pred_sets: list[list[str]]) -> dict[str, float]:
        if len(y_true) == 0:
            return {"coverage": 0.0, "avg_set_size": 0.0}
        covered = [str(y) in s for y, s in zip(y_true, pred_sets)]
        sizes = [len(s) for s in pred_sets]
        return {
            "coverage": float(np.mean(covered)),
            "avg_set_size": float(np.mean(sizes)),
        }
