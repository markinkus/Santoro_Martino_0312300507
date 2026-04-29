# Da recensione a decisione operativa: prototipo ML per la gestione dei feedback alberghieri
Repository del ProjectWork n.20 realizzato dallo studente Martino Santoro per la laurea triennale Pegaso: sistema ML per il routing di recensioni hotel e la stima del sentiment, pensato per essere usato in due modi:
- esecuzione immediata con i modelli base già inclusi;
- rigenerazione guidata di dataset, benchmark, modelli `pure`/`hardened`, `advanced_aps_*` ed esperimenti.

Questa è la guida per l'avvio e la navigazione del repository.
## Avvio rapido
```bash
cd github_project_only
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/04_app_streamlit.py
```

Dopo l'avvio la dashboard espone tre aree:
- `Predizione singola`: prova manuale con lettura guidata di reparto, sentiment, priorità, rischio e guardrail;
- `Predizione batch (CSV)`: demo integrata con CSV già pronto oppure upload di CSV personalizzati;
- `Strumenti e rigenerazione`: pulsanti guidati per ricreare dataset, modelli ed esperimenti.

I comandi di avvio qui sopra installano il profilo base, sufficiente per:
- dashboard Streamlit;
- predizione batch o singola;
- rigenerazione dataset e benchmark;
- training e valutazione della base `pure` e `hardened`;
- training e caricamento dei profili `advanced_aps_pure` e `advanced_aps_hardened`.

## Percorso consigliato per test rapido
1. Avviare la dashboard con il comando sopra.
2. Aprire `Predizione batch (CSV)`.
3. Premere `Mostra CSV demo` per vedere [reviews_trappola_demo.csv](data/demo/reviews_trappola_demo.csv).
4. Premere `Esegui demo batch` per lanciare subito la predizione sul CSV incluso.
5. Leggere la spiegazione dei KPI operativi, l'output tabellare e la simulazione SLA.
6. In alternativa, caricare un CSV personale e premere `Esegui batch su CSV caricato`.
7. Se serve, usare `Strumenti e rigenerazione` per ricreare base, benchmark o run avanzati.

In alternativa, da terminale:
```bash
python3 src/03_predict_batch.py \
  --percorso_input data/demo/reviews_trappola_demo.csv \
  --revisione_diagnostica
```

Extra opzionale per eseguire il confronto con MiniLM (configurabile anche da dashboard):
```bash
pip install -r requisiti-transformer.txt
```
Il file `requisiti-transformer.txt` forza `torch==2.2.2+cpu` dalla wheel CPU-only, così il confronto non installa pacchetti CUDA inutili.

## Cosa contiene il repository
- [src](src/): codice applicativo, training base, dashboard e script di riproduzione degli esperimenti;
- [data](data/): dataset principale, benchmark separati, demo CSV e code etichettate per active learning;
- [models](models/): modelli base pronti all'uso;
- [outputs](outputs/): soglie runtime e alcuni output di esempio già leggibili.

## Profili di dipendenze
- `requirements.txt`: setup consigliato per dashboard, demo, dataset, base e `advanced_aps_*`;
- `requisiti-transformer.txt`: extra per [06_compare_transformer.py](src/06_compare_transformer.py), con `torch` CPU-only per evitare dipendenze CUDA inutili.

## Stato dei modelli
- `baseline_pure`: safety benchmark solo held-out;
- `baseline_hardened`: safety augmentation controllata con `repeat=2`;
- `advanced_aps_pure` e `advanced_aps_hardened`: profili ensemble/conformal separati;
- `active_learning_oracle`, `active_learning_v2_no_replay`, `active_learning_v2_replay`: si riproducono a partire dalle code etichettate incluse nel repo, quando vengono generati, diventano profili selezionabili nella dashboard;
- `MiniLM`: non incluso come pesi, richiede `requisiti-transformer.txt` e viene scaricato al primo run del confronto transformer.

La dashboard Streamlit seleziona `baseline_pure` come profilo predefinito; `baseline_hardened`, i profili `advanced_aps_*` e i profili `active_learning_*` restano disponibili dal menu laterale quando i rispettivi artefatti sono presenti.

## Struttura essenziale
```text
dir_cartella_principale/
  src/
    01_generate_dataset.py
    02_train_evaluate.py
    03_predict_batch.py
    04_app_streamlit.py
    05_active_learning_cycle.py
    06_compare_transformer.py
    07_train_advanced.py
    08_safety_delta_report.py
    advanced_light_models.py
    utils_ops.py
    utils_text.py
  data/
    reviews_synth.csv
    benchmarks/
    demo/
    active_learning/
  models/
    department_model.joblib
    sentiment_model.joblib
  outputs/
    thresholds.json
    predictions_demo_refresh.csv
    sla_batch_summary_demo.json
  requirements.txt
  requisiti-transformer.txt
```

## Lettura guidata degli output
Se il repository viene aperto per la prima volta, i file da leggere nell'ordine giusto sono questi:
1. [outputs/thresholds.json](outputs/thresholds.json)
   Contiene le soglie runtime usate dalla base ed è il file di output davvero necessario per far funzionare il progetto con i modelli già addestrati.
   Viene letto da [03_predict_batch.py](src/03_predict_batch.py) e [04_app_streamlit.py](src/04_app_streamlit.py) e completa il funzionamento dei modelli base già addestrati.

2. [outputs/predictions_demo_refresh.csv](outputs/predictions_demo_refresh.csv)
   E' un esempio completo di batch prediction: mostra come il sistema arricchisce ogni recensione con `department_pred`, `sentiment_pred`, `risk_score`, `priority`, `impacted_departments` e segnali diagnostici.
   Il relativo riepilogo SLA è [outputs/sla_batch_summary_demo.json](outputs/sla_batch_summary_demo.json), mentre ogni nuovo run salva anche `outputs/sla_batch_summary_<timestamp>.json`.
   Riassume il lotto precedente in chiave operativa: priorità aggregate, distribuzione dei casi e simulazione SLA.

Questi file sono inclusi per aiutare a capire subito il formato degli output senza dover rilanciare comandi, possono essere utili per leggere un esempio completo di batch run.

### Output rigenerabili della base
- `outputs/baseline_pure/metrics.json`
- `outputs/baseline_hardened/metrics.json`
- `outputs/metrics.json`
- `outputs/confusion_department.png`
- `outputs/confusion_sentiment.png`
- `outputs/test_predictions_<timestamp>.csv`
- `outputs/sla_test_summary_<timestamp>.json`

Si ottengono rilanciando [02_train_evaluate.py](src/02_train_evaluate.py)

`outputs/metrics.json` resta un alias compatibile della baseline hardened.

### Output rigenerabili di `advanced_aps`
- `outputs/advanced_aps_pure/metrics_advanced.json`
- `outputs/advanced_aps_hardened/metrics_advanced.json`
- `outputs/advanced_aps/metrics_advanced.json`
- `outputs/advanced_aps/thresholds_advanced.json`
- `outputs/advanced_aps/confusion_department_advanced.png`
- `outputs/advanced_aps/confusion_sentiment_advanced.png`
- `outputs/advanced_aps/test_predictions_advanced_<timestamp>.csv`

Si ottengono rilanciando [07_train_advanced.py](src/07_train_advanced.py) e servono per attivare nella dashboard i profili `advanced_aps_pure` e `advanced_aps_hardened`. `outputs/advanced_aps/` resta un alias compatibile del profilo hardened.

### Output rigenerabili di active learning
- `outputs/active_learning_queue_<timestamp>.csv`
- `outputs/active_learning_oracle/*`
- `outputs/active_learning_v2_no_replay/*`
- `outputs/active_learning_v2_replay/*`
Si ottengono rilanciando [05_active_learning_cycle.py](src/05_active_learning_cycle.py) e [02_train_evaluate.py](src/02_train_evaluate.py), sono artefatti di esperimenti offline e servono a studiare come cambiano dataset e metriche quando si aggiungono esempi selezionati, non vengono letti dal runtime base. 

### Output che non conviene trattare come artefatti stabili del repo prodotto
- `outputs/predictions_*.csv`
- `outputs/sla_batch_summary_*.json`
- `outputs/safety_exp/*` (generato solo dal confronto `Safety before/after`, non richiesto per l'inferenza)
- `outputs/transformer_comparison_*.json`
- `outputs/advanced_project_summary.json`
- `outputs/base_calibration.json`

Sono output di run temporanei, risultati sperimentali e documentazione.

### File importanti in `data/active_learning`
- [data/active_learning/active_learning_queue_oracle_labeled.csv](data/active_learning/active_learning_queue_oracle_labeled.csv)
- [data/active_learning/active_learning_v2_queue_labeled.csv](data/active_learning/active_learning_v2_queue_labeled.csv)

Sono input riproducibili per ricreare i run finali di active learning, senza questi file si può simulare un nuovo ciclo, ma non ricostruire gli stessi esperimenti archiviati.

## Cosa fa la dashboard
La dashboard in [04_app_streamlit.py](src/04_app_streamlit.py) per ogni recensione:
- combina `title` e `body`, dando piu peso al `body`;
- applica il preprocessing testuale:
- stima `department` e `sentiment`;
- applica il runtime layer operativo;
- calcola `risk_score`, `priority`, `impacted_departments` e segnali di review diagnostica.

La tab `Strumenti e rigenerazione` permette di lanciare da interfaccia:
- rigenerazione dataset e benchmark;
- retraining base;
- addestramento `advanced_aps`;
- simulazione di un nuovo ciclo di `active learning`;
- confronto `Safety before/after` per misurare e visualizzare il delta `base` vs `hardened_r2`;
- riproduzione dei run `oracle`, `v2_no_replay`, `v2_replay`, con metriche sintetiche e caricamento come profili se i modelli sono presenti;
- confronto con MiniLM.

## Riproducibilità
### Rigenerare dataset e benchmark
```bash
python3 src/01_generate_dataset.py \
  --n 350 \
  --seed 42 \
  --percorso_output data/reviews_synth.csv \
  --cartella_benchmark data/benchmarks \
  --n_benchmark_colloquiali 220
```

Note:
- il dataset non parte da un CSV reale preesistente, ma viene costruito da `src/01_generate_dataset.py` combinando template testuali, vocabolari di reparto, frasi positive/negative, casi ambigui, rumore lessicale e template safety dedicati;
- il generatore miscela anche frasi colloquiali italiane, contesti realistici d'uso e lessico vicino alle recensioni spontanee;
- il file principale viene salvato in `data/reviews_synth.csv`;
- i benchmark vengono salvati in `data/benchmarks/` come `reviews_in_distribution.csv`, `reviews_ambiguous.csv`, `reviews_noisy.csv`, `reviews_safety_critical.csv` e `reviews_colloquial.csv`;
- con lo stesso `seed`, gli stessi parametri e lo stesso codice ottieni sempre gli stessi file;
- se cambi `seed`, `n` o il profilo di generazione, ottieni una variante diversa ma coerente con la stessa logica sintetica.

### Rigenerare la base pure
In questa configurazione `reviews_safety_critical.csv` resta solo benchmark held-out:
```bash
python3 src/02_train_evaluate.py \
  --dati data/reviews_synth.csv \
  --cartella_benchmark data/benchmarks \
  --revisione_diagnostica \
  --cartella_modelli models/baseline_pure \
  --cartella_output outputs/baseline_pure \
  --seed 42
```

### Rigenerare la base hardened
In questa configurazione `reviews_safety_critical.csv` entra nel training come augmentation controllata (`repeat=2`):
```bash
python3 src/02_train_evaluate.py \
  --dati data/reviews_synth.csv \
  --cartella_benchmark data/benchmarks \
  --percorso_augment_sicurezza data/benchmarks/reviews_safety_critical.csv \
  --ripetizioni_augment_sicurezza 2 \
  --revisione_diagnostica \
  --cartella_modelli models/baseline_hardened \
  --cartella_output outputs/baseline_hardened \
  --seed 42
```

Artefatti principali generati (oltre a metriche, confusion matrix e predizioni di test):
- `models/department_model.joblib`;
- `models/sentiment_model.joblib`;
- `outputs/thresholds.json`.

### Confronto safety before/after
Serve a misurare se l'hardening con `reviews_safety_critical.csv` migliora davvero la robustezza rispetto alla baseline pulita. 
La dashboard lo mostra anche come tabella e grafico nella tab `Strumenti e rigenerazione`.

```bash
python3 src/02_train_evaluate.py \
  --dati data/reviews_synth.csv \
  --cartella_benchmark data/benchmarks \
  --revisione_diagnostica \
  --cartella_modelli models/safety_exp/base \
  --cartella_output outputs/safety_exp/base \
  --seed 42

python3 src/02_train_evaluate.py \
  --dati data/reviews_synth.csv \
  --cartella_benchmark data/benchmarks \
  --percorso_augment_sicurezza data/benchmarks/reviews_safety_critical.csv \
  --ripetizioni_augment_sicurezza 2 \
  --revisione_diagnostica \
  --cartella_modelli models/safety_exp/hardened_r2 \
  --cartella_output outputs/safety_exp/hardened_r2 \
  --seed 42

python3 src/08_safety_delta_report.py \
  --before outputs/safety_exp/base/metrics.json \
  --after outputs/safety_exp/hardened_r2/metrics.json \
  --safety_benchmark reviews_safety_critical.csv \
  --out_json outputs/safety_exp/safety_delta_report_r2.json \
  --out_md outputs/safety_exp/safety_delta_report_r2.md
```

Output principali:
- `outputs/safety_exp/safety_delta_report_r2.json`;
- `outputs/safety_exp/safety_delta_report_r2.md`.

### Rigenerare `advanced_aps_pure`
```bash
python3 src/07_train_advanced.py \
  --dati data/reviews_synth.csv \
  --cartella_benchmark data/benchmarks \
  --revisione_diagnostica \
  --metodo_conformal aps \
  --alpha_conformal 0.10 \
  --cartella_modelli models/advanced_aps_pure \
  --cartella_output outputs/advanced_aps_pure
```

### Rigenerare `advanced_aps_hardened`
```bash
python3 src/07_train_advanced.py \
  --dati data/reviews_synth.csv \
  --cartella_benchmark data/benchmarks \
  --percorso_augment_sicurezza data/benchmarks/reviews_safety_critical.csv \
  --ripetizioni_augment_sicurezza 2 \
  --revisione_diagnostica \
  --metodo_conformal aps \
  --alpha_conformal 0.10 \
  --cartella_modelli models/advanced_aps_hardened \
  --cartella_output outputs/advanced_aps_hardened
```

### Simulare un nuovo ciclo di active learning
```bash
python3 src/05_active_learning_cycle.py \
  --dataset data/reviews_synth.csv \
  --pool data/benchmarks/reviews_noisy.csv \
  --numero_top 40 \
  --solo_revisione_diagnostica \
  --strategia hybrid_v2 \
  --peso_incertezza 0.50 \
  --peso_diversita 0.30 \
  --peso_operativo 0.20
```

Output atteso:
- `outputs/active_learning_queue_<timestamp>.csv`

### Riprodurre i run archiviati di active learning
Le code etichettate per i run finali sono già incluse in [data/active_learning](data/active_learning).
`Oracle` non indica un modello speciale: è una simulazione di active learning in cui la coda di esempi difficili è già etichettata correttamente. Serve come caso ideale per capire quanto potrebbe aiutare un ciclo di revisione umano senza errori. La cartella `models/active_learning_oracle/` non viene versionata come modello finale, ma viene creata rilanciando i comandi sotto.

`Oracle`
```bash
python3 src/05_active_learning_cycle.py \
  --dataset data/reviews_synth.csv \
  --pool data/benchmarks/reviews_noisy.csv \
  --coda_etichettata data/active_learning/active_learning_queue_oracle_labeled.csv \
  --percorso_dataset_output data/reviews_synth_active_oracle.csv \
  --percorso_coda_output outputs/active_learning_queue_oracle_rebuilt.csv

python3 src/02_train_evaluate.py \
  --dati data/reviews_synth_active_oracle.csv \
  --cartella_benchmark data/benchmarks \
  --percorso_augment_sicurezza data/benchmarks/reviews_safety_critical.csv \
  --ripetizioni_augment_sicurezza 2 \
  --revisione_diagnostica \
  --cartella_modelli models/active_learning_oracle \
  --cartella_output outputs/active_learning_oracle
```

`V2 no replay`
```bash
python3 src/05_active_learning_cycle.py \
  --dataset data/reviews_synth.csv \
  --pool data/benchmarks/reviews_noisy.csv \
  --coda_etichettata data/active_learning/active_learning_v2_queue_labeled.csv \
  --percorso_dataset_output data/reviews_synth_active_v2.csv \
  --percorso_coda_output outputs/active_learning_v2_queue_rebuilt.csv

python3 src/02_train_evaluate.py \
  --dati data/reviews_synth_active_v2.csv \
  --cartella_benchmark data/benchmarks \
  --percorso_augment_sicurezza data/benchmarks/reviews_safety_critical.csv \
  --ripetizioni_augment_sicurezza 2 \
  --revisione_diagnostica \
  --cartella_modelli models/active_learning_v2_no_replay \
  --cartella_output outputs/active_learning_v2_no_replay
```

`V2 replay`
```bash
python3 src/05_active_learning_cycle.py \
  --dataset data/reviews_synth.csv \
  --pool data/benchmarks/reviews_noisy.csv \
  --coda_etichettata data/active_learning/active_learning_v2_queue_labeled.csv \
  --percorso_dataset_output data/reviews_synth_active_v2_replay.csv \
  --percorso_coda_output outputs/active_learning_v2_queue_rebuilt_replay.csv \
  --dimensione_replay 120 \
  --seed_replay 42

python3 src/02_train_evaluate.py \
  --dati data/reviews_synth_active_v2_replay.csv \
  --cartella_benchmark data/benchmarks \
  --percorso_augment_sicurezza data/benchmarks/reviews_safety_critical.csv \
  --ripetizioni_augment_sicurezza 2 \
  --revisione_diagnostica \
  --cartella_modelli models/active_learning_v2_replay \
  --cartella_output outputs/active_learning_v2_replay
```

Questi run restano esperimenti offline: non sono profili serviti della dashboard.

### Confronto con MiniLM
```bash
pip install -r requisiti-transformer.txt

HF_HOME=.hf_cache python3 src/06_compare_transformer.py \
  --dati data/reviews_synth.csv \
  --cartella_benchmark data/benchmarks \
  --modello_reparto_baseline models/baseline_pure/department_model.joblib \
  --modello_sentiment_baseline models/baseline_pure/sentiment_model.joblib \
  --seed 42
```
- il confronto scarica il modello `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` al primo avvio se non è già in cache locale, mentre il risultato viene salvato come `outputs/transformer_comparison_<timestamp>.json`
- con `HF_HOME=.hf_cache` MiniLM viene salvato in una cache locale ignorata da git; senza questa variabile viene usata la cache Hugging Face dell'utente, di solito `~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2`;
- nel refresh con seed `42` il file generato è `outputs/transformer_comparison_20260428_005027.json`;
- sul test split, baseline dep/sent macro-F1 `0.8856 / 0.8998`, MiniLM dep/sent macro-F1 `0.8717 / 0.8853`.

## Metriche principali
- `accuracy`: quota di predizioni corrette sul totale. È utile ma può mascherare sbilanciamenti tra classi.
- `f1_macro`: media semplice della F1 per classe. Nel progetto è la metrica più importante per confrontare modelli, perché pesa allo stesso modo `pos/neg` e i tre reparti.
- `recall_by_class`: copertura della singola classe. Serve per capire se un modello sta ignorando, per esempio, i `neg` o un reparto specifico.
- `ECE` (`Expected Calibration Error`): misura quanto la confidenza del modello è coerente con la sua accuratezza reale. Piu basso è meglio.
- `Brier score`: errore quadratico sulle probabilità predette. Anche qui valori piu bassi indicano probabilità piu affidabili.
- `coverage`: quota di casi che il sistema lascia in full-auto senza attivare `needs_review_diag`.
- `needs_review_rate`: complemento del coverage; quantifica quante recensioni vengono mandate a revisione diagnostica.
- `priority_distribution`: distribuzione delle priorità operative `LOW/MEDIUM/HIGH/URGENT` dopo il runtime layer.
- `sla_summary`: simulazione del carico per reparto a partire dalle predizioni finali, utile per leggere l'impatto operativo.
- `conformal coverage` e `avg_set_size`: metriche dei profili `advanced_aps_*`; misurano rispettivamente quante volte il reparto vero cade nel set conforme e quanto il set è largo in media.

## Note pratiche
- `pip install -r requirements.txt` basta per dashboard, base e `advanced_aps_*`;
- se i modelli base sono presenti, la dashboard è subito utilizzabile;
- se `advanced_aps_pure` o `advanced_aps_hardened` non esistono, la dashboard resta operativa e mostra i profili disponibili;
- `requisiti-transformer.txt` è opzionale e forza `torch==2.2.2+cpu`, quindi evita dipendenze CUDA inutili;
- nella UI, se mancano le dipendenze opzionali del confronto MiniLM, compare un alert con installazione guidata tramite pulsante;
- se MiniLM non è in cache, il primo confronto transformer richiede connessione internet;
- il repository include solo gli artefatti strettamente utili o gli esempi più leggibili; il resto si rigenera.
