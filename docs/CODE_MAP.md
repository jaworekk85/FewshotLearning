# Mapa kodu i dalsza praca

## Bieżąca ścieżka SSL

| Plik | Odpowiedzialność |
|---|---|
| `fewshot_metric/supervised.py` | Podziały danych, pętla treningu, scheduler, checkpointy i test końcowy |
| `fewshot_metric/pilot.py` | Wspólne widoki L/U, CE, pseudoetykiety i podstawowe trójki |
| `fewshot_metric/models.py` | CNN CIFAR-10, mała CNN grayscale, dodatkowe ResNety |
| `fewshot_metric/mined_triplets.py` | Dobór trójek z prawdziwych i pewnych pseudoetykiet |
| `fewshot_metric/mixed_contrastive.py` | Wspólny graf relacji etykietowanych, instancyjnych i pseudo |
| `fewshot_metric/active.py`, `active_selection.py` | Oracle, harmonogram i strategie active learningu |
| `fewshot_metric/representations.py` | Probe, kNN, geometria i wizualizacje |
| `ema_teacher.py`, `neighbor_reliability.py`, `reconstruction.py` | Przebadane rozszerzenia; nie domyślny punkt startowy |

Najważniejsze komendy:

| Skrypt | Kiedy użyć |
|---|---|
| `scripts/run_gray_ssl.py` | Ograniczona seria Fashion-MNIST/MNIST, audyty i wykresy |
| `scripts/run_supervised.py` | Jeden supervised/SSL run z wybraną metodą |
| `scripts/run_pilot_pair.py` | Kilka wariantów CIFAR na tym samym budżecie i seedzie |
| `scripts/analyze_representations.py` | Analiza zapisanych modeli |
| `scripts/run_next_pilot.py` | Historyczne, sekwencyjne plany CIFAR z zależnościami od wcześniejszych wyników |
| `scripts/export_handoff.py` | Sprawdzenie i odświeżenie migawki wyników do GitHuba |
| `scripts/build_research_overview.py` | Odtworzenie PDF z pełnych lokalnych artefaktów CIFAR |

Pozostałe `summarize_*`, benchmarki i analizy obsługują konkretne serie
eksperymentów. Zachowano je, żeby nie utracić możliwości odtworzenia analiz.
Nie trzeba ich czytać, aby uruchomić pięć głównych metod.

## Starsza rekonstrukcja manuskryptu

`config.py`, `train.py`, `losses.py`, `data.py` i `transforms.py` są też używane
przez starszy, konfigurowalny protokół `v4_protocol.json`, `run_experiment.py`
i `run_grid.py`. Ta ścieżka ma inne długości uczenia i polityki danych niż
bieżące piloty. Nie należy łączyć jej wyników z pilotami bez sprawdzenia protokołu.
5760 pozycji manifestu to **niewykonana historyczna siatka**, nie bieżący plan. Historyczny opis:
[EXPERIMENT_SPEC.md](archive/EXPERIMENT_SPEC.md).

## Rozsądny zakres zmian

Nowy pomysł zacznij od osobnego JSON z konfiguracją i osobnego `output_dir`.
Nie nadpisuj dotychczasowych runów. Utrzymaj wspólne seedy, etykiety i prezentacje L;
raportuj osobno dodatkowe wejścia U, nauczyciela, selekcji i koszt dekodera.
Nową logikę straty/selekcji trzymaj w osobnym module, nie kolejnej kopii trenera.

Kod biblioteki ma około 3800 linii; skrypty treningu/analiz/raportu około 3000.
Duża część skryptów to prezentacja wyników, a nie kolejne implementacje uczenia.
Porządkowanie przekazania nie zmienia funkcji strat, augmentacji ani historycznych
konfiguracji: ich zmiana wymaga nowych eksperymentów.

[Archiwum protokołów](archive/README.md) opisuje ukończone eksperymenty i starszą
rekonstrukcję manuskryptu. Nie jest listą przyszłych zadań. Bieżący stan badań
opisują wyniki oraz README; kolejny eksperyment nie został tu ustalony.
