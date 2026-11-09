# Archiwum protokołów

Dokumenty opisują wykonane eksperymenty i ich ograniczenia.
Bieżące wyniki 74 ukończonych treningów: [RESULTS.md](../RESULTS.md).
Konfiguracje w `configs/` zachowano do odtwarzania i interpretacji wyników.

## Wykonane serie

| Dokument | Zakres |
|---|---|
| [PILOT.md](PILOT.md) | Trzy początkowe kontrole nadzorowane CIFAR-10 |
| [PILOT_SSL.md](PILOT_SSL.md) | CE, twarde pseudoetykiety i podstawowe trójki |
| [PILOT_NEXT.md](PILOT_NEXT.md) | Miękkie pseudoetykiety, semi-hard, pierwsze strategie AL i audyty |
| [PILOT_RECONSTRUCTION.md](PILOT_RECONSTRUCTION.md) | AE/VAE oraz kontrola forwardów U |
| [PILOT_NORMALIZATION.md](PILOT_NORMALIZATION.md) | BatchNorm i brak normalizacji |
| [PILOT_ACTIVE_PREPARED.md](PILOT_ACTIVE_PREPARED.md) | Pięć dodatkowych strategii AL |
| [PILOT_VALIDATION_STUDY.md](PILOT_VALIDATION_STUDY.md) | Trzy seedy, harmonogramy 20/40/80 epok i miks trójek z AL |
| [PILOT_ACTIVE_SUBSET_CONTROL.md](PILOT_ACTIVE_SUBSET_CONTROL.md) | Końcowe zbiory random/BADGE dostępne od początku |
| [PILOT_RELIABILITY.md](PILOT_RELIABILITY.md) | Diagnostyka pewności pseudoetykiet na walidacji |
| [PILOT_NEIGHBOR_RELIABILITY.md](PILOT_NEIGHBOR_RELIABILITY.md) | Wagi sąsiedzkie i selekcja niezgodnych predykcji |
| [PILOT_MINED_TRIPLETS.md](PILOT_MINED_TRIPLETS.md) | Reguły doboru trójek w embeddingu i pikselach |
| [PILOT_TEACHER_CONTRASTIVE.md](PILOT_TEACHER_CONTRASTIVE.md) | Temperatura celów, EMA i contrastive |

Fashion-MNIST opisuje [osobny protokół](../FASHION_MNIST_SSL.md).
[Zasady interpretacji](SCIENTIFIC_ASSESSMENT.md) zbierają ograniczenia porównań.

## Starsza rekonstrukcja

[EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md) dokumentuje pochodzenie starszej ścieżki
kodu i `v4_protocol.json`. Siatka 5760 konfiguracji nie została wykonana
i nie jest bieżącym planem badań.

Usunięto dawną listę przyszłych etapów (m.in. ResNet-18, reprodukcja FixMatch,
osobny pretrening VAE, FSL i propozycje publikacji), kosztorys chmury oraz
instrukcję planowanego pierwszego baseline'u ResNet-18. Nie stanowią zadań
do wykonania. Usunięto też stare ETA, PID-y i zapowiedzi kolejnych kolejek.
Historia tych dokumentów pozostaje w Git.

Wyniki AL, AE/VAE, EMA i kolejnych reguł doboru relacji pozostają częścią
dokumentacji, również gdy nie wykazały przekonującej przewagi nad kontrolami.
Pilot jednego seedu nie rozstrzyga o przydatności całej rodziny metod.
Archiwum nie wyznacza dalszych eksperymentów.
