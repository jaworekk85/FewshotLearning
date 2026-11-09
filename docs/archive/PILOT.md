> Dokumentacja ukończonego eksperymentu; wyniki: [RESULTS.md](../RESULTS.md).
> Opis służy interpretacji i odtworzeniu wyników, nie jest listą zadań.

# Szybki pilot na CIFAR-10

Zaczynamy od istniejącego `ConvEncoder` (`architecture=cnn`): trzy konwolucje
32/64/128 kanałów, pooling, embedding 128D, głowica dziesięcioklasowa.
Łącznie **111 498 parametrów**. Nie zmieniano architektur odtworzonych z PDF-u.

Cel pilota: sprawdzenie działania treningu i efektów dodawania członów straty.

## Zakończona kontrola supervised

Run `outputs/supervised/cnn_pilot_full_seed0/`, seed 0:

- trening na 49 000 oznaczonych obrazów, osobno 1000 walidacyjnych;
- 20 epok, 7660 aktualizacji, wybór ostatniego checkpointu;
- **accuracy na oficjalnym teście (10 000 obrazów): 79.75%**;
- końcowa walidacja: 79.4%, trening z augmentacjami: 81.57%;
- **czas CPU: 1036.4 s, czyli 17 min 16 s**, około 49–55 s/epokę.

To pojedynczy pełnonadzorowany run; nie jest wynikiem małego budżetu ani SSL.
Nie jest też gwarantowaną górną granicą innych metod.
Metryki, podziały, historia i checkpointy są w katalogu runu.
Nie uruchamiać ponownie pod tą samą nazwą: runner chroni ukończone wyniki.

## Zakończone baseline'y małych budżetów

Ten sam CNN, seed=0, losowa inicjalizacja, 7660 aktualizacji i wybór ostatniego
checkpointu. Podziały są sparowane; 1000 etykiet walidacyjnych raportujemy osobno.

| Budżet na klasę | Etykiety treningowe | Test accuracy | Czas CPU |
| --- | --- | --- | --- |
| 10 | 100 | **29.77%** | 12 min 50 s |
| 50 | 500 | **49.62%** | 17 min 4 s |
| pełna pula po walidacji | 49 000 | **79.75%** | 17 min 16 s |

Małe budżety osiągają około 100% accuracy treningowej z augmentacjami,
przy znacznie niższej accuracy walidacyjnej/testowej — wyraźny sygnał
przeuczenia. Nie dowodzi to, że dowolna strata SSL poprawi wynik.
To pojedynczy seed, bez niepewności między losowaniami.

Artefakty: `outputs/supervised/cnn_pilot_N10_seed0/`,
`outputs/supervised/cnn_pilot_N50_seed0/` oraz
`outputs/supervised/low_label_baselines_seed0.json`.
Przetworzone obrazy: odpowiednio 766 000 i 957 680, wobec 980 000 w pełnej
kontroli. Liczba aktualizacji jest wspólna; różne końcowe batche powodują,
że liczba widoków i koszt obliczeniowy nie są dokładnie równe.

## Szybkość

Pomiar lokalny CPU, sześć wątków, float32, batch 128, dane syntetyczne w RAM:

- CNN: około **0.104 s/aktualizację**, około 1235 obrazów/s;
- wcześniejszy ResNet-18: około 2.397 s/aktualizację;
- CNN jest w tych pomiarach około **23 razy szybszy na aktualizację**.

Konfiguracja pilota ma **20 epok odniesienia / 7660 aktualizacji**, zamiast
200 epok większej kontroli. To około **13 minut samego treningu CNN na CPU**.
Dochodzi ładowanie, augmentacje, walidacja, zapis checkpointów i końcowy test.
Rzeczywisty ukończony run z tym narzutem trwał **17 min 16 s**.
To czas pojedynczego supervised runu; SSL z dodatkowymi widokami kosztuje więcej.
Czasu GPU w tej serii nie mierzono.

Benchmarki zapisano w `outputs/benchmarks/`; wykonano je kolejno, bez
równoległego obciążania CPU innym benchmarkiem.

## Protokół

Konfiguracja: `configs/cifar10_pilot_supervised.json`.
Pilot rezerwuje dodatkowe 100 etykiet walidacyjnych na klasę i wybiera ostatni
checkpoint. Małe budżety zachowują liczbę aktualizacji pełnej kontroli,
powtarzając oznaczoną pulę. Wagi pełnej kontroli nie służą jako pretrening.
Późniejszy protokół dwóch widoków L opisuje [PILOT_SSL.md](PILOT_SSL.md).
