> Historical lab note. Status/ETA below may be outdated; see [current results](../RESULTS.md).

# Szybki pilot na CIFAR-10

Zaczynamy od istniejącego `ConvEncoder` (`architecture=cnn`): trzy konwolucje
32/64/128 kanałów, pooling, embedding 128D, głowica dziesięcioklasowa.
Łącznie **111 498 parametrów**. Nie zmieniano architektur odtworzonych z PDF-u.

Cel: tanio sprawdzić działanie treningu i wykrywać efekty dodawania członów
straty. Pierwszy wynik supervised jest już zmierzony; stabilność rankingu
wariantów SSL wymaga kolejnych eksperymentów.
Najciekawsze efekty trzeba później potwierdzić na większym encoderze.

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
Nie zmierzono jeszcze czasu GPU.

Benchmarki zapisano w `outputs/benchmarks/`; wykonano je kolejno, bez
równoległego obciążania CPU innym benchmarkiem.

## Pierwsza kontrola

```powershell
python scripts/run_supervised.py --config configs/cifar10_pilot_supervised.json --device cpu --download --run-name cnn_pilot_full_seed0 --stop-after-epoch 2
```

Po dwóch rzeczywistych epokach sprawdzić czas, train loss i walidację.
Następnie wznowić zaplanowany trening, zachowując harmonogram:

```powershell
python scripts/run_supervised.py --config configs/cifar10_pilot_supervised.json --device cpu --run-name cnn_pilot_full_seed0 --resume
```

Wszystkie ograniczenia dotyczące walidacji i testu opisuje
`SUPERVISED_BASELINE.md`. Pilot rezerwuje dodatkowe 100 etykiet walidacyjnych
na klasę. Domyślnie wybiera ostatni checkpoint.

## Najmniejsza seria porównawcza

1. Najpierw ukończyć supervised control na dostępnych pełnych etykietach.
   Sprawdzić krzywą i czy poprawa trwa jeszcze pod koniec harmonogramu.
2. Na tym samym CNN porównać N=10 i N=50, początkowo seed=0.
3. Dla każdego budżetu porównać CE, CE+triplet_L,
   CE+triplet_U i CE+triplet_L+triplet_U: **8 runów**.
4. Przy ciekawym wyniku powtórzyć z trzema sparowanymi seedami: 24 runy.

Nowy wspólny protokół dwóch widoków L obsługuje CE, triplet_L, triplet_U,
ich połączenie i pseudoetykiety: `PILOT_SSL.md`, konfiguracja
`configs/cifar10_pilot_ssl.json`. Stare CE z jednym widokiem zachowujemy jako
osobny punkt odniesienia. Nowych wyników SSL jeszcze nie zmierzono.

Podgląd konfiguracji i baseline małego budżetu:

```powershell
python scripts/run_supervised.py --config configs/cifar10_pilot_supervised.json --budget 50 --print-spec
python scripts/run_supervised.py --config configs/cifar10_pilot_supervised.json --budget 50 --device cpu --run-name cnn_pilot_N50_seed0
```

Małe budżety zachowują tę samą liczbę aktualizacji co pełna kontrola,
powtarzając dozwoloną oznaczoną pulę. Wynik pełnej kontroli nie służy jako
pretrening do małych budżetów.

Nie odrzucać metody wyłącznie po krótkim pilotażu: słaba reprezentacja może
wynikać z małej pojemności lub niedokończonego treningu. Porównać też krzywe,
a w kolejnym etapie k-NN i linear probe, nie tylko wizualizację 2D.
