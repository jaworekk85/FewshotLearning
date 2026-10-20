> Historical lab note. Status/ETA below may be outdated; see [current results](../RESULTS.md).

# CIFAR-10: pierwszy eksperyment kontrolny

Stan na 15 października 2026: przygotowany kod i testy syntetyczne.
Nie zmierzono jeszcze accuracy tego modelu na CIFAR-10.

## Co uruchamiamy

- ResNet-18 trenowany od losowej inicjalizacji; wejście 32 × 32,
  pierwsza konwolucja 3 × 3 ze stride 1, bez początkowego max-poolingu.
- Reprezentacja 512D i liniowa głowica dziesięcioklasowa.
- Crop + flip, normalizacja CIFAR-10, CE, SGD z momentum,
  learning rate 0.1, weight decay 0.0005, cosine decay, batch 128.
- 200 epok odniesienia: 383 aktualizacje na epokę przy 49 000 obrazów
  treningowych, łącznie 76 600 aktualizacji.
- Z oficjalnych danych treningowych rezerwujemy 100 obrazów na klasę do
  walidacji. Kontrola z pełnymi etykietami uczy się na pozostałych 49 000.
- Domyślnie raportujemy ostatni checkpoint. Test jest oceniany po treningu.
  Wybór najlepszego checkpointu walidacyjnego jest osobną jawną opcją.

Robocze kryterium przejścia dalej: końcowa accuracy na oficjalnym teście >90%,
z krzywymi potwierdzającymi działający trening. To cel kontrolny, nie uzyskany
wynik ani gwarancja. Po zamrożeniu protokołu uruchomić trzy seedy.
Mocny wynik pełnonadzorowany nie gwarantuje wyniku przy małym budżecie.

## Środowisko i sprzęt

Na tym laptopie jest RTX 3050 Ti 4 GB, ale obecny interpreter ma PyTorch
`2.14.1+cpu`. Pełny trening GPU wymaga osobnego środowiska z działającą CUDA.
Nie zmieniano globalnego środowiska i nie wynajmowano żadnej maszyny.

Na serwerze użyć środowiska PyTorch z CUDA zgodnego ze sterownikiem.
Na start wystarczy jedna karta; do dalszego SSL wygodny będzie GPU 16–24 GB,
4–8 vCPU, 16–32 GB RAM i 50–100 GB SSD. Przy 4 GB VRAM może być potrzebny
mniejszy batch; trzeba zapisać tę zmianę i używać jej konsekwentnie.

## Uruchomienie

Podgląd konfiguracji bez treningu ani pobierania:

```powershell
python scripts/run_supervised.py --print-spec
```

Kontrola przepływu na syntetycznych danych, bez CIFAR-10:

```powershell
python scripts/run_supervised.py --smoke --device cpu --run-name smoke_check
```

Pomiar szybkości na docelowej maszynie, bez pobierania danych:

```powershell
python scripts/benchmark_supervised.py --device cuda --batch-size 128 --steps 30 --output outputs/benchmarks/gpu.json
```

Pomiar używa syntetycznych obrazów w pamięci urządzenia. Wyliczona liczba
godzin obejmuje sam trening; nie uwzględnia augmentacji, ładowania danych,
walidacji, checkpointów ani throttlingu podczas wielogodzinnej pracy.
Do wyceny długiej serii użyć następnie czasu rzeczywistej epoki CIFAR-10.

Pełny eksperyment, po przygotowaniu CUDA:

```powershell
python scripts/run_supervised.py --budget all --device cuda --download --run-name full_seed0
```

Pierwsze dwie epoki tego samego zaplanowanego treningu, potem wznowienie:

```powershell
python scripts/run_supervised.py --budget all --device cuda --download --run-name full_seed0 --stop-after-epoch 2
python scripts/run_supervised.py --budget all --device cuda --download --run-name full_seed0 --resume
```

Limit `--stop-after-epoch` pozostawia pełny harmonogram 200 epok i nie
ocenia testu. Pozwala zmierzyć czas epoki i wznowić trening.
Nie używać `--epochs 2` do takiego pomiaru: zmieniłoby to harmonogram LR.

Mały budżet, z niezależną inicjalizacją, bez wczytywania wag pełnonadzorowanych:

```powershell
python scripts/run_supervised.py --budget 50 --seed 0 --device cuda --run-name N50_seed0
```

Każdy budżet ma domyślnie tę samą liczbę aktualizacji. W małej puli loader
powtarza dane w obrębie epoki odniesienia. Ostatnie batche mogą być mniejsze;
rzeczywista liczba przetworzonych obrazów jest zapisywana w historii.
Konfiguracja jest punktem startowym; baseline małego budżetu również wymaga
sprawdzenia krzywych i regularyzacji na wydzielonym zadaniu rozwojowym.

## Etykiety walidacyjne i test

Domyślna konfiguracja rezerwuje dodatkowe 1000 etykiet walidacyjnych.
Trzeba raportować ten zasób obok budżetu treningowego, również gdy służy
do oglądania krzywych i strojenia parametrów. Domyślna reguła `last` nie
wybiera checkpointu na podstawie walidacji.

Wariant bez dodatkowej walidacji:

```powershell
python scripts/run_supervised.py --budget 50 --validation-per-class 0 --selection last --device cuda --run-name N50_no_extra_val_seed0
```

To inny protokół: zmienia dostępną pulę, podział i domyślną liczbę kroków.
Nie mieszać go w jednej sparowanej tabeli z wariantem mającym walidację.
Nie stroić parametrów na oficjalnym teście. Odczyt testu przez wiele
rozwojowych uruchomień i wybór najlepszego wariantu też jest strojeniem.

## Artefakty i weryfikacja

Każde uruchomienie ma katalog `outputs/supervised/<run-name>/`:

- `config.json`: rozwiązana konfiguracja i wersja PyTorch;
- `splits.json`: dokładne indeksy danych treningowych i walidacyjnych;
- `history.csv`: loss, accuracy, learning rate, czas i przetworzone obrazy;
- `latest.pt`: checkpoint do wznowienia z optymalizatorem, schedulerem i RNG;
- `best.pt`: tylko przy jawnym wyborze `best_validation`;
- `model.pt` i `metrics.json`: końcowy wybrany model i wynik testu.

Runner odmawia nadpisania istniejącego uruchomienia. Wznowienie zachowuje
harmonogram i sprawdza zgodność konfiguracji oraz podziału.
Identyczne wznowienie sprawdzono na CPU z `num_workers=0`; inne urządzenia,
wersje bibliotek i równoległe ładowanie nie gwarantują identyczności bitowej.
Syntetyczne kontrole są oznaczone `synthetic_smoke_only` i nie trafiają do
starych tabel `outputs/results.csv`.

```powershell
python -m unittest discover -s tests -v
```

Przepis porównawczy: [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).
Ten projekt raportuje 93.02% dla ResNet-18; nasz wynik trzeba dopiero zmierzyć.
