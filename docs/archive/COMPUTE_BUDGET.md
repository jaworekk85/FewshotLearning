> Historical lab note. Status/ETA below may be outdated; see [current results](../RESULTS.md).

# Zasoby i czas: CIFAR-10

Stan i cennik sprawdzono 15 października 2026.

## Zalecenie

Do nauki kodu i krótkich kontroli wystarczy ten laptop. Pierwszy pełny baseline
można uruchomić na jego GPU po przygotowaniu środowiska CUDA.
Do powtarzalnych serii SSL polecana jedna karta 16–24 GB (np. RTX 3090/4090),
4–8 vCPU, 16–32 GB RAM i 50–100 GB SSD. Zacząć od zasobów uczelnianych,
jeśli są dostępne; alternatywa to wynajem jednej maszyny GPU na godziny.

Wynajmu ani instalacji nowego środowiska jeszcze nie wykonano.

## Co zmierzono

Lokalny GPU: **NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB**, sterownik 555.99.
Aktualny interpreter: PyTorch 2.14.1+cpu, bez dostępu do CUDA.

Pomiar CPU na ResNet-18 dla CIFAR-10, float32, batch 128, sześć wątków:

- dwie aktualizacje rozgrzewki i pięć zmierzonych;
- mediana **2.397 s/aktualizację**, około **53.4 obrazów/s**;
- dla 76 600 aktualizacji: **około 51 h samego treningu**;
- bez ładowania/augmentacji danych, walidacji, checkpointów i długotrwałego
  wpływu temperatury. Pełny trening będzie wymagał dodatkowego czasu.

Pomiar: `outputs/benchmarks/cpu_20261015.json`.
Niewielka zmiana inicjalizacji pierwszej konwolucji po pomiarze nie zmienia
architektury ani liczby operacji; nie jest to pomiar accuracy.

## Wstępne widełki do planowania

Dotyczą pojedynczego pełnonadzorowanego treningu ResNet-18, około 200 epok,
49 000 obrazów treningowych, batch 128. Accuracy nie jest jeszcze zmierzona.

| Maszyna | Szacunkowy czas | Podstawa |
| --- | --- | --- |
| Lokalny CPU | co najmniej około 51 h | ekstrapolacja lokalnego pomiaru samego treningu |
| RTX 3050 Ti Laptop 4 GB | orientacyjnie 2–8 h | szacunek inżynierski; brak pomiaru GPU |
| Serwer RTX 4090 24 GB | orientacyjnie 0.5–3 h | szacunek inżynierski; brak pomiaru GPU |

Widełki GPU nie są benchmarkiem, gwarancją ani podstawą rozliczenia dużej serii.
Zależą od batcha, CPU, ładowania danych, wersji bibliotek i limitu mocy GPU.
Przed serią zmierzyć dwie rzeczywiste epoki bez zmiany pełnego harmonogramu.
Jeśli laptop wymaga mniejszego batcha, pomiar i liczbę kroków przeliczyć ponownie.

SSL może wymagać znacznie więcej czasu niż CE. Liczyć faktyczne widoki i
przejścia sieci: triplet przetwarza trzy obrazy, a FixMatch z proporcją U:L=7
przetwarza dodatkowo dwa widoki siedmiokrotnie większego batcha U.
Nie przenosić czasu baseline'u bezpośrednio na wszystkie warianty.
Pretrening VAE i każdą rundę active learning również uwzględnić w budżecie.

## Orientacyjny koszt chmury

[Oficjalny cennik Runpod](https://www.runpod.io/pricing) pokazywał:

- RTX 3090 24 GB: 0.50 USD/h;
- RTX 4090 24 GB: 0.89 USD/h;
- RTX A5000 24 GB: 0.27 USD/h.

Ceny i dostępność zależą od konkretnej oferty. Do kosztu doliczyć dysk,
czas instalacji/bezczynności i ewentualne podatki.
Przy przyjętych widełkach czasu koszt samego GPU dla jednego baseline'u na
RTX 4090 wynosi około 0.45–2.67 USD; nie jest to oferta ani pomiar.

Dla serii najpierw zmierzyć jeden run każdej metody, potem liczyć:

`koszt = suma(czas metody × liczba budżetów × liczba seedów × stawka GPU)`.

Przykładowy pilot 2 budżety × 4 metody × 3 seedy to 24 runy, poza kontrolą
pełnonadzorowaną. Czas SSL trzeba zmierzyć przed wyceną tych 24 runów.

## Jak zawęzić szacunek

```powershell
python scripts/benchmark_supervised.py --device cuda --batch-size 128 --warmup 5 --steps 30 --output outputs/benchmarks/gpu.json
python scripts/run_supervised.py --device cuda --download --run-name timing_seed0 --stop-after-epoch 2
```

Druga komenda zachowuje docelowy harmonogram 200 epok i zapisuje dwie epoki
w `history.csv`, bez odczytu testu. Koszt całego runu oszacować po rozgrzewce
z czasu rzeczywistej epoki, uwzględniając następnie czas końcowego testu.

Na laptopie najpierw utworzyć osobne środowisko z PyTorch CUDA kompatybilnym
ze sterownikiem. Oficjalne pakiety i pary torch/torchvision:
[PyTorch previous versions](https://docs.pytorch.org/get-started/previous-versions/).
Nie zastępować globalnych pakietów roboczych bez potrzeby.
