> Historical lab note. Status/ETA below may be outdated; see [current results](../RESULTS.md).

# Kolejne strategie active learning — ukończony pilot

Seria ukończona 17.10.2026 bez błędów; wyniki w
`outputs/pilot_active_prepared/results.json`. Kolejny uruchomiony etap opisuje
`PILOT_VALIDATION_STUDY.md`.

Przygotowano 17.10.2026 bez uruchamiania. Następnie, na polecenie użytkownika,
**uruchomiono pięć strategii 17.10.2026 o 13:48 czasu polskiego**, lokalnie na
CPU, po ukończeniu AE/VAE i normalizacji. Worker PID 28696. Plan:
`configs/cifar10_pilot_active_prepared.json`, `manual_start_only: true`;
uruchomiony jawnie z `--start-prepared`. Stan i postęp są w
`outputs/pilot_active_prepared/queue.json`, logi w `queue_20261017.stdout.log`
i `queue_20261017.stderr.log` w tym samym katalogu. Przy restarcie najpierw
sprawdzić stan procesu; nie uruchamiać drugiego workera równolegle.

## Pytanie badawcze i dotychczasowy punkt odniesienia

Jak wybierać prawdziwe etykiety uzupełniające pseudoetykietowanie, zamiast
wybierać wyłącznie obrazy o największej entropii? Hipoteza do sprawdzenia:
etykieta poprawiająca błędną, również pewną, pseudoetykietę w gęstym regionie
reprezentacji może być użyteczniejsza od etykiety izolowanego trudnego obrazu.
Nowość względem literatury nie została potwierdzona.

Ukończone kontrole, seed 0, 20 epok, ostatni checkpoint:

| Strategia | Walidacja | Test | Końcowe liczby etykiet na klasę, min–max |
| --- | ---: | ---: | ---: |
| Losowa | 55,30% | 56,08% | 45–56 |
| Entropia | 54,70% | 54,91% | 36–67 |
| Entropia + odległości w embeddingu | 52,80% | 54,32% | 43–58 |

Źródła: `outputs/pilot_next/cnn_N30to500_seed0_active_*/metrics.json`.
Trzecia strategia to top 400 według entropii z 4096 kandydatów, następnie
farthest-first cosine względem L i już wybranych punktów. To nie BADGE.
Jej słabszego wyniku nie można wyjaśnić samą nierównowagą klas: jest bardziej
zbalansowana niż entropia. Inne hipotezy: niereprezentatywne izolowane punkty,
słaba geometria embeddingu, zły pomiar niepewności, zbyt krótki czas treningu
na późno pozyskanych etykietach. Jeden seed nie rozstrzyga mechanizmu.

## Pięć gotowych wariantów

Wzory używają predykcji klasyfikatora `p(x)`, surowego embeddingu `z(x)` oraz
jego wersji unormowanej `u(x)`. `H` to entropia, `JS` to dywergencja
Jensena–Shannona z logarytmem naturalnym. Wszystkie parametry są jawne w JSON.

1. **`badge`** — BADGE jako mocniejszy punkt odniesienia. Dla każdej próbki
   przyjmujemy roboczą klasę `argmax p` i gradient wag ostatniej warstwy:
   `g(x) = (p(x) - onehot(argmax p(x))) ⊗ z(x)`.
   Wybór przez k-means++: największa norma gradientu jako pierwszy punkt,
   potem losowanie proporcjonalnie do kwadratu odległości od najbliższego
   wybranego gradientu. Używamy surowego z, bez normalizacji; gradient biasu
   nie jest dołączany. Implementacja faktoryzowana nie tworzy macierzy N×C×D.
   Nie jest to reprodukcja pełnego protokołu eksperymentalnego autorów.
2. **`entropy_density`** — wynik `H(p(x)) * rho(x)`.
   `rho(x) = mean_k exp((cos(u(x), u(x_j)) - 1) / tau_d)` dla 10 najbliższych
   innych kandydatów z bieżącej puli 4096, `tau_d=0.1`. Sam punkt jest
   wykluczony z sąsiedztwa. Ograniczenie: gęstość zależy od jakości embeddingu
   i losowej puli, a sam ranking nie gwarantuje różnorodności całej partii.
3. **`neighbor_disagreement_density`** — wynik `JS(p(x), q_L(x)) * rho(x)`.
   `q_L` jest rozkładem klas 15 najbliższych **wyłącznie prawdziwie
   oznaczonych** sąsiadów; głosy mają wagi `softmax(cos / 0.2)`.
   Może wskazać pewną predykcję sprzeczną z sąsiedztwem. Ograniczenie:
   sąsiedztwo również może się mylić, szczególnie na początku treningu.
4. **`view_disagreement`** — ranking według
   `H(mean_v p_v(x)) - mean_v H(p_v(x))` dla czterech widoków: jeden bez
   augmentacji i trzy z obecnym słabym crop/flip. To niespójność augmentacji,
   nie BALD i nie niepewność zespołu modeli. Brak dodatkowego składnika
   gęstości pozwala oddzielić efekt samej niestabilności.
5. **`neighbor_disagreement_density_mix`** — 20 przykładów z rankingu z p. 3
   i 20 losowych spośród pozostałych kandydatów. Mieszanka eksploracji
   i selekcji, bez duplikatów. `random_fraction=0.5`; przy nieparzystym
   rozmiarze partii liczba losowych zapytań jest zaokrąglana w dół.

Parametry są ustawieniami pilota, nie wynikami strojenia. Nie wybieramy ich
na podstawie testu ani nie gwarantujemy, że któryś wariant wygra.

## Wspólny protokół i koszty

- Ta sama mała CNN z BatchNorm, inicjalizacja seed=0, SGD i cosine scheduler,
  CE + twarde pseudoetykiety z progiem 0.95, 20 epok × 383 aktualizacje.
- Start 30 prawdziwych etykiet na klasę (300), +40 po epokach 3,6,9,12,15,
  koniec 500 łącznie. Nie wymuszamy 50 etykiet na klasę i nie używamy
  ukrytych klas do równoważenia. Ostatnie pięć epok z pełnym budżetem.
- Te same 1000 wyłączonych obrazów walidacji; etykiety walidacji są osobnym
  zasobem i nie trafiają do selekcji. Oficjalny test dopiero po ostatniej epoce.
- Kandydaci: losowe 4096 obrazów z bieżącego U. Pierwsza pula jest wspólna,
  następne mogą się różnić, bo strategie usuwają inne obrazy z U. Generatory
  losowania puli i selekcji są niezależne od treningu i zapisywane w checkpoint.
- W każdym z pięciu nowych wariantów te same dodatkowe pomiary: jeden widok
  kandydatów i L bez augmentacji oraz trzy słabe widoki samych kandydatów.
  Obliczenia w `eval()` bez gradientów i bez aktualizacji BatchNorm.
  Augmentacje i DataLoader selekcji mają niezależne RNG. Nie dodajemy kroków
  optymalizacji ani treningowych widoków L/U.
- Przy 5 rundach koszt forwardów selekcji to 83 820 wejść zamiast 22 380
  w starych kontrolach; różnica wynosi 61 440. Jest raportowana osobno od
  wejść treningowych. Obliczenia rankingów też kosztują: zapisujemy czas
  selekcji w `history.csv`, `acquisition_seconds`.
- Stare trzy kontrole pozostają odniesieniem. Test regresyjny sprawdza,
  że dodatkowe pomiary przy losowym doborze nie zmieniają zapytań ani
  końcowych wag, włącznie ze stanem BatchNorm. Nie powielamy kontroli tylko
  z powodu dodatkowych pomiarów, ale nie deklarujemy identycznego kosztu
  obliczeniowego. L/U exposure jest zgodne pomiędzy strategiami AL przy
  wspólnym harmonogramie; stałe 500 etykiet od początku to osobne odniesienie.

Dotychczasowe AL trwały 77–80 min dla dwóch pierwszych strategii i 134 min
dla ostatniej przy wolniejszej pracy laptopa. Dla pięciu nowych runów można
wstępnie rezerwować **7–13 godzin CPU plus audyty**, z dużą niepewnością.
To nie zmierzone ETA tej serii; przed startem uwzględnić aktualne obciążenie.

## Artefakty i analiza przyczyn

Wyniki: `outputs/pilot_active_prepared/`, pełne checkpointy
i wznowienie na granicach epok, `history.csv`, `acquisitions.json`, końcowe
`metrics.json`, standardowy audyt reprezentacji po każdym modelu.
Wspólna tabela `outputs/pilot_comparison.md/.csv/.json` uwzględni nowy katalog
dopiero po pojawieniu się prawdziwych runów. Konfiguracje nie są wynikami.

Każda runda zapisuje indeksy i hash kandydatów, ujawnione po wyborze etykiety,
wyniki wybranych punktów: entropię, pewność, klasę przewidywaną, dostępne
składniki gęstości/JS/normy gradientu i pochodzenie ranked/random.
Etykiety zapytań wolno wykorzystać dopiero po wyborze do analizy błędów,
nie do cofania ani zmiany selekcji. Etykiet reszty U selektor nie otrzymuje.

Po uzyskaniu wyników zbadać krzywe walidacji, klasy zapytań, udział błędnych
predykcji w kupionych etykietach, redundancję/gęstość wybranych obrazów oraz
rzeczywisty koszt. W kolejnych eksperymentach warto dopisać pomiary jakości
pseudoetykiet i reprezentacji w każdej rundzie; obecnie standardowy audyt
reprezentacji jest końcowy. Oryginalne rundy nie mają zachowanych wszystkich
ówczesnych predykcji, więc nie odtwarzamy ich z końcowego modelu.

## Bezpieczny podgląd i późniejszy start

Podgląd waliduje pięć konfiguracji. Nie trenuje i nie tworzy workera:

```powershell
python scripts/run_next_pilot.py --plan configs/cifar10_pilot_active_prepared.json --print-plan
```

Użyte polecenie startu; służy również do wznowienia po zakończeniu procesu:

```powershell
python scripts/run_next_pilot.py --plan configs/cifar10_pilot_active_prepared.json --start-prepared --device cpu
```

Polecenie startu poczeka na ukończenie obecnej kolejki normalizacji
(`outputs/pilot_normalization/results.json`). Bez `--start-prepared` plan
odmawia uruchomienia przed utworzeniem katalogu wyników. Nie jest to usługa
działająca w tle. Aby wznowić po restarcie, użyć tego samego jawnego polecenia.
GPU można wybrać przez `--device cuda`, dopiero w środowisku z PyTorch CUDA.

Nie zmieniamy teraz harmonogramu, bazowej straty ani liczby epok. Osobne,
przyszłe badania: dodatkowy czas po ostatniej akwizycji (dla wszystkich
strategii wraz z random), aktywna selekcja przy miękkich pseudoetykietach,
reprezentacja AE/VAE, wiele seedów i budżetów. Te pomysły nie należą do
obecnego pięciorunowego planu i nie mają uruchomionych workerów.

Źródła:
[BADGE](https://arxiv.org/abs/1906.03671),
[kod referencyjny BADGE](https://github.com/JordanAsh/badge/blob/master/query_strategies/badge_sampling.py),
[Consistency-based Semi-supervised Active Learning](https://arxiv.org/abs/1910.07153).
Różnorodność i niepewność to znane kierunki; heurystyki z p. 2–5 nie są
przedstawiane jako potwierdzony nowy wkład naukowy.
