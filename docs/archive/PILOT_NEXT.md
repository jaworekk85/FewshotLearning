> Historical lab note. Status/ETA below may be outdated; see [current results](../RESULTS.md).

# Kolejna seria: reprezentacje, dobór trójek, miękkie pseudoetykiety, active learning

Po ukończeniu tej serii osobna kolejka uruchomi blok odszumiających AE/VAE
i kontrole kosztu, opisane w `PILOT_RECONSTRUCTION.md`.

Zlecona 16.10.2026. Jeden seed (0), obecny mały CNN, bez pretreningu.
Plan: `configs/cifar10_pilot_next.json`. Kolejka czeka na ukończone
`outputs/pilot/cnn_N50_seed0_ce_2v_40epochs/metrics.json`, następnie wykonuje
analizy i treningi sekwencyjnie. Nie konkuruje z CE40 o CPU.

## Analiza reprezentacji — wspólna dla obecnych i przyszłych modeli

Najpierw osiem zakończonych modeli, w tym CE40. Po każdym z siedmiu nowych
treningów automatycznie ten sam audyt. Dane: wyłącznie zapisany L i te same
1000 obrazów walidacji; bez testu i bez ujawniania etykiet U.

- PCA oraz t-SNE (seed=0, te same ustawienia), PNG i SVG.
- k-NN dla k=1,5,15; kosinus w 128-wymiarowym embeddingu po normalizacji.
- Klasyfikator liniowy na zamrożonej reprezentacji: wyłącznie prawdziwe L,
  StandardScaler dopasowany na L, LogisticRegression C=1, do 1000 iteracji.
- Udział sąsiadów tej samej klasy, silhouette cosine, stosunek rozproszenia
  wewnątrz klas do całkowitego i efektywny rząd reprezentacji.
- Strata i udział aktywnych losowych trójek na walidacji z prawdziwymi klasami.
- Surowe embeddingi, etykiety L/walidacji, indeksy i współrzędne wykresów w NPZ.

Wyniki: `outputs/representations/<nazwa_run>/`, zestawienie `comparison.csv`
i `comparison.json`. Hash wag i podziału chroni przed ponownym użyciem
analizy innego modelu. t-SNE jest ilustracją; klastry na rysunku nie zastępują
pomiarów w pełnej przestrzeni. Osie niezależnych t-SNE nie są wspólnym układem.

Biblioteki wykresów są w lokalnym `.venv-analysis`, współdzielącym istniejący
PyTorch. Globalne pakiety Pythona nie zostały zmienione. Instalacja:

```powershell
python -m venv --system-site-packages .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install matplotlib scikit-learn
```

Dowolny późniejszy model zgodny z formatem runów można zbadać poleceniem:

```powershell
.\.venv-analysis\Scripts\python.exe scripts/analyze_representations.py outputs/pilot/NAZWA_RUN --output-dir outputs/representations
```

## Lepszy dobór trójek

1. `ce_2v_triplet_l_semihard`: te same dwa widoki L i CE co w obecnym
   `ce_2v_triplet_l`. Losowy pozytyw tej samej znanej klasy, negatyw wybierany
   z przykładów innej klasy, których odległość od kotwicy mieści się między
   odległością pozytywu i odległością pozytywu + margines.
2. `ce_2v_pseudo_semihard_triplet`: analogiczny dobór negatywów w obecnych
   trójkach kierowanych pseudoetykietami, z kandydatami z L i pewnej części U.

Gdy nie ma negatywu semi-hard, wybieramy najbliższy dozwolony negatyw.
Rejestrujemy liczbę trójek, aktywnych ograniczeń i trafień semi-hard.
Brak klasowego pozytywu lub negatywu w pseudo-trójce oznacza pominięcie.
Margines 0.2, waga 1; bez dodatkowych przejść encodera. Ten sam losowy seed,
obrazy, augmentacje i 7660 aktualizacji co w poprzednich odpowiednikach.
Semi-hard może również pogarszać wynik, zwłaszcza przy błędnych pseudoetykietach.

## Miękkie pseudoetykietowanie

Dwa warianty odróżniają wagę przykładu od samego celu klasyfikacji:

- `ce_2v_pseudo_soft_weight`: twarda przewidywana klasa, miękka waga pewności.
- `ce_2v_pseudo_soft`: ta sama waga i cel będący rozkładem klas,
  `softmax(weak_logits / 0.5)`, odłączony od gradientu.

Pierwsza epoka zachowuje maskę pewności >=0.95. Potem waga wynosi 1 dla
pewności powyżej średniej z poprzedniej epoki, a poniżej niej maleje jak
`exp(-(confidence-mean)^2/(2*std^2))`; std ma minimum 0.05. Statystyki
aktualizujemy raz na epokę, wyłącznie z predykcji na U. Strata jest średnią
ważonego CE po całym batchu U. Cele odświeżają się na bieżąco w każdym kroku;
nie przechowujemy stałych pseudoetykiet przez całą epokę.

To wariant inspirowany SoftMatch, bez pełnej reprodukcji jego receptury,
EMA statystyk per krok i wyrównywania rozkładu klas. Nie nazywamy go wynikiem
SoftMatch. Zapisujemy średnią wagę i parametry ważenia na kolejną epokę.

## Active learning

Wszystkie trzy warianty korzystają z tego samego bazowego pseudoetykietowania
z twardym progiem 0.95, bez nowych trójek i bez miękkich wag. Dzięki temu
zmiana doboru etykiet jest osobnym eksperymentem.

- Start: 30 prawdziwych etykiet na klasę, 300 łącznie; wspólna inicjalizacja.
- Po epokach 3, 6, 9, 12, 15: po 40 nowych prawdziwych etykiet.
- Epoki 16–20 korzystają z 500 etykiet. Końcowy rozkład klas może być nierówny.
- W każdej rundzie losujemy bez etykiet do 4096 kandydatów z pozostałej U.
  To jawne ograniczenie kosztu pilota; nie skanujemy całej U.
- `random`: losowy wybór z kandydatów — obowiązkowa kontrola.
- `entropy`: największa entropia predykcji.
- `entropy_diverse`: 10 razy więcej najbardziej niepewnych kandydatów niż
  liczba zapytań, następnie zachłanny wybór według odległości kosinusowej od
  dotychczasowego L i już wybranych przykładów. To heurystyka, nie BADGE.

Każdy wariant wykonuje ten sam typ forwardów do oceny kandydatów i L,
także kontrola losowa. Koszt ten jest zapisany osobno. Zapytania nie korzystają
z prawdziwych klas kandydatów. Dopiero wybrane indeksy przekazujemy do oracle,
który ujawnia etykiety, kontroluje limit 500 i wyklucza walidację. Zapytane
obrazy przechodzą z U do L i nie mogą być zapytane ponownie.

Model jest kontynuowany między rundami; optymalizator i harmonogram nie są
resetowane. Generator minibatchy L zachowuje stan, U jest przebudowane po
usunięciu zapytanych obrazów. Nie zmieniamy architektury ani liczby kroków.
Główne porównanie: strategie aktywne kontra stopniowe losowe pozyskiwanie
etykiet. Dotychczasowy wariant z 500 zbalansowanymi etykietami od początku
jest dodatkowym odniesieniem, ma inny harmonogram dostępu do informacji.

`history.csv` rozróżnia etykiety użyte do uczenia w danej epoce i budżet po
zapytaniu. Wynik walidacji w rundzie pozyskania dotyczy modelu sprzed treningu
na nowych etykietach. Zapisujemy indeksy zapytań, ujawnione klasy, końcowy
histogram klas, L/U views, forwardy selekcji i checkpoint całego stanu.

## Uruchomienie, wznowienie i koszty

```powershell
python scripts/run_next_pilot.py
```

Polecenie czeka na CE40, potem wykonuje audyty i siedem nowych treningów.
Ponowne uruchomienie wznawia checkpointy i pomija ukończone zgodne runy oraz
analizy. Blokada procesu zapobiega równoległym kopiom tej kolejki i jest
zwalniana przez system również po restarcie komputera. Sam restart nie
uruchamia kolejki automatycznie. Stan: `outputs/pilot_next/queue.json`.
Końcowe wyniki: `outputs/pilot_next/results.json`.

Każdy nowy trening ma 20 epok i seed=0. Na podstawie dotychczasowych czasów:
około 35–45 minut na trójki L oraz około 75–95 minut na każdy z sześciu
wariantów korzystających z U, plus analizy. Cała seria orientacyjnie 8–11
godzin po CE40. Nowe mechanizmy mogą zmienić te czasy.

1000 dodatkowych etykiet walidacyjnych jest wspólnym zasobem; nie liczymy
ich jako treningowych ani nie dopuszczamy do U. Oficjalny test oceniamy po
ostatniej epoce każdego z góry określonego runu. Nie dobieramy na nim wag,
progów ani harmonogramów. Jest to seria eksploracyjna; kolejne seedy będą
potrzebne przed wnioskowaniem o małych różnicach.

Źródła: [FaceNet](https://arxiv.org/abs/1503.03832),
[SoftMatch](https://arxiv.org/abs/2301.10921),
[BADGE](https://arxiv.org/abs/1906.03671) — punkt odniesienia dla późniejszego
porównania aktywnej selekcji, nie implementacja obecnej heurystyki.
