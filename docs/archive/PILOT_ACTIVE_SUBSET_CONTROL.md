> Historical lab note. Status/ETA below may be outdated; see [current results](../RESULTS.md).

# Jakość zbioru wybranego przez AL przy etykietach dostępnych od początku

Zlecone 18.10.2026. **Dwa nowe treningi**. Cel: oddzielić jakość końcowego
zbioru 500 oznaczonych obrazów od historii stopniowego dostępu do etykiet.
Uruchomiono **18.10.2026 o 23:07 czasu polskiego**, lokalnie na CPU,
worker **PID 31320**. Logi: `outputs/pilot_active_subset_control/queue_20261018.stdout.log`
i `queue_20261018.stderr.log`. Przed startem przeszły wszystkie **71 testów**.

## Dokładne źródła danych

Oba źródła to ukończone runy miękkich celów i wag, bez trójek:

- losowanie: `outputs/pilot_validation_study/cnn_N30to500_seed0_soft_active_random`;
- BADGE: `outputs/pilot_validation_study/cnn_N30to500_seed0_soft_active_badge`.

Nie używamy wcześniejszych runów z twardymi pseudoetykietami ani wariantów
z miksem trójek. Nie wybieramy kolejnych obrazów i nie kupujemy etykiet.
Nowe modele otrzymują wszystkie **dokładnie te same końcowe 500 prawdziwych
etykiet swojego runu źródłowego** od pierwszej aktualizacji.

Migawki:
`configs/active_subset_control/random_final500.json`,
`configs/active_subset_control/badge_final500.json`.
Zawierają indeksy L, oryginalną walidację, tylko wcześniej ujawnione etykiety
L oraz SHA256 źródłowych config/metrics/splits/acquisitions. Weryfikujemy,
że 500 indeksów to suma początkowych 300 i rzeczywistych 200 zapytań AL.
Indeksy L są sortowane kanonicznie, następnie tasowane w treningu. Nie
zachowujemy kolejności zapytań jako dodatkowej zmiennej tego porównania.

Rzeczywiste liczby na klasę pozostają takie, jakie wybrały strategie.
Nominalne `budget_per_class=50` oznacza tu **500 łącznie**, nie wymuszenie
50 na każdą klasę. Nie losujemy nowych danych przez `balanced_splits`.
Ta sama walidacja 1000 obrazów co w obu źródłach, wyłączona z L i U.

## Wspólny trening

- Ta sama CNN, BatchNorm i seed 0; od tych samych losowych wag, których hash
  sprawdzamy względem inicjalizacji modeli źródłowych AL.
- Miękkie cele i adaptacyjne wagi pseudoetykiet (`ce_2v_pseudo_soft`).
- 20 epok × 383 aktualizacje = 7660, SGD lr 0,03, momentum 0,9,
  weight decay 0,0005, cosine scheduler rozpisany na pełne 20 epok.
- L: 500 od początku, dwa słabe widoki, batch size 128, zachowany końcowy
  niepełny minibatch, identycznie dla obu nowych runów i poprzedniego
  wariantu ze stałymi 500 etykietami.
- U: pozostałe 48 500 obrazów, bez etykiet w loaderze, widok weak + strong,
  batch 128. Skład U odpowiada wybranemu końcowemu L od początku treningu.
- Rzeczywiste oczekiwane wejścia encodera na run: L **1 915 360**,
  U **1 960 960**. Oba nowe runy muszą mieć identyczne liczniki.
- Żadnego przenoszenia wag, optymalizatora, pseudoetykiet ani statystyk
  BatchNorm z modeli AL. Pseudoetykiety powstają na bieżąco w nowym treningu.
- Test oficjalny 10 000 obrazów wyłącznie po 20. epoce, ostatni checkpoint.
  Walidacji/testu nie używamy do nowej selekcji L ani strojenia ustawień.

## Interpretacja

Najważniejsze porównanie: **nowy BADGE od początku kontra nowy random od
początku**, przy identycznym uczeniu i ekspozycji. Różni je wybrany zbiór L
(w tym rozkład klas) i wynikający z tego skład U.

Porównanie online kontra od początku dla dokładnie tego samego L zmienia
łącznie: moment ujawnienia etykiet, ich historię ekspozycji, skład U w czasie
oraz liczbę przejść L. Online miało 1 766 896 wejść L — o około 7,75% mniej
przez niepełne minibatch'e przy zmieniającej się liczbie L — i tylko pięć
epok z pełnym budżetem. Nie przypisujemy poprawy jednej z tych przyczyn.
Nie nazywamy tego izolowanym pomiarem samego opóźnienia etykietowania.

Jeśli zbiór BADGE wygra po treningu od początku, będzie to wskazówka, że
wcześniejszy protokół online mógł ograniczać jego użyteczność. Jeśli nie,
sam moment ujawnienia etykiet nie wyjaśni gorszego wyboru w tym porównaniu.
To dwa konkretne zbiory i seed 0; nie dowodzi przewagi strategii w ogólności.
Początkowe 300 było wspólne, aktywnie wybierano tylko 200 z końcowych 500.

Wynik 65,34% z 80 epok dotyczy osobnego harmonogramu i innego zbioru L.
Te dwie kontrole mają 20 epok, żeby porównać je z dotychczasowymi runami AL
bez dodatkowej zmiany długości harmonogramu. Koszt pierwotnej selekcji AL
już poniesiono; nowe treningi mierzą jakość jej końcowego zbioru, nie pełny
koszt całej procedury wyboru i ponownego uczenia.

## Uruchomienie i artefakty

```powershell
python scripts/prepare_active_subset_control.py
python scripts/run_next_pilot.py --plan configs/cifar10_pilot_active_subset_control.json --print-plan
python scripts/run_next_pilot.py --plan configs/cifar10_pilot_active_subset_control.json --start-prepared --device cpu
```

Pierwsze polecenie tylko przygotowuje migawki i plan; jest idempotentne,
odmawia nadpisania plików o innej treści. Samo nie trenuje. Plan pozostaje
manual-start-only, z blokadą pojedynczego workera i wznowieniem na granicach
epok. Przed wznowieniem sprawdzić PID w `queue.json`.

Stan: `outputs/pilot_active_subset_control/queue.json`.
Runy: `cnn_N500_seed0_soft_frozen_random` i `cnn_N500_seed0_soft_frozen_badge`.
Automatyczny raport po każdym ukończonym runie:
`outputs/pilot_active_subset_control_analysis/comparison.md` i `.json`.
Weryfikuje niezmienność źródeł, dokładność indeksów, wspólną inicjalizację
i identyczne liczniki nowych runów. Standardowy audyt reprezentacji również
wykonuje się po każdym modelu. Wspólna tabela pilota obejmuje nowy katalog.

Odświeżenie raportu bez treningu:

```powershell
.\.venv-analysis\Scripts\python.exe scripts/summarize_active_subset_control.py
```

Orientacyjny czas na lokalnym CPU: około **80–100 minut na run**, łącznie
**2,5–3,5 godziny** z analizami, jeśli utrzyma się ostatnie tempo.
Nie uruchamiamy chmury ani nowych płatnych zasobów.
