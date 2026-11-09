> Dokumentacja ukończonego eksperymentu; wyniki: [RESULTS.md](../RESULTS.md).
> Opis służy interpretacji i odtworzeniu wyników, nie jest listą zadań.

# Jakość zbioru wybranego przez AL przy etykietach dostępnych od początku

Ukończono dwa treningi porównujące jakość końcowego zbioru 500 oznaczonych
obrazów przy dostępie do wszystkich etykiet od początku. Konfiguracja:
`configs/cifar10_pilot_active_subset_control.json`.

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

Porównanie sprawdza, czy różnica między końcowymi zbiorami BADGE i random
utrzymuje się po treningu od początku.
To dwa konkretne zbiory i seed 0; nie dowodzi przewagi strategii w ogólności.
Początkowe 300 było wspólne, aktywnie wybierano tylko 200 z końcowych 500.

Wynik 65,34% z 80 epok dotyczy osobnego harmonogramu i innego zbioru L.
Te dwie kontrole mają 20 epok, żeby porównać je z dotychczasowymi runami AL
bez dodatkowej zmiany długości harmonogramu. Koszt pierwotnej selekcji AL
już poniesiono; nowe treningi mierzą jakość jej końcowego zbioru, nie pełny
koszt całej procedury wyboru i ponownego uczenia.

## Artefakty

Wersjonowane runy: `results/runs/pilot_active_subset_control/`:
`cnn_N500_seed0_soft_frozen_random` i `cnn_N500_seed0_soft_frozen_badge`.
Lokalny raport: `outputs/pilot_active_subset_control_analysis/comparison.md`
i `.json`. Weryfikuje niezmienność źródeł, dokładność indeksów, wspólną
inicjalizację i identyczne liczniki nowych runów.

Odświeżenie raportu bez treningu:

```powershell
python scripts/summarize_active_subset_control.py
```
