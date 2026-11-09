> Dokumentacja ukończonego eksperymentu; wyniki: [RESULTS.md](../RESULTS.md).
> Opis służy interpretacji i odtworzeniu wyników, nie jest listą zadań.

# Reprezentacje, dobór trójek, miękkie pseudoetykiety i active learning

Ukończona seria CIFAR-10: siedem treningów, seed=0, mały CNN bez pretreningu.
Konfiguracja: `configs/cifar10_pilot_next.json`.

## Analiza reprezentacji

Audyt objął osiem wcześniejszych modeli, w tym CE40, oraz siedem modeli tej serii.
Dane: wyłącznie zapisany L i te same
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

## Artefakty

Lokalne wyniki: `outputs/pilot_next/results.json`; audyty:
`outputs/representations/`. Wersjonowana migawka jest w `results/`.

1000 etykiet walidacyjnych jest osobnym zasobem, wyłączonym z L i U.
Oficjalny test oceniano po ostatniej epoce każdego runu; nie dobierano na nim
wag, progów ani harmonogramów. To seria eksploracyjna jednego seedu.

Źródła: [FaceNet](https://arxiv.org/abs/1503.03832),
[SoftMatch](https://arxiv.org/abs/2301.10921),
[BADGE](https://arxiv.org/abs/1906.03671) — punkt odniesienia dla późniejszego
porównania aktywnej selekcji, nie implementacja obecnej heurystyki.
