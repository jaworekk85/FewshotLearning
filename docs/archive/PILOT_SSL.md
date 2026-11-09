> Dokumentacja ukończonego eksperymentu; wyniki: [RESULTS.md](../RESULTS.md).
> Opis służy interpretacji i odtworzeniu wyników, nie jest listą zadań.

# Pilot SSL: wspólne batche, trójki i pseudoetykiety

Seria CIFAR-10, N=50, seed=0 została ukończona. Wyniki CE z jednym widokiem
(29,77% dla N=10 i 49,62% dla N=50) pozostają osobnymi odniesieniami;
nie należą do poniższego protokołu dwóch widoków L.
Konfiguracja: `configs/cifar10_pilot_ssl.json`.

## Warianty

Wszystkie używają tego samego CNN ze 111 498 parametrami, losowej inicjalizacji
seed=0, podziału danych, SGD, harmonogramu i dwóch słabych widoków danych L.
CE jest średnią po obu widokach. Trójki korzystają z obliczonych embeddingów
i nie wymagają dodatkowych przejść obrazów przez encoder.

| `--method` | CE na dwóch widokach L | triplet_L | triplet_U | pseudo-CE na U |
| --- | --- | --- | --- | --- |
| `ce_2v` | tak | nie | nie | nie |
| `ce_2v_triplet_l` | tak | tak | nie | nie |
| `ce_2v_triplet_u` | tak | nie | tak | nie |
| `ce_2v_triplet_lu` | tak | tak | tak | nie |
| `ce_2v_pseudo` | tak | nie | nie | tak |
| `ce_2v_triplet_lu_pseudo` | tak | tak | tak | tak |

Dodatkowo `ce_2v_u_control` przepuszcza te same widoki U, lecz nie dodaje ich
bezpośredniej straty. Jest kontrolą kosztu i wpływu statystyk BatchNorm.
Przy wspólnym forwardzie BatchNorm korzysta ze wszystkich widoków: samo
przepuszczenie U może zmienić uczenie. Tego efektu nie przypisywać automatycznie
pseudoetykietom lub triplet loss.

## Pseudoetykietowanie

1. Obraz U ma słaby widok (crop + flip) i mocny (crop + flip + RandAugment).
2. Predykcja na słabym widoku jest odłączona od gradientu i tworzy twardą
   pseudoetykietę, jeśli pewność >= 0.95.
3. Cross-entropy na mocnym widoku jest mnożona przez maskę akceptacji.
4. Średnia jest liczona po **całym batchu U**, także odrzuconych obrazach.
   Nie uśredniamy tylko po zaakceptowanych obrazach.

To wariant inspirowany FixMatch, a nie wierna reprodukcja jego wyników:
ma inny encoder, dwa widoki L, U:L około 1 i krótki harmonogram. Wspólny
forward L/U daje słabe i mocne predykcje bez dodatkowego przejścia nauczyciela.
Nie ma osobnego nauczyciela EMA ani pretreningu na pełnych etykietach.

Pseudoetykiety nie zwiększają budżetu prawdziwych etykiet. Obrazy U nigdy nie
zwracają ground-truth label; walidacja i test nie wchodzą do U.
Domyślnie przy N=50 pula L ma 500 obrazów, walidacja 1000, a U 48 500.
Etykiety walidacyjne pozostają osobnym zasobem.

## Relacje metryczne

`triplet_L` losuje pozytyw z drugiego widoku innego oznaczonego obrazu tej
samej klasy w batchu, a negatyw spośród innych znanych klas. Jeśli w batchu
jest tylko jeden obraz danej klasy, pozytywem jest drugi widok tego obrazu.
Gdy nie ma negatywu innej klasy, kotwica nie wnosi straty metrycznej.

`triplet_U` używa słabego widoku jako kotwicy, mocnego widoku tego samego
obrazu jako pozytywu i innego losowego obrazu batcha jako negatywu.
Negatyw nie jest gwarantowanym przykładem innej klasy.

Embeddingi dla metryki są L2-normalizowane, z marginesem 0.2.
Głowica CE dostaje te same nienormalizowane reprezentacje we wszystkich
wariantach. Wagi dodatkowych strat domyślnie wynoszą 1.0; można je jawnie
zmieniać flagami `--triplet-l-weight`, `--triplet-u-weight`, `--pseudo-label-weight`.
Nie dobierać tych wag na oficjalnym teście.

## Trójki dobierane według pseudoetykiet

`ce_2v_pseudo_guided_triplet` dodaje do CE i pseudo-CE trójki klasowe:

1. Kotwica to obraz U z pseudoetykietą o pewności co najmniej 0.95.
2. Pozytyw to inny obraz o tej samej znanej lub zaakceptowanej przewidywanej klasie.
3. Negatyw to obraz o innej znanej lub zaakceptowanej przewidywanej klasie.
4. Kandydaci pochodzą z aktualnego batcha L i zaakceptowanej części batcha U.
   Brak pozytywu lub negatywu oznacza pominięcie kotwicy, bez losowego zastępstwa.

Kotwica i negatyw korzystają ze słabego widoku, pozytyw z drugiego widoku
(słabego dla L, mocnego dla U). Wektory są L2-normalizowane. Suma strat
trójek jest dzielona przez pełny rozmiar batcha U, również pominięte kotwice.
Pewność modelu nie gwarantuje poprawności pseudoetykiety; błędna klasa może
zniekształcić relacje metryczne. Raportujemy odsetek poprawnie zbudowanych
trójek (kompletność, nie poprawność klas) oraz pochodzenie pozytywów i negatywów.

`--pseudo-triplet-pool pseudo_only` wyłącza znane L jako kandydatów,
umożliwiając osobną ablację wpływu prawdziwych etykiet. Domyślnie używamy
`labeled_and_pseudo`. Waga tego członu: `--pseudo-triplet-weight` (domyślnie 1).

`ce_2v_pseudo_random_triplet` to kontrola: te same widoki L/U, CE, pseudo-CE
i dotychczasowe trójki instancji U, bez wykorzystania pseudoetykiet w ich doborze.
Oba warianty wykonują te same przejścia encodera; różnią się relacjami metrycznymi.
Dobór trójek dodaje koszt operacji na embeddingach. Inicjalizacja, obrazy,
augmentacje i liczba aktualizacji pozostają sparowane z `ce_2v_pseudo`.
Żaden wariant nie zwiększa budżetu prawdziwych etykiet.

```powershell
python scripts/run_supervised.py --config configs/cifar10_pilot_ssl.json --method ce_2v_pseudo_guided_triplet --print-spec
```

To eksperymentalna kombinacja istniejących mechanizmów; samo połączenie
pseudoetykiet i uczenia reprezentacji nie jest jeszcze dowodem nowości naukowej.
Pokrewne podejście łączy spójność semantyczną i instancyjną:
[SimMatch](https://arxiv.org/abs/2203.06915).

## Sparowanie i rozliczenie kosztu

Losowanie L, losowanie U i dobór trójek mają niezależne generatory.
Augmentacje są deterministyczne względem seed, indeksu obrazu, epoki i numeru
wizyty. Dodanie operacji na U nie zmienia obrazów/augmentacji otrzymanych z L.
Zapisujemy hash początkowych wag; między metodami powinien być identyczny.
Obecna wersja wymaga `num_workers=0` dla tego protokołu sparowania.

W `history.csv` i `metrics.json` są osobne liczniki przetworzonych widoków
L/U. Sumowane są rzeczywiste wejścia do encodera, a nie liczba krotnego
wykorzystania wektorów przy tworzeniu relacji. Wszystkie widoki są liczone
jako treningowe przejścia ze wspólnym grafem i BatchNorm; walidacja i test
nie są wliczane.

Główne porównanie utrzymuje taką samą ekspozycję L i liczbę aktualizacji.
U zwiększa koszt. Wykonana kontrola CE40 w przybliżeniu
wyrównuje liczbę widoków dla wariantu o dwóch batchach L i dwóch U. Rzeczywiste
liczniki, czasy, rozmiary batchy i harmonogramy trzeba raportować; równa
liczba widoków nie daje identycznych aktualizacji ani czasu/FLOPs.

Akceptacja pseudoetykiet jest raportowana jako liczba zaakceptowanych
prezentacji obrazów, nie liczba unikatowych dodatkowych anotacji.
Zapisujemy udział akceptacji, średnią pewność i histogram przewidywanych klas.
Nie mierzymy ich poprawności ukrytymi etykietami U podczas wyboru wariantu.

## Odtwarzanie

Uruchomienie pojedynczego wariantu opisuje [GETTING_STARTED.md](../GETTING_STARTED.md).
Własne powtórzenie wymaga osobnej nazwy runu lub katalogu wyników.

Źródła: [FixMatch](https://arxiv.org/abs/2001.07685),
[implementacja autorów USB](https://github.com/microsoft/Semi-supervised-learning/blob/main/semilearn/algorithms/fixmatch/fixmatch.py).
