> Dokumentacja ukończonego eksperymentu; wyniki: [RESULTS.md](../RESULTS.md).
> Opis służy interpretacji i odtworzeniu wyników, nie jest listą zadań.

# AE/VAE we wspólnym pilocie

Ukończono pięć treningów według `configs/cifar10_pilot_reconstruction.json`.
Protokół zachował wejścia, augmentacje, seed i liczbę aktualizacji wcześniejszej
serii. Wariant z dodatkowym maskowaniem oraz osobny 25-epokowy run zostały
wycofane przed treningiem. Nie należą do wykonanych porównań.

## Jedno porównanie

Wspólna tabela: `outputs/pilot_comparison.md`, `.csv` i `.json`.
Odświeża się po każdym audycie modelu. Ręczne odświeżenie:

```powershell
python scripts/summarize_pilot.py
```

Tabela obejmuje CE, trójki, pseudoetykietowanie, miękkie warianty, active learning
i AE/VAE. Pokazuje rzeczywiste wejścia L/U, aktualizacje, epoki, wejścia dekodera
i dokładność. CSV/JSON ponadto zawierają czas, parametry, forwardy selekcji oraz
hash początkowych wag i podziałów danych.

CE/L-only ma mniej wejść U; active learning ma harmonogram 300 → 500 etykiet.
Strategie AL porównujemy przede wszystkim z losowym pozyskiwaniem w tych samych
rundach. Wcześniejsze mniejsze minibatche L oznaczają trochę inną ekspozycję L
niż przy stałych 500 etykietach — dlatego pokazujemy rzeczywiste liczniki.

## Pięć wykonanych treningów

Konfiguracja: `configs/cifar10_pilot_reconstruction.json`.

| Wariant | Cel |
| --- | --- |
| `ce_2v_u_control` | Kontrola wpływu U na BatchNorm |
| `dae` | CE + rekonstrukcja AE |
| `dae_pseudo` | CE + rekonstrukcja AE + pseudoetykiety |
| `dvae` | CE + rekonstrukcja VAE + KL |
| `dvae_pseudo` | CE + rekonstrukcja VAE + KL + pseudoetykiety |

Istniejące CE i CE+pseudoetykiety są ponownie wykorzystywanymi odniesieniami.
Nie powtarzamy ich na maskach. Usunięto osobny 25-epokowy run; CE40 pozostaje
dotychczasową kontrolą dłuższego treningu.

Wszystkie nowe runy: CIFAR-10, N=50 (500 prawdziwych etykiet), walidacja 1000,
U 48 500, seed=0, 20 epok referencyjnych po 383 aktualizacje. Ten sam CNN 128D,
głowica CE, początkowe wagi, SGD i harmonogram.
Każdy nowy model ma 1 915 360 widoków L i 1 960 960 widoków U: łącznie
3 876 320 wejść encodera, zgodnych z dotychczasowymi wariantami SSL o stałym N=50.

## Rekonstrukcja bez zmiany wejścia

Zachowujemy dwa słabe widoki L oraz słaby i mocny U. Transform tylko zapamiętuje
cel po cropie/odbiciu mocnej gałęzi, przed jej dotychczasowym RandAugment.
Nie zmienia losowań ani samych wejść. Dekoder odtwarza ten kadr ze standardowej
mocno augmentowanej reprezentacji. RandAugment obejmuje też możliwe zaburzenia
geometryczne; zadaniem jest odtworzenie kadru sprzed tych zaburzeń. Nie jest
to nowy eksperyment z wycinaniem 25% bloków.

Cel RGB pochodzi z obrazu i nie wymaga etykiety. Nie przechodzi przez encoder
ani nauczyciela. Pozostaje jedno wspólne wywołanie encodera L1/L2/Uweak/Ustrong.
Dodatkowe moduły mają niezależną inicjalizację i RNG. Test porównuje wejścia
z oryginalnym `PilotTraining` bit po bicie w kilku epokach i wizytach.
Osobny test potwierdza identyczne końcowe wagi klasyfikatora po wyłączeniu
dodatkowej straty AE względem oryginalnego pseudoetykietowania.

## Cele i koszt dodatkowych modułów

Dekoder: 128D → linear do 64x4x4 → trzy bloki upsample+conv → 32x32x3,
ReLU między blokami, sigmoid na końcu, bez połączeń omijających embedding.
Dekodowany jest mocny U: 980 480 prezentacji na run. MSE jest średnią po całym
obrazie, kanałach i batchu; waga 1.0.

VAE używa wspólnego embeddingu jako średniej posterioru. Dodatkowa głowica
128→128 przewiduje z niego log-wariancję ograniczoną do [-10,10]. Dekoder
dostaje próbkę `mu + exp(0.5*logvar)*epsilon`, klasyfikator zawsze deterministyczne
mu. KL do N(0,I) jest średnią po batchu i współrzędnych, a jego waga przez pięć
epok rośnie od 0.002 do 0.01. To ograniczona rodzina posterioru, nie reprodukcja M2.

Pseudoetykietowanie zachowuje próg 0.95, twarde odłączone cele ze słabego U
i pseudo-CE na tym samym mocnym U co wcześniej. Nie dodajemy tu trójek ani
miękkiego ważenia. Parametry klasyfikatora: zawsze 111 498. Parametry treningowe
AE: 267 117, VAE: 283 629. Dodatkowy koszt dekodera i gradientu raportujemy
osobno; równe epoki/wejścia encodera nie oznaczają równych FLOPs.

## Artefakty i uruchamianie

`latest.pt` zawiera wszystkie moduły, optymalizator, harmonogram i RNG.
`model.pt` to zwykły encoder+klasyfikator, zgodny z audytem reprezentacji.
Po każdym runie powstają PCA/t-SNE/kNN/linear probe i rekonstrukcje ośmiu stałych
obrazów walidacyjnych: cel, istniejący mocny widok i wynik. Bez etykiet U/testu.

Lokalne wyniki: `outputs/pilot_reconstruction/results.json`.
Wersjonowane konfiguracje, historie i metryki: `results/runs/pilot_reconstruction/`.
