> Dokumentacja ukończonego eksperymentu; wyniki: [RESULTS.md](../RESULTS.md).
> Opis służy interpretacji i odtworzeniu wyników, nie jest listą zadań.

# Porównanie BatchNorm i braku normalizacji

Ukończona ablacja. Konfiguracja: `configs/cifar10_pilot_normalization.json`.
Zakres to mała ablacja CE i CE+pseudoetykiety z BatchNorm i bez normalizacji.
GroupNorm usunięto z kolejki na prośbę użytkownika przed rozpoczęciem treningów.
Nie rozszerzamy wszystkich kombinacji trójek, AE i AL na pełną siatkę.

| Normalizacja | CE | CE + pseudoetykiety |
| --- | --- | --- |
| BatchNorm | istniejący wynik 49.25% test | istniejący wynik 58.25% test |
| Brak normalizacji | ukończony trening | ukończony trening |

Nie powtarzamy dwóch ukończonych wyników z BatchNorm ani wspólnej kontroli
CE+forward U wykonanej w bloku rekonstrukcji.

## Wspólne warunki

CIFAR-10, seed=0, N=50 (500 etykiet treningowych), walidacja 1000, 20 epok
referencyjnych, 7660 aktualizacji. Ten sam SGD, lr=0.03, momentum=0.9,
weight decay=0.0005, cosine na cały trening i wybór ostatniej epoki.

W architekturze CNN zamieniamy wyłącznie trzy warstwy normalizacji:
BatchNorm na Identity. Wymiary warstw i embedding
128D są te same. Konwolucje, embedding i klasyfikator mają identyczne wagi
początkowe; normalizacja jest zamieniana po inicjalizacji pozostałych warstw,
bez dodatkowych losowań RNG. Pełny hash modelu różni się między architekturami,
dlatego dodatkowo zapisujemy hash wspólnych parametrów konwolucyjnych/liniowych.

Liczba parametrów: BatchNorm 111 498, bez normalizacji
111 050 (brak 448 parametrów affine). Identity nie agreguje danych
z różnych obrazów w paczce. Nie zmieniamy dotychczasowego skalowania pikseli
mean/std — ablacja dotyczy normalizacji wewnętrznych warstw sieci.

CE: dwa widoki L i łącznie 1 915 360 wejść. CE+pseudoetykiety: te same L
oraz 1 960 960 widoków U, łącznie 3 876 320 wejść. Ekspozycje są zgodne
z odpowiednikami BatchNorm. CE i SSL mają różny koszt; wspólna tabela pokazuje
rzeczywiste liczniki. Usunięcie BatchNorm nie oznacza sztucznego dodania U do CE.

Przy stałym optymalizatorze oceniamy wpływ zamiany normalizacji w obecnej
recepturze. To nie dowód, że któryś wariant jest najlepszy po osobnym strojeniu.

## Wyniki

Wykonano dwa treningi bez normalizacji; wyniki i konfiguracje są w
`results/runs/pilot_normalization/`. Wspólna tabela wszystkich rodzin:
`results/cifar10/comparison.csv` i `.json`, z kolumną normalizacji.
