> Dokumentacja ukończonego eksperymentu; wyniki: [RESULTS.md](../RESULTS.md).
> Opis służy interpretacji i odtworzeniu wyników, nie jest listą zadań.

# Trójki: źródła etykiet, odległe pozytywy i sprzeczne negatywy

Ukończono cztery treningi według `configs/cifar10_pilot_mined_triplets.json`;
dwa wcześniejsze wyniki wykorzystano ponownie.
Nowość dotyczy dotychczasowych eksperymentów w repo, nie deklaracji nowości naukowej.
Audyt 46 ukończonych runów jest w `outputs/pilot_mined_triplets_analysis/novelty_audit.json`.

## Porównania

| Wariant | Anchor | Pozytyw | Negatyw | Odległość do doboru | Instancyjne negatywy |
|---|---|---|---|---|---|
| A: L–U–L | prawdziwe L | pewne U, ta sama klasa | prawdziwe L, inna klasa | embedding | wszystkie inne U |
| B: miks | L lub pewne U | ta sama klasa; dla anchor L tylko U | L lub pewne U, inna klasa | embedding | wszystkie inne U |
| C: miks RGB | jak B | jak B | jak B | piksele słabych widoków | wszystkie inne U |
| D: miks z maską | jak B | jak B | jak B | embedding | usunięte pary z tą samą pewną pseudoetykietą |

We wszystkich nowych wariantach zachowujemy trzy składniki metryczne:
nadzworowany L semi-hard, nowy klasowy L/U oraz instancyjny U.
Każdy ma współczynnik 1/30; razem nominalnie 0,1.
W A/B/C człon instancyjny pozostaje w pełni nienadzorowany. W D jego selekcja
korzysta z pseudoetykiet: to jawna ablacja możliwego konfliktu celów.

Kontrole już ukończone:

- Soft SSL bez trójek: `outputs/pilot_next/cnn_N50_seed0_ce_2v_pseudo_soft`.
- Stary miks L/pseudo/U: `outputs/pilot_validation_study/cnn_N50_seed0_ce_2v_pseudo_soft_mixed_triplet`.

B vs C zmienia wyłącznie metrykę miningu; B vs D wyłącznie maskę negatywów
instancyjnych. A vs B porównuje dopuszczalne źródła elementów trójki.
Nowe warianty różnią się od starego miksu również rozgrzewką i ważeniem;
porównanie ze starym miksem nie izoluje samego doboru pozytywów.

## Dokładna reguła

W istniejącym batchu L+U, używając wyłącznie słabych widoków:

1. Pseudoetykieta jest argmax predykcji bieżącego modelu. Dopuszczamy U z pewnością ≥0,95.
2. Losujemy bez zwracania do 128 anchorów z dopuszczalnej puli.
3. Dla każdego anchora znajdujemy dozwolone pozytywy tej samej etykiety.
   Losujemy jeden z najdalszych 25% (zaokrąglenie liczby kandydatów w górę).
4. Negatywem jest najbliższy dozwolony obraz o innej etykiecie.
5. Brak pozytywu lub negatywu daje zero dla tego anchora; nie powtarzamy próbki.

Dla wyboru embeddingowego używamy L2 pomiędzy znormalizowanymi embeddingami
słabych widoków. Dla RGB odwracamy normalizację wejściową i używamy L2 pikseli
tych samych słabych crop/flip widoków. To nie jest odległość nieaugmentowanych
obrazów źródłowych; wrażliwość na crop i flip jest ograniczeniem tej ablacji.
Dobór jest detached i nie korzysta z ukrytych etykiet U ani z walidacji.

Samą stratę zawsze liczymy w embeddingach:

\[
L_{class}=\frac{1}{128}\sum_{(a,p,n)}r_a r_p r_n
\left[\|z_a^w-z_p^s\|_2-\|z_a^w-z_n^w\|_2+0.2\right]_+.
\]

Wszystkie z są normalizowane L2. `w` to słaby widok, `s` mocny.
Dla prawdziwej etykiety r=1, dla U r=confidence. Iloczyn jest heurystyczną
wagą, a nie oszacowaniem prawdopodobieństwa poprawności całej trójki.
Stały mianownik 128 zapobiega wzmacnianiu straty, gdy mało trójek się kwalifikuje.
Nowy człon klasowy włączamy od epoki 4. CE, soft SSL i istniejące L/U działają
od początku; wszystkie cztery warianty mają tę samą rozgrzewkę.

W D pary i≠j o tej samej pseudoetykiecie i obustronnej pewności ≥0,95
nie są negatywami instancyjnymi. Uśredniamy po pozostałych parach.
Gdy żadna para nie pozostaje, człon daje różniczkowalne zero.
To świadoma zmiana składu i liczby par, raportowana w licznikach.

## Ekspozycja i odtwarzalność

Ten sam CNN 111 498 parametrów, seed0, 50 etykiet na klasę, walidacja 1000
obrazów, 20 epok × 383 kroki = 7660 aktualizacji; CE i soft SSL bez zmian.
Każdy run przetwarza 1 915 360 widoków L oraz 1 960 960 widoków U.
Nie ma nowych forwardów CNN, nauczyciela, active learning ani banku zewnętrznego.
Odległości i selekcja dodają obliczenia; liczymy czas całego treningu oraz nowego
członu klasowego (`metric_class_seconds`; timer lokalnego CPU).

Losowanie miningu używa osobnego zapisywanego generatora. Wznowienie od granicy
epoki jest sprawdzane względem treningu nieprzerwanego dla wszystkich czterech
wariantów. Nie zmieniamy RNG augmentacji ani wejść do wspólnego BatchNorm.
Checkpoint zapisywany jest po każdej epoce. Test oceniamy dopiero na końcu;
checkpoint=last. To pilot jednego seedu, nie badanie potwierdzające.

Raport sprawdza wspólne dane, inicjalizację i liczniki ekspozycji. Zapisujemy liczbę
prób/ważnych/aktywnych trójek, źródła anchorów i elementów, sumę wag,
sumy odległości pozytywów i negatywów oraz liczbę wykluczonych par instancyjnych.
Po każdym modelu automatycznie powstaje dotychczasowy audyt reprezentacji:
kNN, linear probe, lokalna czystość klas, PCA/t-SNE i pozostałe metryki.

## Artefakty

Wersjonowane runy: `results/runs/pilot_mined_triplets/`.
Lokalny raport: `outputs/pilot_mined_triplets_analysis/comparison.md`.
Odświeżenie raportu bez treningu: `python scripts/summarize_mined_triplets.py`.
