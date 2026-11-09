> Dokumentacja ukończonego eksperymentu; wyniki: [RESULTS.md](../RESULTS.md).
> Opis służy interpretacji i odtworzeniu wyników, nie jest listą zadań.

# Sąsiedztwo jako sygnał wiarygodności pseudoetykiet

Ukończono cztery treningi; dwa istniejące wyniki kontrolne wykorzystano ponownie.
Konfiguracja: `configs/cifar10_pilot_neighbor_reliability.json`.

Cel: sprawdzić, czy niezgodność pewnych pseudoetykiet z oznaczonymi sąsiadami
pomaga ustalać ich wagi lub wybierać nowe obrazy do oznaczenia.
Diagnoza na walidacji uzasadnia hipotezę, ale nie dowodzi poprawy uczenia ani nowości metody.

## Sześć wierszy porównania, cztery nowe treningi

| Budżet | Wagi sąsiedzkie | Wybór nowych etykiet |
|---|---|---|
| 500 od początku | wyłączone, kontrola | brak |
| 500 od początku | włączone | brak |
| 300 → 500 | wyłączone, kontrola | losowanie |
| 300 → 500 | wyłączone | pewna niezgodność + entropia |
| 300 → 500 | włączone | losowanie |
| 300 → 500 | włączone | pewna niezgodność + entropia |

Kontrola statyczna pochodzi z `outputs/pilot_next/cnn_N50_seed0_ce_2v_pseudo_soft`.
Kontrola losowego AL pochodzi z `outputs/pilot_validation_study/cnn_N30to500_seed0_soft_active_random`.
Obie są ukończone; konfigurację i ekspozycję sprawdzamy przed ich ponownym wykorzystaniem.
Rozpoczęta pełna powtórka kontroli została zastąpiona krótką weryfikacją zgodności;
jej częściowych wyników nie zaliczamy do nowych wyników badawczych.
19.10.2026 pierwsza epoka odtworzyła 32 wspólne metryki i liczniki wcześniejszego
treningu, przy identycznym podziale oraz inicjalizacji. Potwierdzenie:
`outputs/pilot_neighbor_reliability/control_verification.json`.
Pełną powtórkę zatrzymano, a skróconą kolejkę uruchomiono z czterema nowymi treningami.

Ten sam CNN, seed=0, inicjalizacja, walidacja 1000 obrazów, SGD i cosine,
20 epok × 383 kroki = 7660 aktualizacji. Miękkie cele z temperaturą 0,5
i dotychczasowe adaptacyjne wagi Gaussa. Bez trójek w tej serii, aby oddzielić efekt.
Budżet statyczny: 50 prawdziwych etykiet na klasę.
Active learning: początkowo 30 na klasę, następnie 40 nowych po epokach 3/6/9/12/15;
łącznie 500, bez wymuszania końcowego balansu klas.

## Wagi

Na początku każdej epoki zapisujemy reprezentacje **wyłącznie obecnie oznaczonych L**,
bez augmentacji, w trybie eval. Bank jest stały do kolejnej epoki; po akwizycji
następny bank obejmuje nowe etykiety. Bank odbudowuje się również po wznowieniu.

Dla istniejącego słabego widoku U obliczamy dodatkowy forward eval, bez gradientu.
Nie tworzymy nowej augmentacji. Dla znormalizowanych embeddingów:

\[
q_i(c)=\sum_{j\in NN_{15}(i)}
 \operatorname{softmax}_j(\cos(z_i,z_j)/0.2)\,\mathbf1[y_j=c],
\qquad d_i=JS(p_i^{eval},q_i).
\]

Nie korzystamy z etykiet U ani walidacji. Oba rozkłady w JS pochodzą z trybu eval;
bank L jest nieaktualizowany wewnątrz epoki, podczas gdy encoder U nadal się uczy.

Od czwartej epoki, dla pewności **treningowej** pseudoetykiety ≥0,95:

\[
r_i=0.25+0.75e^{-d_i/0.2}.
\]

Dla pozostałych obrazów i podczas pierwszych trzech epok: \(r_i=1\).
Końcowe wagi:

\[
w'_i=w_i r_i\frac{\sum_j w_j}{\sum_j w_jr_j}.
\]

Zachowujemy sumę bazowych wag Gaussa w każdym batchu. Gdy jest zerowa, pozostaje zerowa.
Nie normalizujemy osobno klas. Po przeskalowaniu pojedyncza waga może przekroczyć 1.
Cele i bazowa pewność nadal pochodzą z dotychczasowego wspólnego forwardu train
L1/L2/Uweak/Ustrong; nowa reguła zmienia tylko wagi. Wszystkie sygnały sterujące są detached.

## Akwizycja

Ta sama pula 4096 kandydatów i harmonogram co w kontroli losowej.
Z każdego zapytania 20 obrazów wybieramy według malejącego JS **wśród predykcji
o pewności ≥0,95**, a pozostałe 20 według entropii, z wyłączeniem już wybranych.
Jeśli pewnych kandydatów jest mniej niż 20, brak uzupełnia ranking entropii.
Wybór nie ma dostępu do etykiet kandydatów; oracle ujawnia je dopiero po wyborze.
Komponent wyboru i wyniki punktacji są zapisywane w `acquisitions.json`.

## Porównywalność i koszty

Bank epokowy i dodatkowy forward U obliczamy tylko przy nowych wagach.
Sam nowy selektor AL oblicza sąsiedztwo w rundach akwizycji.
Tryb eval nie aktualizuje BatchNorm; zachowujemy tryby wszystkich modułów i RNG.
Testy wymagają identycznych bitowo wag kontroli względem starego soft SSL.
Dlatego wykorzystujemy wcześniejsze wyniki, zamiast powtarzać ich pełne treningi.
Oddzielnie zapisujemy dodatkowe forwardy banku, U oraz punktacji akwizycji.

| Ekspozycja przez 20 epok, dla nowych wag | Statyczne L | Active L |
|---|---:|---:|
| Widoki L z gradientem | 1 915 360 | 1 766 896 |
| Widoki U z gradientem | 1 960 960 | 1 960 960 |
| Dodatkowy bank L, eval | 10 000 | 8 200 |
| Dodatkowe U, eval | 980 480 | 980 480 |
| Punktacja akwizycji, eval | 0 | 22 380 |

W kontrolach oraz przy samym nowym selektorze AL bank epokowy i dodatkowe U eval
mają licznik 0. Punktacja akwizycji nadal wynosi 22 380 w każdym wariancie active.

Statyczną parę porównujemy ze sobą, a cztery warianty active ze sobą.
Zachowujemy wcześniejszą różnicę ekspozycji statyczne L vs active L; jej nie ukrywamy.
Wspólny raport sprawdza inicjalizację, walidację, liczniki w każdej grupie i zachowanie sumy wag.
Oficjalny test jest oceniany dopiero dla końcowego checkpointu; wybór checkpointu = last.
To pilot z jednym seedem, wybrany po analizie walidacji, bez twierdzeń potwierdzających statystycznie.

## Artefakty

Wersjonowane runy: `results/runs/pilot_neighbor_reliability/`.
Lokalny raport: `outputs/pilot_neighbor_reliability_analysis/comparison.md`.
Reprezentacje przeanalizowano po każdym modelu.
Odświeżenie raportu bez treningu:

```powershell
python scripts/summarize_neighbor_reliability.py
```
