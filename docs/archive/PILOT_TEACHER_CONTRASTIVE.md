> Dokumentacja ukończonego eksperymentu; wyniki: [RESULTS.md](../RESULTS.md).
> Opis służy interpretacji i odtworzeniu wyników, nie jest listą zadań.

# Temperatura celów, nauczyciel EMA i mieszane pary kontrastowe

Ukończono pięć treningów; wcześniejszy wynik soft SSL wykorzystano ponownie.
Konfiguracja: `configs/cifar10_pilot_teacher_contrastive.json`. Osobno wykonano analizę istniejącej CNN z pełnym nadzorem,
bez nowego treningu ani włączania jej etykiet do eksperymentów z budżetem 500.

## Wykonane porównania

| Wariant | Temperatura celu | Źródło pseudoetykiet | Nowa strata kontrastowa |
|---|---:|---|---|
| Kontrola, istniejąca | 0,5 | student | brak |
| A | 1 | student | brak |
| B | 0,5 | nauczyciel EMA od epoki 4 | brak |
| C | 1 | nauczyciel EMA od epoki 4 | brak |
| D | 0,5 | student | prawdziwe klasy L + augmentacje instancji L/U |
| E | 0,5 | student | jak D + pewne pary U–U i L–U |

Wszędzie: ta sama CNN 111 498 uczonych parametrów, seed0, 50 prawdziwych etykiet
na klasę, identyczna inicjalizacja i podziały, 20 epok × 383 kroki = 7660 aktualizacji
SGD i ta sama polityka cosine. Trójki oraz sąsiedzka modyfikacja wag są wyłączone,
żeby sprawdzić nowe składniki względem dotychczasowego soft SSL.
W każdej metodzie pozostaje nadzorowane CE na dwóch widokach L i ważona strata
miękkich pseudoetykiet na U. Obie mają współczynnik 1.

## Nauczyciel

Student wykonuje dotychczasowy wspólny forward `[L1,L2,Uweak,Ustrong]` z joint BN.
Nauczyciel jest oddzielną kopią tego samego modelu, bez gradientów i bez optymalizatora.
Jest inicjalizowany identycznie, a po każdym kroku SGD:

\[
\theta_T\leftarrow\alpha_t\theta_T+(1-\alpha_t)\theta_S,
\quad \alpha_t=\min(0.99,1-1/(t+1)),\quad t=1,2,\ldots
\]

Na początku szybko dogania studenta, później średni wiek wkładu przy limicie 0,99
wynosi około 99 aktualizacji, czyli 0,26 epoki. To nie jest limit 0,999 z przykładu
w rozmowie. Wszystkie bufory BN, w tym running mean/variance i licznik batchy,
kopiujemy od studenta po każdym kroku. Nauczyciel wykonuje wyłącznie forwardy eval,
więc nie dopasowuje osobno BatchNorm na dodatkowych wejściach.

Przez pierwsze 3 epoki cele nadal tworzy student; nauczyciel śledzi jego parametry
bez dodatkowych forwardów treningowych. Od epoki 4 dostaje istniejący słaby widok U
i generuje zarówno miękkie cele, jak i pewności do dotychczasowych wag Gaussa.
Nie tworzymy dodatkowych augmentacji. Student nadal ma ten sam wspólny batch,
aby zachować bazową politykę BN. Etykiety prawdziwe trafiają do CE studenta.

Podstawowym wynikiem jest **student, ostatnia epoka**. Walidację i test nauczyciela
pokazujemy osobno, bez wyboru lepszej z dwóch sieci. Eksporty to `model.pt`
oraz `teacher_model.pt`. Pełny stan EMA, licznik aktualizacji, optimizer,
scheduler, RNG i stan pseudoetykiet są zapisywane do wznowienia.

Dodatkowy koszt EMA jest jawny:

- student: jak kontrola, 1 915 360 widoków L i 1 960 960 widoków U;
- nauczyciel przy tworzeniu celów: 833 408 forwardowanych widoków U;
- walidacja nauczyciela: 20 000 widoków, wyłącznie diagnostycznie;
- końcowy test nauczyciela: 10 000 widoków;
- dodatkowa kopia parametrów i aktualizacje EMA, bez backwardu przez nauczyciela.

To porównanie przy równych etykietach i krokach SGD, **nie przy równym koszcie
obliczeń**. Raportujemy czasy i liczniki. Jeśli EMA okaże się obiecujące, osobny
baseline o zbliżonym koszcie będzie potrzebny do twierdzenia o przewadze obliczeniowej;
tej dodatkowej kontroli nie uruchamiamy z góry.

## Jedna strata kontrastowa, mieszane źródła par

Bez nowej głowicy projekcyjnej: używamy aktualnego embeddingu 128D,
normalizowanego L2. To nasz mieszany wariant inspirowany SupCon/SimCLR,
a nie odtworzenie ich pełnych receptur.

Dla wszystkich widoków batcha konstruujemy jeden graf dodatnich wag:

1. Dwa widoki tego samego obrazu L lub U są pozytywami z wagą 1.
2. Dwa różne obrazy z prawdziwą tą samą etykietą są pozytywami z wagą 1.
3. W E dodatkowo dopuszczamy U o pewności ≥0,95. Różne obrazy z tą samą
   dostępną etykietą (true/pseudo) są pozytywami: L–U oraz U–U.
   Waga pary wynosi iloczyn pewności U; dla L pewność przyjmujemy jako 1.
4. Para własnych widoków zawsze ma wagę 1 i pierwszeństwo przed regułą klasową.
5. Przekątna jest wyłączona. Pozostałe pary są negatywami; nie dodajemy osobnej
   straty instancyjnej, która odpychałaby równocześnie zdefiniowane pozytywy.

Nie znamy prawdziwych klas U. Różne obrazy o odrzuconej pseudoetykiecie nadal
mogą być fałszywymi negatywami; graf usuwa sprzeczności według dostępnych etykiet,
nie gwarantuje poprawności wszystkich relacji semantycznych.

\[
L_{con}=-\frac1N\sum_i\frac1{\sum_j w_{ij}}
\sum_j w_{ij}\log\frac{\exp(z_i^Tz_j/0.2)}{\sum_{k\ne i}\exp(z_i^Tz_k/0.2)}.
\]

Każdy anchor ma co najmniej drugi widok swojego obrazu. Wszystkie wagi par są
detached. Parametry pseudoetykiet pochodzą z predykcji słabego widoku,
bez odczytu ukrytych etykiet U. Cele CE i soft SSL pozostają jak w kontroli.
Od epoki 4:

\[
L=L_{CE}+L_{softPL}+0.05L_{con}.
\]

W pierwszych 3 epokach nowy człon daje zero. D i E różnią się tylko dopuszczeniem
pseudoetykietowanych par klasowych. W batchu są pary dodatnie i ujemne, nie
losujemy kolejnych obrazów ani augmentacji. Nie ma dodatkowego forwardu CNN,
ale macierze podobieństwa i ich gradienty dodają koszt, uwzględniony w czasie runu.
Liczymy dodatnie pary LL, LU, UU, pary instancyjne, negatywy, sumy wag i aktywne anchory.
Liczniki par obejmują widziane grafy także podczas rozgrzewki; licznik aktywnych
anchorów dotyczy kroków rzeczywistego uczenia kontrastowego.

## Reprezentacje na wspólnej podstawie

`scripts/compare_reference_representations.py` porównuje istniejące CE500, soft20,
soft80, najlepsze nowe trójki i CNN uczoną na 49 000 etykietach. Uwzględnia
modele tej serii, a dla EMA również nauczyciela.

Wszystkie mają te same 1000 obrazów walidacyjnych, w tej samej kolejności,
i te same 500 prawdziwie oznaczonych obrazów referencyjnych do kNN i linear probe.
Żaden obraz walidacyjny nie występuje w treningu żadnego z porównywanych modeli.
Pełna CNN widziała więcej etykiet i inny protokół widoków podczas treningu:
jest diagnostycznym odniesieniem dla reprezentacji, nie konkurentem przy budżecie 500.

Raport: `outputs/representation_reference/comparison.md`.
Wykresy: `pca_comparison.png/.svg`, `tsne_comparison.png/.svg` w tym samym katalogu.
Metryki geometryczne są liczone w embeddingu 128D. Układy PCA/t-SNE dla modeli
są niezależne; nie porównujemy odległości pomiędzy punktami z różnych paneli.
Poza wykresami raportujemy head accuracy, kNN15, probe, lokalną czystość klas,
silhouette i rozrzut. Nie stroimy wizualizacji pod oczekiwany wygląd klas.

## Artefakty

Wersjonowane runy: `results/runs/pilot_teacher_contrastive/`.
Lokalny raport: `outputs/pilot_teacher_contrastive_analysis/comparison.md` i `.json`.
Wspólne porównanie reprezentacji: `outputs/representation_reference/comparison.md`.

Samodzielna analiza bez treningu:

```powershell
python scripts/compare_reference_representations.py
python scripts/summarize_teacher_contrastive.py
```

To pilot jednego seedu. Wybór parametrów znanych metod nie jest samodzielnym
wkładem naukowym; małe różnice nie dowodzą powtarzalnej przewagi.
