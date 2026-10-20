> Historical lab note. Status/ETA below may be outdated; see [current results](../RESULTS.md).

# Weryfikacja efektów i połączenie miękkiego SSL, mieszanych trójek i AL

**Seria ukończona 18.10.2026 bez błędów:** 13 nowych treningów i 3 ponownie
wykorzystane wyniki. Miękkie SSL seed 0: test 60,12% po 20 epokach,
63,14% po 40 i 65,34% po 80. Następny uruchomiony eksperyment opisuje
`PILOT_ACTIVE_SUBSET_CONTROL.md`.

Zlecone 17.10.2026. Plan: `configs/cifar10_pilot_validation_study.json`.
16 pozycji: **3 istniejące wyniki użyte ponownie, 13 nowych treningów**.
Uruchomiono **17.10.2026 o 22:51 czasu polskiego**, lokalnie na CPU,
poprzedni worker **PID 18548**. Logi: `outputs/pilot_validation_study/queue_20261017.stdout.log`
i `queue_20261017.stderr.log`. Przed startem przeszły wszystkie 62 testy.
Po doprecyzowaniu przez użytkownika przerwano wariant wyłącznie instancji U
po pierwszej zapisanej epoce, zachowując checkpoint i oznaczając run jako
`superseded` (wyłączony z tabeli wyników, bez testu). Plan zmieniono na miks
L/pseudo/U. Kopie starego planu i stanu są w katalogu kolejki jako
`plan_before_mixed_triplets.json` i `queue_before_mixed_triplets.json`.
Poprawioną kolejkę uruchomiono **17.10.2026 o 22:59**, worker **PID 32944**;
aktualne logi: `queue_mixed_20261017.stdout.log` i `queue_mixed_20261017.stderr.log`
w katalogu kolejki. Przed wznowieniem przeszły wszystkie **64 testy** oraz
ponowne testy wznowienia z aktywnym składnikiem pseudo-trójek.
Bez zmiany zbioru (CIFAR-10), CNN (111 498 parametrów), budżetu treningowych
etykiet (500) i bazowych augmentacji. Nie jest to pełna siatka hiperparametrów.

## 1. Powtarzalność

CE, CE + twarde pseudoetykiety oraz CE + miękkie cele i wagi, każdy dla
seedów 0,1,2, 20 epok. Trzy ukończone runy seeda 0 są tylko odczytywane,
bez ponownego treningu. **Sześć nowych runów** dla seedów 1 i 2.

W obrębie seeda identyczne L, walidacja i inicjalizacja modelu. Seed zmienia
jednocześnie podział i inicjalizację; nie izolujemy tych źródeł zmienności.
Raport pokazuje surowe wyniki, średnią i próbne SD oraz sparowane różnice po
seedach. Przy n>1 podaje eksploracyjne przedziały t-Studenta, przy n=1 nie
podaje SD/CI. Trzy seedy są pierwszym sprawdzeniem stabilności, nie gwarancją
istotności ani wystarczającej mocy statystycznej.

CE ma mniej wejść encodera niż SSL; wszystkie mają 20×383 aktualizacje.
Rzeczywisty koszt i L/U views pozostają raportowane osobno.

## 2. Czas treningu miękkich pseudoetykiet

**Dwa nowe runy**, seed 0, 40 i 80 epok. Odniesienie 20 epok już istnieje.
Wszystkie zaczynają od tych samych losowych wag i tych samych 500 etykiet.
Każdy jest uczony od początku z cosine schedulerem rozpisanym na swój pełny
horyzont. Nie przedłużamy runu, którego learning rate spadł już do zera.
Porównujemy kompletne harmonogramy, a nie sam efekt dopisania kolejnych
epok do identycznego wcześniejszego przebiegu. Ostatni checkpoint, test raz
po zakończeniu każdego runu; dodatkowy koszt 2×/4× względem 20 epok.

## 3. Reprezentacje

Istniejące embeddingi, PCA/t-SNE i pomiary nie wymagają nowych treningów.
Zestawienie siedmiu kluczowych modeli na tej samej walidacji seeda 0:

- `outputs/pilot_validation_study_analysis/study.md` i `study.json`;
- `representations.csv`, `representations.png` i `representations.svg`;
- pełne wcześniejsze embeddingi i wykresy: `outputs/representations/`.

Przykład obserwacji: twarde pseudoetykiety + klasowe trójki semi-hard mają
purity10 50,82% zamiast 47,25% i silhouette 0,0933 zamiast 0,0228. Jednak
within/total zmienia się z 0,5546 do 0,5890, czyli nie wszystkie miary się
poprawiają. To sygnał zmiany geometrii, nie dowód lepszej klasyfikacji ani
uniwersalnego pożytku z trójek. Raport używa kNN5, linear probe, purity10,
silhouette i scatter w przestrzeni 128D. Probe trenuje wyłącznie na L,
nie na walidacji. t-SNE jest ilustracją i nie służy do pomiaru separacji.

Każdy nowy ukończony model otrzyma ten sam standardowy audyt. Zestawienie
statystyczne aktualizuje się po każdym ukończonym runie; tabela geometrii
w raporcie pokazuje ustalony zestaw początkowych modeli seeda 0, a pełne
audyty nowych modeli znajdują się we wspólnym katalogu reprezentacji.

## 4. Miękkie pseudoetykiety + mieszane trójki + active learning

Z góry ustalony mały układ porównania, bez wyboru zwycięzcy na teście:

| Dostęp do etykiet | Miękkie pseudoetykiety | + miks trójek L/pseudo/U |
| --- | --- | --- |
| 500 od początku | istniejący seed 0 | nowy run |
| Losowo 300→500 | nowy run | nowy run |
| BADGE 300→500 | nowy run | nowy run |

**Pięć nowych runów**, seed 0, po 20 epok. BADGE jest ustalonym punktem
odniesienia z literatury; nie wybieramy kolejnej heurystyki według najwyższego
test accuracy wcześniejszej serii. Nie twierdzimy, że BADGE musi pomóc.
Start 30 etykiet na klasę, +40 po epokach 3,6,9,12,15; końcowo 500 łącznie,
bez gwarancji równych proporcji klas. Dla AL te same 4096 kandydatów według
niezależnego RNG, jeden widok selekcji bez augmentacji i ten sam koszt
forwardów. Kolejne pule mogą się różnić, bo strategie kupują inne indeksy.
Wnioskowanie o kryterium wyboru wymaga porównania z odpowiednim AL random.
Stałe 500 etykiet od początku jest dodatkowym odniesieniem z innym dostępem
do informacji; nie jest dokładnie tym samym zbiorem 500 co finalne AL random.

### Dokładna konstrukcja mieszanki trójek

Trzy składniki równocześnie, o równych wagach strat (nie równych liczbach
trójek). Łączna strata:

`CE_L + L_pseudo_soft + (0.1/3)*(L_labeled + L_pseudo_triplet + L_instance)`.

1. **Labeled:** kotwica L, pozytyw o tej samej prawdziwej klasie z L,
   negatyw o innej prawdziwej klasie. Pozytyw z innego obrazu, gdy jest
   dostępny, w przeciwnym razie drugi widok kotwicy. Negatyw semi-hard,
   a przy jego braku najbliższy dozwolony negatyw.
2. **Pseudo:** kotwica U o pewności >=0.95, klasowy pozytyw i negatyw
   z puli L oraz zaakceptowanych pseudoetykiet U. Nigdy drugi widok
   tego samego obrazu jako klasowy pozytyw. Negatyw semi-hard z tym samym
   fallbackiem. Prawdziwe etykiety U nie są odczytywane. Brak pary oznacza
   zero; składnik jest normalizowany przez cały batch U.
3. **Instance / bez etykiet:** opis poniżej. Dotyczy całego batcha U,
   niezależnie od pewności pseudoetykiet.

Każdy składnik jest osobno uśredniany. Jeśli nie ma poprawnych trójek
danego typu, jego składnik jest zerowy; nie przenosimy wagi na inne typy.
Ten sam obraz może uczestniczyć w składniku pseudo i instancji. Waga 0,1
jest sumą trzech wag 1/30, a nie wagą 0,1 dla każdego typu z osobna.

W istniejącym minibatchu U są słabe widoki `x_i^w` i mocne `x_i^s`.
Po normalizacji embeddingów L2:

`L_instance = mean_(i != j) max(||z_i^w-z_i^s||_2 - ||z_i^w-z_j^s||_2 + 0.2, 0)`.

- Kotwica: słaby widok obrazu i.
- Pozytyw: mocny widok **tego samego obrazu i**.
- Negatywy: mocne widoki **wszystkich innych obrazów j z tego batcha**.
- Bez sprawdzania prawdziwych klas i pseudoetykiet; dodatnia para jest znana
  z tożsamości obrazu. To uczenie samonadzorowane / instancji.
- Średnia po wszystkich N(N−1) parach, nie suma zależna od batch size.
  Batch N=1 daje zero bez niepoprawnych negatywów.
- Margines 0,2 i łączna waga 0,1 są ustalone przed uruchomieniem; nie stroimy
  ich na teście. Trójki instancji nie są mnożone przez pewność pseudoetykiety;
  składnik pseudo ma osobny twardy warunek pewności 0,95.
- Te same wejścia encodera i wspólny BatchNorm co w miękkim SSL, bez
  dodatkowych augmentacji ani przejść encodera. Dodatkowy koszt to macierz
  odległości i jej gradient. Logujemy liczbę par i aktywnych ograniczeń.
- Inny obraz może mieć tę samą prawdziwą klasę. Takie negatywy są świadomą
  cechą tego kryterium instancji; to jeden z mechanizmów możliwego konfliktu
  z celem klasyfikacji. Nie odfiltrowujemy ich ukrytymi etykietami.

Stary `ce_2v_pseudo_random_triplet` używał twardych pseudoetykiet i jednego
losowego negatywu na kotwicę. Nowy wariant dodaje miks trzech składników do
miękkiego SSL; nie izoluje wpływu każdego z trzech typów osobno. Izolację
wpływu całej mieszanki daje porównanie z nową kontrolą soft.

## Uruchomienie, artefakty, czas

```powershell
python scripts/run_next_pilot.py --plan configs/cifar10_pilot_validation_study.json --print-plan
python scripts/run_next_pilot.py --plan configs/cifar10_pilot_validation_study.json --start-prepared --device cpu
```

Kolejka ma blokadę pojedynczego workera, wznawia checkpointy i pomija zgodne
ukończone wyniki. Przy restarcie najpierw sprawdzić PID z `queue.json`, potem
użyć tego samego polecenia. Nie modyfikować planu działającego workera.
Odczyt istniejącego wyniku wymaga zgodności całej konfiguracji; brak wyniku
oznaczonego do ponownego wykorzystania jest błędem, nie nowym treningiem.

Stan: `outputs/pilot_validation_study/queue.json`. Pełne artefakty jak w
poprzednich pilotach. Raport analityczny:
`outputs/pilot_validation_study_analysis/study.md`. Globalne porównanie
obejmuje także nowe runy i pozostaje w `outputs/pilot_comparison.md/.csv/.json`.

Raport można odświeżyć bez treningu:

```powershell
.\.venv-analysis\Scripts\python.exe scripts/summarize_validation_study.py
```

Orientacyjny koszt 13 nowych treningów na obecnym CPU: **około 22–30 godzin**
z audytami. Bazuje na pomiarach CE około 36 min i SSL/AL około 78–80 min
na 20 epok. 40+80 epok miękkiego SSL to łącznie około 8 godzin. Nie jest to
pomiar nowego batch-all; ETA aktualizować według faktycznych czasów epok.
Nie uruchamiamy chmury ani płatnych zasobów.

Walidacja: osobne 100 etykiet na klasę (1000) w każdym podziale, zawsze
wyłączone z L i U. Test CIFAR-10 10 000, dopiero po końcu treningu.
Nie zwiększamy budżetu etykiet przez pseudoetykietowanie. Nie dodajemy teraz
innych zbiorów, architektur, dwóch mocnych widoków ani dodatkowych metod AL.
