# Mały pilot SSL: Fashion-MNIST

Uruchamiamy jedną ograniczoną serię zamiast kolejnej dużej siatki. Plan:
[configs/fashion_mnist_ssl.json](../configs/fashion_mnist_ssl.json).

## Zakres

- Budżety: **10, 30, 50 prawdziwych etykiet na klasę**, czyli 100, 300, 500 łącznie.
- Metody: CE, twarde pseudoetykiety, miękkie cele i wagi, miękkie PL + trójki B,
  miękkie PL + contrastive z pseudo-parami. Implementacje strat są wspólne z CIFAR-10.
- Seed0: 15 porównań i **jedno dodatkowe odniesienie z pełnym nadzorem** na 59 000
  etykietach. Razem 16 treningów. To pilot; nie deklarujemy istotności statystycznej.
- Na razie nie dodajemy active learningu, EMA, AE/VAE ani kolejnych miningów.

## Dane i architektura

Fashion-MNIST: 60 000 obrazów treningowych i 10 000 oficjalnych testowych,
10 klas, natywne obrazy 1×28×28. Z train odkładamy 100 obrazów każdej klasy do
walidacji. Te same obrazy walidacyjne i ich kolejność obowiązują we wszystkich
metodach i budżetach danego seedu. Pule etykiet są zagnieżdżone:
100 ⊂ 300 ⊂ 500. U to reszta train z wyłączeniem L i walidacji.

`GrayConvEncoder`, **38 378 parametrów**:

```text
1×28×28
 → Conv 1→16, 3×3 + BatchNorm + ReLU + MaxPool2: 16×14×14
 → Conv 16→32, 3×3 + BatchNorm + ReLU + MaxPool2: 32×7×7
 → AdaptiveAvgPool 4×4: 32×4×4
 → Linear 512→64: embedding 64D
 → Linear 64→10: logity klas
```

Nie powielamy kanału szarości do RGB. Klasyfikator dostaje surowy embedding;
straty relacyjne i metryki używają normalizacji L2. Na wejściu stałe mean=0,5,
std=0,5; nie dopasowujemy normalizacji do walidacji/testu.

Słaby widok: mały obrót/przesunięcie/zmiana skali (5°, 4%, skala 0,95–1,05).
Mocny: 15°, 12%, skala 0,85–1,1. Fashion-MNIST dopuszcza odbicie poziome;
obsługiwany w tym samym kodzie MNIST go wyłącza. Nie przenosimy operacji
kolorystycznych ani odwracania kolorów z CIFAR. Te augmentacje są naszym
ustawieniem pilota, nie wierną reprodukcją referencyjnego FixMatch/SoftMatch.

## Porównywalność

Każda z pięciu metod: 20 epok referencyjnych po 100 aktualizacji, czyli 2000
kroków SGD, lr=0,03, momentum=0,9, weight decay=0,0005, cosine, ostatni checkpoint.
Epoka referencyjna **nie oznacza przejścia przez całą pulę U**. Obrazy z L
są wielokrotnie prezentowane; U odwiedzamy w losowych minibatchach.

Batch L=64 (dwa słabe widoki), batch U=64 (słaby i mocny). `drop_last=True`
zapewnia zawsze pełny batch L, także w najmniejszym budżecie. Odrzucona reszta
nie jest usuwana z puli: ponowne tasowanie przy następnym obiegu daje jej
kolejną szansę. Metody mają po **256 000 widoków L**; SSL dodatkowo po
**256 000 widoków U**. CE pomija U. Równy nadzór nie oznacza równych FLOPs.

Metody w parze mają identyczny podział, początkowe wagi, losowanie L i jego
augmentacje. RNG dla U i doboru relacji są oddzielone. Kolejka sprawdza hash
inicjalizacji i rzeczywiste liczniki. Trójki B zachowują margines 0,2, wagi
1/30, start części pseudo od epoki 4 i limit/mianownik 128 anchorów.
Contrastive: waga 0,05, temperatura 0,2, pseudo-próg 0,95, start od epoki 4.
Miękkie PL: cel T=0,5, gaussowskie wagi z poprzedniej epoki; pierwsza epoka
ma twardy próg 0,95. Pseudoetykiety nie zużywają budżetu adnotacji.

Pełny nadzór ma 10 przejść referencyjnych po 922 kroki (9220 aktualizacji)
i 59 000 etykiet. Jest osobnym odniesieniem do reprezentacji i klasyfikacji,
**nie porównaniem o równym koszcie ani gwarantowanym górnym ograniczeniem**.
Nie wybieramy checkpointów ani parametrów po oficjalnym teście.

## Reprezentacje i wykresy

Po każdym treningu: dokładność walidacyjna/testowa, liniowy probe uczony
wyłącznie na danym L, kNN15, czystość 10 sąsiadów, silhouette w oryginalnym
embeddingu i t-SNE tych samych 1000 obrazów walidacyjnych. Nie rysujemy PCA.
Parametry t-SNE i preprocessing pozostają takie jak w audycie CIFAR-10.

Dla pełnego nadzoru probe powstaje osobno na tych samych 100, 300 i 500
etykietach co porównywane modele. Encoder pełnego nadzoru znał wszystkie 59 000
etykiet — to jawnie uprzywilejowane odniesienie diagnostyczne.

W `outputs/fashion_mnist_ssl/plots/` automatycznie powstają wykresy
accuracy/probe/purity względem budżetu i t-SNE dla każdego budżetu. t-SNE ma
niezależne współrzędne między panelami; prawdziwe klasy walidacji służą
wyłącznie diagnozie i kolorom. Nie wykonujemy analizy ukrytych klas U.

## Uruchomienie i wznowienie

```powershell
python scripts/run_gray_ssl.py --print-plan
python scripts/run_gray_ssl.py --benchmark
python -u scripts/run_gray_ssl.py --device cpu
```

Ostatnia komenda wznawia przerwany trening od ostatniej ukończonej epoki,
pomija zgodne, ukończone treningi i wykorzystuje zapisane audyty. Blokada
zapobiega równoległemu uruchomieniu tej samej kolejki. Nie zmieniamy planu w
trakcie serii. Inny plan wymaga nowego katalogu wyników.

Stan: `queue.json`; wyniki: `comparison.json/csv`; pomiar czasu: `benchmark.json`;
checkpointy, liczniki, podziały i historie: podkatalogi runów. Stan „completed”
oznacza ukończenie rzeczywistych treningów i audytów, nie sam wpis w planie.
Osobne testy używają wyłącznie tymczasowych fixture’ów, poza wynikami badań.

Pobranie zbioru: oficjalny mirror autorów
<https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion>,
sumy MD5 sprawdzane przez torchvision. Kod obsługuje też MNIST, lecz ten plan
uruchamia wyłącznie Fashion-MNIST. Rozszerzenie na MNIST lub kolejne seedy
powinno zachować ten sam protokół i mieć osobny katalog.
