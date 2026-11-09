# FewshotLearning — badania uczenia półnadzorowanego

Eksperymenty na CIFAR-10 i Fashion-MNIST: jak wykorzystać obrazy bez etykiet,
gdy znamy klasy tylko niewielkiej części zbioru? Kod obejmuje CE,
pseudoetykiety twarde/miękkie, trójki, contrastive i active learning.

**Dla doktoranta:** zacznij od [instrukcji uruchomienia](docs/GETTING_STARTED.md),
[wyników](docs/RESULTS.md) i [mapy kodu](docs/CODE_MAP.md).

## Stan badań

74 ukończone treningi: **58 CIFAR-10 + 16 Fashion-MNIST**. Zapisane konfiguracje,
metryki, podziały i historie są w [results/](results/README.md).

| Zbiór | Mały budżet | CE | Najważniejsze odniesienie |
|---|---|---:|---|
| CIFAR-10 | 50 etykiet/klasę | 49,25% | Miękkie PL20: 60,12%; PL80: 65,34% przy 4× większym horyzoncie |
| Fashion-MNIST | 10 etykiet/klasę | 71,33% | Miękkie PL + contrastive: 76,63% |
| Fashion-MNIST | 50 etykiet/klasę | 81,94% | Miękkie PL + trójki: 83,42% |

To dokładności testowe dla seed0. Trzy seedy sprawdzono dla CE/hard PL/soft PL
na CIFAR-10; większość pozostałych wyników jest pilotażowa. Pełny nadzór:
CIFAR-10 79,75%, Fashion-MNIST 89,87%, przy znacznie większym budżecie etykiet.

[Raport CIFAR-10: metody, wykresy, reprezentacje i AL — PDF](reports/Raport_badan_CIFAR10_2026-10-20.pdf)

## Szybki start

Python 3.10+. Uruchamiaj polecenia w katalogu repo, we własnym środowisku:

```powershell
python -m pip install -e ".[analysis]"
python -m unittest discover -s tests
python scripts/download_data.py fashion_mnist
python scripts/run_gray_ssl.py --print-plan
```

Seria 16 treningów Fashion-MNIST jest ukończona, a wyniki są w repo.
Powyższy podgląd pokazuje jej konfigurację. Odtworzenie we własnym środowisku:
`python scripts/run_gray_ssl.py --device cpu`. Dane i nowe wyniki zostają lokalnie.
Pełny protokół: [Fashion-MNIST](docs/FASHION_MNIST_SSL.md).

## Organizacja

| Katalog | Zawartość |
|---|---|
| `fewshot_metric/` | Wspólna biblioteka modeli, strat, treningu i analiz |
| `scripts/` | Komendy do treningu, audytów i eksportu; [przewodnik](docs/CODE_MAP.md) |
| `configs/` | Zapisane protokoły i warianty |
| `tests/` | Testy na małych fixture’ach, bez pobierania zbiorów |
| `results/` | Mała, wersjonowana migawka rzeczywistych wyników i wykresów |
| `reports/` | Gotowy raport dla doktoranta |
| `docs/` | Instrukcje i interpretacja; [archiwum](docs/archive/README.md) opisuje wykonane protokoły |

`data/`, `outputs/`, środowiska Pythona i checkpointy nie trafiają do Gita.
Pełne lokalne artefakty pozostały zachowane. Klon zawiera wyniki do analizy,
ale do wznowienia dawnych treningów trzeba osobno przekazać wagi albo odtworzyć
trening od początku. Nie ma konieczności używania Git LFS do tej migawki.

## Ważne rozróżnienie

Nazwa repo pochodzi od wcześniejszego kierunku badań. Obecne piloty to **SSL
przy małym budżecie etykiet**, nie test adaptacji do nowych klas. Starszy
protokół `v4_protocol.json` rekonstruuje opis manuskryptu. Siatka 5760 konfiguracji
nie została wykonana i nie jest kolejką do uruchomienia.
Oryginalne wyniki doktoranta nie były dostępne.
Manuskrypt `access_v4.pdf` pozostaje plikiem lokalnym, poza publicznym repo.

Repo nie zawiera obecnie zatwierdzonej listy kolejnych eksperymentów.
Konfiguracje ukończonych serii służą odtwarzaniu wyników; nie oznaczają zlecenia
ponownego treningu ani rozszerzania siatki.

Nie publikujemy tu deklaracji nowej metody ani istotności niewielkich różnic
z pojedynczego seedu. [Wyniki i ograniczenia](docs/RESULTS.md).
