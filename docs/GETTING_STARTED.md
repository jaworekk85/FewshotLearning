# Start dla doktoranta

Repo zawiera kod, wyniki 74 ukończonych treningów i raport CIFAR-10. Nie trzeba
uruchamiać eksperymentów, żeby przeczytać wyniki w [results](../results/).

## Instalacja

Python 3.10 lub nowszy. W katalogu repo:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[analysis]"
python -m unittest discover -s tests
```

Na Linux/macOS aktywacja: `source .venv/bin/activate`. Polecenia `python scripts/...`
są wspólne. Instalacja z `[analysis]` dodaje scikit-learn do audytów reprezentacji
i PyMuPDF do raportu. PyTorch musi obsługiwać CUDA, jeśli wybierasz `--device cuda`;
lokalne wyniki były liczone na CPU. Sprawdzenie:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Najpierw sprawdź przepływ, bez pobierania danych

```powershell
python scripts/run_supervised.py --config configs/cifar10_pilot_ssl.json --method ce_2v_pseudo_soft --smoke --device cpu --run-name smoke_soft
```

To krótki test na syntetycznych obrazach, **nie wynik badawczy**. Nie nadpisuje
zakończonych treningów. Ponowne wykonanie wymaga nowej nazwy lub usunięcia
wyłącznie własnego katalogu testowego.

## Pierwsza rzeczywista seria: Fashion-MNIST

```powershell
python scripts/download_data.py fashion_mnist
python scripts/run_gray_ssl.py --print-plan
python scripts/run_gray_ssl.py --benchmark
python scripts/run_gray_ssl.py --device cpu
```

Plan: pięć metod × trzy budżety i pełny nadzór, łącznie 16 treningów.
Zmierzony czas dotychczasowej serii wynosił około 1,4 godziny samych treningów
na lokalnym CPU; wynik zależy od sprzętu. `--max-jobs 1` ogranicza liczbę nowych
treningów przy pierwszym uruchomieniu. Ta sama komenda wznowi przerwaną kolejkę.
Szczegóły: [FASHION_MNIST_SSL.md](FASHION_MNIST_SSL.md).

## Jeden wariant CIFAR-10

```powershell
python scripts/download_data.py cifar10
python scripts/run_supervised.py --config configs/cifar10_pilot_ssl.json --method ce_2v_pseudo_soft --budget 50 --seed 0 --device cpu --run-name cifar10_soft_N50_seed0
python scripts/analyze_representations.py outputs/supervised/cifar10_soft_N50_seed0 --output-dir outputs/representations
```

Wznowienie treningu: dodaj `--resume` i zachowaj konfigurację/nazwę. Nie zmieniaj
horyzontu schedulera w istniejącym runie. Walidacja i oficjalny test są osobnymi
zbiorami; mały budżet dotyczy wyłącznie etykiet treningowych.

## Gdzie trafiają pliki?

- `data/`: pobrane dane, pomijane przez Git.
- `outputs/`: nowe wyniki, checkpointy i logi, pomijane przez Git.
- `results/`: wersjonowana, mała migawka wyników do przekazania i analizy.
- `reports/`: gotowy raport PDF.

Kopia repo z GitHuba nie zawiera wag. Do wznowienia istniejących lokalnych runów
lub ponownego wyliczenia embeddingów potrzebna jest osobna kopia `outputs/`.
Pełny eksport zawiera konfiguracje, podziały i historie, więc nowe treningi
można odtworzyć od początku. Nie gwarantujemy identyczności bitowej między
wersjami PyTorch, CPU i GPU.
