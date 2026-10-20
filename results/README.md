# Migawka wyników do przekazania

74 rzeczywiste, ukończone treningi; bez testów syntetycznych i manifestów planów.
Interpretacja i najważniejsze tabele: [docs/RESULTS.md](../docs/RESULTS.md).

- `cifar10/`: porównanie 55 głównych runów i katalog 58 runów z wcześniejszym supervised.
- `fashion_mnist/`: porównanie 16 treningów.
- `runs.json`: spis wszystkich 74 pozycji i ich pochodzenie.
- `runs/`: konfiguracje, metryki, podziały i historie tych treningów.
- `representations/`, `analyses/`: zapisane metryki audytów, także probe i nieliniowych głowic.
- `figures/`: wybrane wykresy, w tym t-SNE i zależność od budżetu.
- `manifest.json`: sumy kontrolne eksportu i źródeł kodu.
- `environment.json`: wersje bibliotek w momencie eksportu.

Weryfikacja po sklonowaniu, bez `data/` i `outputs/`:

```powershell
python scripts/export_handoff.py --check
```

Odświeżenie tej migawki z pełnych lokalnych artefaktów: `python scripts/export_handoff.py`.
Eksport nie uruchamia treningów, nie kopiuje wag ani danych i nie kasuje lokalnych wyników.
Gotowy raport PDF znajduje się w [reports/](../reports/Raport_badan_CIFAR10_2026-10-20.pdf).
