> Historical lab note. Status/ETA below may be outdated; see [current results](../RESULTS.md).

# Diagnostyka wiarygodności pseudoetykiet

Analiza końcowych modeli, bez treningu, nowych etykiet i odczytu zbioru testowego.
Plan `configs/reliability_audit.json` obejmuje 15 modeli: CE, twarde pseudoetykiety,
miękkie wagi i cele, trójki mieszane, dłuższe treningi, dodatkowe seedy oraz
random/BADGE online i z ich końcowym zbiorem etykiet dostępnym od początku.

```powershell
.\.venv-analysis\Scripts\python.exe scripts/analyze_reliability.py --plan configs/reliability_audit.json
.\.venv-analysis\Scripts\python.exe scripts/summarize_reliability.py
```

Wyniki w `outputs/reliability/report.md`, `comparison.json` i katalogu każdego modelu.
`confident_errors.md/.json` zawierają dodatkowo wykrywanie błędów wśród predykcji
z pewnością co najmniej 0,95 oraz macierze pomyłek.

Mierzymy accuracy, NLL, Brier (suma po klasach), ECE (15 równych przedziałów),
pokrycie i błędy przy ustalonych progach pewności. Powtarzamy pomiary na jednym
ustalonym słabo augmentowanym widoku każdego obrazu walidacyjnego.
Wagi są hipotetyczne: końcowy model w trybie eval i reguła przewidziana
na kolejną epokę, z parametrami zapisanymi w ostatnim wierszu historii.
Nie są to rzeczywiste historyczne wagi próbek podczas treningu z joint BatchNorm.
Udział błędnych argmax w sumie wag oraz masa miękkiego celu przypisana
nieprawdziwym klasom opisują potencjalny problem, nie mierzą jego wpływu na gradient.
Większy udział błędów po zmianie pokrycia nie dowodzi gorszej metody.

Bez znajomości ocenianych etykiet wyliczamy niepewność, entropię, JS względem
sąsiadów z prawdziwie oznaczonego zbioru treningowego (15 sąsiadów,
cosine/temperatura 0,2) oraz JS pomiędzy czterema słabymi augmentacjami.
Ich zdolność wykrywania błędów mierzymy AUROC, average precision i precyzją
w górnych 5/10/20% rankingu; odniesienie to losowanie z tego samego zbioru.
Nie dopasowujemy progów ani kalibratora. Nie używamy ukrytych etykiet U.

Wizualizacja obejmuje kalibrację, rozkład pewności błędnych/poprawnych predykcji
i PCA reprezentacji z zaznaczonymi błędami. Lokalną koncentrację błędów mierzymy
w oryginalnej reprezentacji, wśród 10 innych obrazów walidacyjnych.
Predykcje, reprezentacje, indeksy, wagi i wyniki rankingów są zachowane w NPZ.
Zmiana źródłowego checkpointu, konfiguracji, podziału lub historii unieważnia cache.

To analiza eksploracyjna na walidacji: wzbogacenie rankingu w błędy nie dowodzi,
że ich zaetykietowanie poprawi uczenie. Efekt nowej metody wymaga osobnego
eksperymentu z tym samym budżetem, harmonogramem oraz kontrolą losową.
Modele 20/40/80 epok pochodzą z oddzielnych treningów z innym horyzontem cosine.
