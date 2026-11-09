# Zasady interpretacji ukończonych pilotów

Bieżące wyniki: [RESULTS.md](../RESULTS.md). Ten dokument opisuje ograniczenia
porównań, bez listy przyszłych eksperymentów i propozycji publikacji.

## Co porównują piloty

- Stały budżet etykiet i ekspozycja L: te same oznaczone obrazy, początkowe
  wagi, augmentacje L, architektura, optymalizator i liczba aktualizacji
  w obrębie sparowanych wariantów. SSL wykorzystuje dodatkowe obrazy U.
- Równa ekspozycja L nie oznacza równego kosztu: U, nauczyciel, dekoder
  i selekcja AL dodają obliczenia. Liczniki widoków i czas są osobnymi miarami.
- Wspólny BatchNorm korzysta z L i U. Kontrola `ce_2v_u_control` sprawdza
  wpływ samych forwardów U; nie należy przypisywać go wyłącznie nowej stracie.
- Dłuższe harmonogramy 20/40/80 epok są osobnymi treningami od początku,
  z innym horyzontem schedulera, a nie przedłużeniem tego samego runu.
- W AL strategie porównujemy z losowym pozyskiwaniem przy tym samym
  harmonogramie. Stałe 500 etykiet od początku ma inny dostęp do informacji.

## Ograniczenia wniosków

- Trzy seedy sprawdzono dla CE/hard PL/soft PL na CIFAR-10.
  Większość rozszerzeń i cała seria Fashion-MNIST mają tylko seed=0.
  Niewielkie różnice nie potwierdzają stabilnej przewagi.
- Dodatkowe 1000 etykiet walidacyjnych jest osobnym zasobem.
  Walidacja i test są wyłączone z treningu oraz puli zapytań AL.
- Probe uczy się wyłącznie na L. Etykiety walidacji służą ocenie geometrii
  i kolorowaniu wykresów. t-SNE nie zastępuje pomiaru w oryginalnym embeddingu.
- Pełny nadzór korzysta ze znacznie większej liczby etykiet i innego kosztu;
  jest odniesieniem diagnostycznym, nie ścisłym górnym ograniczeniem.
- Implementacje są pilotami inspirowanymi znanymi mechanizmami;
  nie stanowią wiernej reprodukcji FixMatch/SoftMatch ani porównania z SOTA.
- Rekonstrukcja manuskryptu nie potwierdza jego oryginalnych wyników.
  Siatka 5760 konfiguracji nie została wykonana.
- Obecne eksperymenty dotyczą SSL. Nie oceniają adaptacji do nowych klas
  w protokole few-shot i nie dowodzą nowości naukowej połączenia metod.
