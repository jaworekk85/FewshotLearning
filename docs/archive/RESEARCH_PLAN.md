> Historical lab note. Status/ETA below may be outdated; see [current results](../RESULTS.md).

# Plan badań i pierwszy eksperyment

Uzgodniony kierunek, 15 października 2026: klasyfikacja półnadzorowana przy
budżecie N prawdziwych etykiet na klasę, z późniejszym active learning i FSL.
Najpierw potwierdzamy działający trening supervised na CIFAR-10.

Aktualizacja: szybki wariant rozwojowy używa najpierw istniejącego CNN ze
111 498 parametrami i 20 epokami (`PILOT.md`). Kontrola ResNet-18 pozostaje
kolejnym etapem potwierdzenia; nie jest warunkiem rozpoczęcia taniego pilotażu.

Pierwsza kontrola CNN zakończona: **79.75% na oficjalnym teście CIFAR-10**,
seed 0, 49 000 etykiet treningowych, 20 epok, **17 min 16 s na CPU**.
Artefakty: `outputs/supervised/cnn_pilot_full_seed0/`.
Zakończone baseline'y małych budżetów, ten sam seed=0 i 7660 aktualizacji:
N=10: **29.77%** test accuracy, N=50: **49.62%**. Szczegóły w `PILOT.md`.
Nowy pilot z dwoma widokami L i wariantami SSL jest uruchomiony dla N=50,
seed=0; stan jest zapisywany w `outputs/pilot/queue_N50_seed0*.json`.
Ocenę ograniczeń porównań i potrzebnego wkładu opisuje `SCIENTIFIC_ASSESSMENT.md`.

## Pochodzenie obecnego repo

Kod odtworzono na podstawie `access_v4.pdf`, a nie oryginalnego repo doktoranta.
Autorzy mają pozyskać oryginalne wyniki i kod od wykonawcy eksperymentów.
Lokalnie nie ma per-run danych potwierdzających liczby z artykułu.
Manifesty 5760 konfiguracji to plan uruchomień w rekonstrukcji.
Nie przypisywać tej implementacji zgodności liczbowej z oryginalnym badaniem.

## Kolejność eksperymentów

1. **Kontrola pełnonadzorowana.** ResNet-18 dla CIFAR-10, pełna dostępna
   pula treningowa, krzywe, checkpointy, jawny budżet walidacyjny.
   Roboczy próg >90% accuracy; wynik wymaga pomiaru. Instrukcja:
   `SUPERVISED_BASELINE.md`.
2. **Mały budżet supervised.** Ta sama architektura, niezależne losowe
   inicjalizacje, np. N=10 i N=50, trzy sparowane seedy w pilotażu.
   Wagi kontroli z pełnymi etykietami nie mogą inicjalizować tych eksperymentów.
3. **SSL z tripletami.** CE na prawdziwych etykietach oraz osobno ważone
   straty metryczne na L i U. Dla U pozytyw to inny widok tego samego obrazu;
   losowy inny obraz nie jest gwarantowanym negatywem klasowym.
   Pierwsza ablacja: CE; CE+triplet_L; CE+triplet_U; CE+oba.
4. **Pseudoetykietowanie.** Odtworzyć FixMatch jako mocny punkt odniesienia:
   weak/strong views, próg, dokładne skalowanie maskowanej straty, proporcja
   batchy L/U i pełny harmonogram. Sam obecny `pseudo_ce` nie jest wierną
   implementacją FixMatch. Dopiero potem testować dodanie naszej metryki.
5. **Odtworzenie VAE.** Wrócić do wcześniejszego obiecującego wyniku:
   pretrening na dozwolonej puli treningowej bez etykiet, następnie
   klasyfikator na N*K etykietach. Porównać zamrożony encoder i fine-tuning.
   Oddzielić koszt pretreningu od dostrajania; użyć tych samych podziałów
   i zgodnej pojemności encodera co pozostałe metody. Rekonstrukcja obrazów
   nie gwarantuje reprezentacji klasowej; wcześniejsza obserwacja wymaga replikacji.
6. **Rozszerzenie wyników.** Dopiero po pilotażu zamrozić metody i protokół,
   zwiększyć liczbę seedów i budżetów, dodać drugi/trzeci dataset.
7. **Active learning.** Stały seed oznaczony, kolejne rundy zapytań,
   random acquisition jako obowiązkowy kontrolny sposób wyboru.
   Ten sam trainer, pula U, seedy startowe i harmonogram dla różnych selekcji.
8. **Few-shot.** Przenieść wybrane mechanizmy na protokół z rozłącznymi
   klasami bazowymi i nowymi, np. 1/5-shot; oddzielna ewaluacja adaptacji.

Nie uruchamiać całej siatki wariantów przed zakończeniem pilotażu.
Nowy runner obsługuje obecnie punkt 1 i kontrolę z punktu 2.
Wspólny pilot SSL obsługuje już dwa widoki L, trójki L/U oraz
pseudoetykietowanie weak/strong (`PILOT_SSL.md`). Dodano kontrolę ekspozycji
L/U i oddzielne RNG. Kolejna seria (`PILOT_NEXT.md`) dodaje audyt reprezentacji,
dobór semi-hard, miękkie pseudoetykietowanie i trzy strategie active learning;
kolejka czeka na CE40. VAE pozostaje poza obecną implementacją.

## Informacja dostępna metodom

- L zawiera dokładnie N*K ujawnionych etykiet treningowych; U to pozostałe
  dozwolone obrazy oficjalnego train po usunięciu walidacji.
- Pseudoetykiety są przewidywaniami modelu i nie zwiększają budżetu anotacji.
- Etykiety U pozostają niedostępne podczas optymalizacji i wyboru przykładów.
- Nowe prawdziwe etykiety pozyskane przez active learning zwiększają licznik
  budżetu. Nie można zaglądać w etykiety, żeby zachować idealny balans klas.
- Dodatkowe etykiety walidacyjne oraz etykiety użyte do strojenia raportujemy
  osobno. Podziały testowe nie wchodzą do treningu ani puli zapytań.
- Kontrola pełnonadzorowana jest punktem odniesienia z bogatszą informacją,
  a nie gwarantowanym matematycznym ograniczeniem accuracy innych metod.

## Raportowanie

Accuracy względem łącznej liczby ujawnionych etykiet; dla stałych podzbiorów
również N na klasę. W active learning końcowe klasy mogą mieć różne liczności.
Średnie i odchylenia oraz sparowane różnice z CI osobno dla każdego datasetu.
Wyniki per-seed, loss, krzywe, udział zaakceptowanych pseudoetykiet,
rozkład przewidywanych klas, aktualizacje, rzeczywiste przetworzone widoki,
czas GPU i koszt pretreningu. Równa liczba kroków nie oznacza równego compute.
Nie zakładać z góry przewagi ani określonego wzrostu accuracy.

## Ocena access_v4.pdf

PDF bada **supervised metric learning w ustawieniu label-only**. Jego teza:
triplet wygrywa ze słabym CE, ale przewaga znika po wzmocnieniu CE
przez augmentacje, wiele widoków, normalizację i dłuższy trening.
To może być odrębne badanie metodologiczne, niezależne od późniejszego SSL.

Przed decyzją o wysłaniu:

- pozyskać oryginalne per-run wyniki, kod, podziały i konfiguracje;
  niezależnie przeliczyć tabele i przedziały, sprawdzić liczbę unikatowych runów;
- podać dokładne architektury, marginesy, sampling i regułę wyboru modelu;
  obecny PDF nie wystarcza do odtworzenia wszystkich tych szczegółów;
- rozdzielić budżet treningowy i dodatkowe 100 etykiet walidacyjnych na klasę;
- sprawdzić, czy CE i triplet mają porównywalne rodzaje augmentacji;
  PDF opisuje mocniejsze augmentacje dla metryki niż dla CE;
- sprawdzić zbieżność i dać tripletowi również dłuższy harmonogram:
  v4 wydłuża CE, ale nie powtarza treningu modeli metrycznych;
- zachować wyniki per-dataset, nie opierać tezy wyłącznie na pooled mean;
- uzasadnić wkład względem istniejących badań nad mocnymi baseline'ami.

Sama obecność PDF-u nie potwierdza wyników. Rekonstrukcja kodu nie może
zastąpić oryginalnych artefaktów przy sprawdzaniu deklaracji historycznych.
Wartość publikacyjną i dopasowanie do IEEE Access trzeba ocenić po audycie.
IEEE Access wymaga wyraźnego wkładu: zob.
[Stages of Peer Review](https://ieeeaccess.ieee.org/authors/stages-of-peer-review/).

## Potencjalne osobne publikacje

Możliwe są trzy odrębne pytania, pod warunkiem uzyskania samodzielnych wyników:

1. Kontrole baseline'ów supervised metric learning — obecny PDF po audycie.
2. Wykorzystanie L i U do uczenia reprezentacji — triplet, pseudoetykiety,
   VAE jako punkt odniesienia i mocna baza SSL.
3. Wybór etykiet na podstawie reprezentacji — active learning względem random
   i uznanych strategii. FSL może być rozszerzeniem wybranej pracy.

To mapa możliwych wkładów, nie obietnica trzech artykułów. VAE nie jest
automatycznie osobną publikacją; najpierw ma rozstrzygnąć porównanie metod.

## Podstawowe źródła

- [ResNet-18 na CIFAR-10: przepis porównawczy](https://github.com/kuangliu/pytorch-cifar)
- [FixMatch](https://arxiv.org/abs/2001.07685)
- [SimMatch](https://arxiv.org/abs/2203.06915)
- [Semi-supervised Learning with Deep Generative Models](https://arxiv.org/abs/1406.5298)
- [BADGE](https://arxiv.org/abs/1906.03671)
- [Semi-Supervised and Active Few-Shot Learning](https://arxiv.org/abs/1711.10856)
