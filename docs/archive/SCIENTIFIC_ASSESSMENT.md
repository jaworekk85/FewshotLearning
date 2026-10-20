> Historical lab note. Status/ETA below may be outdated; see [current results](../RESULTS.md).

# Ocena porównań i potencjalnego wkładu naukowego

Ocena obecnego pilota, nie deklaracja gotowości rozprawy ani nowości metody.
Bieżące treningi pozostają uruchomione z wcześniej ustalonym protokołem.

## Co można uczciwie porównywać

1. **Stały budżet anotacji i ekspozycja L.** Te same oznaczone obrazy,
   początkowe wagi, batch L i augmentacje, architektura, CE, optymalizator,
   harmonogram i liczba aktualizacji. SSL ma dodatkową informację U i większy
   koszt. Wniosek dotyczy skuteczności przy stałym budżecie etykiet, a nie
   równej ilości dostępnych obrazów czy automatycznie lepszej funkcji straty.
2. **Stały koszt.** Osobne porównanie przy zbliżonym koszcie obliczeniowym:
   dłuższy CE oraz accuracy względem skumulowanych widoków, czasu i ewentualnie
   FLOPs. Nie ma jednej liczby epok wyrównującej jednocześnie etykiety,
   ekspozycję L/U, aktualizacje i koszt. Harmonogramy trzeba jawnie raportować.

Obecny wspólny pilot kontroluje pierwszy scenariusz. Utrzymuje osobne RNG
samplingu i augmentacji oraz liczniki rzeczywistych wejść do encodera.
Porównanie `ce_2v_triplet_u` z `ce_2v_triplet_lu` dodatkowo utrzymuje identyczne
widoki L/U; relacje L korzystają z obliczonych embeddingów.

## Co jest gotowe, ale nie daje jeszcze końcowego dowodu

- CE z jednym widokiem: zakończone N=10 i N=50, pojedynczy seed.
- CE z dwoma widokami: uruchomione w nowym protokole.
- Pseudoetykiety i trzy warianty trójek: zaimplementowane i w kolejce.
- `ce_2v_u_control`: zaimplementowana kontrola wpływu BatchNorm i forwardów U,
  jeszcze niewłączona do bieżącej kolejki. Jest potrzebna przy przypisywaniu
  zysku samej dodatkowej stracie, bo wspólny BatchNorm wykorzystuje U.
- Dłuższy CE: runner umożliwia zmianę harmonogramu; odpowiedniego kontrolnego
  runu nie wykonano. Zapisane liczniki pomogą dobrać jego budżet widoków.
- Pseudoetykietowanie to mały wariant inspirowany FixMatch; nie jest
  reprodukcją benchmarkowych wyników FixMatch ani porównaniem z aktualnym SOTA.

## Braki przed silnymi wnioskami publikacyjnymi

- Ustalić pytanie i główne porównania przed przeglądaniem końcowych wyników.
- Zamrozić sposób wyboru checkpointu i podobny budżet strojenia metod.
  Wagi strat i progi nie mogą być dobierane na oficjalnym teście.
- Raportować dodatkowe 1000 etykiet walidacyjnych obok N*K etykiet treningowych.
- Sprawdzić krzywe/zbieżność każdej metody, nie tylko wynik po stałym limicie.
- Potwierdzić wyniki w kilku sparowanych losowaniach, z CI dla różnic.
  Pojedynczy seed daje screening, nie miarę stabilnej przewagi.
- Potwierdzić efekty na kolejnych datasetach i większej architekturze.
- Uwzględnić mocne, poprawnie odtworzone baseline'y SSL, np. FixMatch/FreeMatch,
  oraz pokrewne metody reprezentacji, np. SimMatch, stosownie do treści tezy.
- Odtworzyć wcześniej obiecujący VAE na tych samych podziałach i z rozliczeniem
  kosztu pretreningu; nie odrzucać go wyłącznie z powodu zmiany kierunku prac.
- Ocenić reprezentacje k-NN i linear probe oraz miarami klasowymi, obok
  wizualizacji. Etykiety do diagnostyki nie mogą niejawnie sterować treningiem.

## Co mogłoby być wkładem doktorskim

Obecne repo jest infrastrukturą badawczą i pilotażem znanych mechanizmów.
Samo porównanie CE, trójek, pseudoetykiet i VAE na CIFAR-10 nie wykazuje
jeszcze oryginalnego wkładu. Wiele ich połączeń ma już precedensy.

Przykładowe wąskie pytanie do weryfikacji nowości:

> Jak wybierać i ważyć relacje metryczne przy bardzo małym budżecie etykiet,
> kiedy przewidywania dla U są niepewne i mogą tworzyć błędne relacje klasowe?

Kandydatem byłby konkretny mechanizm doboru/ważenia relacji, wraz z hipotezą,
ablacjami i dowodem, w jakich warunkach działa lepiej lub zawodzi.
Taki mechanizm wymaga przeglądu literatury przed nazwaniem go nowym.
Możliwy jest też silny wkład metodologiczny, jeżeli ujawnia nieznany wcześniej,
istotny i odtwarzalny mechanizm błędnej oceny metod.

Active learning może rozszerzyć ten sam problem: które etykiety najbardziej
ograniczają niepewność relacji? FSL wymaga protokołu adaptacji do nowych klas.
Łączenie tych tematów samo nie ustanawia nowości; spójność ma wynikać ze
wspólnego pytania i mechanizmu, a nie liczby modułów w repo.

Najbliższy kamień milowy: zakończyć screening, ocenić różnice względem
kontroli, wybrać jeden mechanizm wart dalszej weryfikacji. Nie obiecywać
z góry liczby publikacji ani przyjęcia pracy na podstawie obecnego pilota.

## Źródła

- [Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://arxiv.org/abs/1804.09170)
- [SimMatch: Semi-supervised Learning with Similarity Matching](https://arxiv.org/abs/2203.06915)
- [SoftMatch](https://arxiv.org/abs/2301.10921)
- [Semi-Supervised and Active Few-Shot Learning](https://arxiv.org/abs/1711.10856)
