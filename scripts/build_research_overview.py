"""Readable Polish research report, rebuilt from saved results without training."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import research_report_support as report
import matplotlib.pyplot as plt
import numpy as np

ROOT, OUT = report.ROOT, report.OUT
PREFIX = "cnn_N50_seed0_"
CE, HARD, SOFT = [PREFIX+s for s in ("ce_2v", "ce_2v_pseudo", "ce_2v_pseudo_soft")]
TRIP = PREFIX+"mined_mixed_embedding"
CON = PREFIX+"contrastive_with_pseudo_pairs"
REFERENCES = report.REFERENCES.copy()
REF_IDS = [1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 16, 17, 13, 14, 18, 19]


def charts(data):
    assets = OUT/"assets"
    assets.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
        "axes.spines.top":False, "axes.spines.right":False, "savefig.facecolor":"white"})
    rows = {r["run"]:r for r in data["rows"]}

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(assets/name, dpi=200)
        plt.close(fig)

    # Match ConvEncoder in fewshot_metric/models.py, including its linear embedding.
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig,ax=plt.subplots(figsize=(7.3,2.65))
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_axis_off()
    nodes=[
        (.015,.57,"1. Obraz wejściowy","RGB: 3 × 32 × 32","0 parametrów"),
        (.35,.57,"2. Blok splotowy 1","Conv 3→32 + BN + ReLU\nMaxPool 2×2 → 32 × 16 × 16","960 parametrów"),
        (.685,.57,"3. Blok splotowy 2","Conv 32→64 + BN + ReLU\nMaxPool 2×2 → 64 × 8 × 8","18 624 parametry"),
        (.685,.065,"4. Blok splotowy 3","Conv 64→128 + BN + ReLU\nGAP → 128 × 1 × 1","74 112 parametrów"),
        (.35,.065,"5. Embedding","Spłaszczenie → Linear 128→128\nWektor z: 128 liczb","16 512 parametrów"),
        (.015,.065,"6. Klasyfikator","Linear 128→10 → 10 logitów\nsoftmax → prawdopodobieństwa","1290 parametrów"),
    ]
    width,height=.30,.36
    for i,(x,y,title,body,count) in enumerate(nodes):
        ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle="round,pad=0.006,rounding_size=.015",
            edgecolor="#16858b" if i in (0,4,5) else "#91aab5",facecolor="#eaf4f3" if i in (4,5) else "#f0f5f7",linewidth=1))
        ax.text(x+width/2,y+.293,title,ha="center",va="center",fontsize=9,fontweight="bold",color="#173d50")
        ax.text(x+width/2,y+.185,body,ha="center",va="center",fontsize=7.9,linespacing=1.55,color="#203345")
        ax.text(x+width/2,y+.055,count,ha="center",va="center",fontsize=7.4,color="#526977")
    for start,end in [((.315,.75),(.35,.75)),((.65,.75),(.685,.75)),
                      ((.835,.57),(.835,.425)),((.685,.245),(.65,.245)),((.35,.245),(.315,.245))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=12,color="#147b84",linewidth=1.5))
    fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01)
    fig.savefig(assets/"cnn_architecture.png",dpi=240,bbox_inches="tight",pad_inches=.025)
    fig.savefig(assets/"cnn_architecture.svg",bbox_inches="tight",pad_inches=.025)
    plt.close(fig)
    data["sources"].append(assets/"cnn_architecture.svg")

    def bars(runs, names, name, limits=(40,65), height=3.1):
        fig, ax = plt.subplots(figsize=(7.3,height))
        y=np.arange(len(runs))
        for delta,key,label,color in [(-.18,"val_accuracy","Walidacja","#839ca9"),
                                      (.18,"test_accuracy","Test","#16858b")]:
            vals=[100*rows[r][key] for r in runs]
            rects=ax.barh(y+delta, vals, height=.31, label=label, color=color)
            ax.bar_label(rects, labels=[report.num(v) for v in vals], padding=3, fontsize=8)
        ax.set_yticks(y,names)
        ax.invert_yaxis()
        ax.set_xlim(*limits)
        ax.set_xlabel(f"Dokładność [%]; oś pozioma zaczyna się od {limits[0]}%")
        ax.grid(axis="x",alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(loc="lower center",bbox_to_anchor=(.5,1.01),ncol=2,fontsize=8,frameon=False)
        save(fig,name)

    bars([CE,HARD,SOFT],["Tylko nadzór (CE)","Twarde pseudoetykiety","Miękkie cele i wagi"],"main_results.png")
    bars([SOFT,PREFIX+"mined_lul_embedding",TRIP,PREFIX+"mined_mixed_pixels",PREFIX+"mined_mixed_embedding_masked"],
         ["Soft PL bez trójek","Trójki: L–U–L","Trójki: miks / embedding","Trójki: miks / piksele","Trójki: miks / maska"],
         "triplets.png",(55,65),3.7)
    bars([SOFT,PREFIX+"contrastive_labeled_instance",CON],
         ["Soft PL bez contrastive","Contrastive: etykiety + instancje","Contrastive: także pseudo-pary"],
         "contrastive.png",(55,65))

    fig,ax=plt.subplots(figsize=(7.3,3.1))
    for i,(suffix,title,color) in enumerate([
        ("ce_2v","CE","#839ca9"),("ce_2v_pseudo","Twarde pseudoetykiety","#16858b"),
        ("ce_2v_pseudo_soft","Miękkie cele i wagi","#d48b35")]):
        vals=np.array([100*rows[f"cnn_N50_seed{s}_{suffix}"]["test_accuracy"] for s in range(3)])
        ax.scatter(i+np.array([-.09,0,.09]), vals,s=43,color=color,zorder=3)
        ax.errorbar(i,vals.mean(),yerr=vals.std(ddof=1),fmt="_",markersize=21,capsize=6,color="#183e50")
    ax.set_xticks(range(3),["CE","Twarde pseudoetykiety","Miękkie cele i wagi"])
    ax.set_ylabel("Dokładność testowa [%]"); ax.set_ylim(45,63); ax.grid(axis="y",alpha=.2)
    save(fig,"three_seeds.png")

    fig,ax=plt.subplots(figsize=(7.3,3.5))
    for runs,label,color in [([CE,PREFIX+"ce_2v_40epochs"],"CE","#839ca9"),
        ([SOFT,PREFIX+"ce_2v_pseudo_soft_40epochs",PREFIX+"ce_2v_pseudo_soft_80epochs"],"Miękkie pseudoetykiety","#16858b")]:
        xs=[rows[r]["epochs_completed"] for r in runs]; ys=[100*rows[r]["test_accuracy"] for r in runs]
        ax.plot(xs,ys,"o-",label=label,color=color)
        for x,y in zip(xs,ys):ax.annotate(report.num(y),(x,y),xytext=(0,7),textcoords="offset points",ha="center",fontsize=8)
    ax.set_ylim(46,69); ax.set_xticks([20,40,80]); ax.set_xlabel("Epoki (383 aktualizacje w każdej)")
    ax.set_ylabel("Dokładność testowa [%]"); ax.grid(alpha=.2); ax.legend(loc="lower right")
    save(fig,"training_horizon.png")

    al_runs=[SOFT,"cnn_N30to500_seed0_soft_active_random","cnn_N30to500_seed0_soft_active_badge",
             "cnn_N500_seed0_soft_frozen_random","cnn_N500_seed0_soft_frozen_badge"]
    bars(al_runs,["Stałe, zbalansowane 500","AL: losowanie","AL: BADGE",
                  "Końcowy wybór losowy od początku","Końcowy wybór BADGE od początku"],
         "active_learning.png",(54,64),3.6)

    selected=[CE,SOFT,PREFIX+"ce_2v_pseudo_soft_80epochs",TRIP,CON,"cnn_pilot_full_seed0"]
    names=["CE, 500 etykiet","Soft PL, 20 epok","Soft PL, 80 epok","Soft PL + trójki B",
           "Soft PL + contrastive/pseudo","Pełny nadzór, 49 000 etykiet"]
    reps={r["run"]:r for r in data["reps"]}
    fig,ax=plt.subplots(figsize=(7.3,3.5)); y=np.arange(len(selected))
    for delta,key,label,color in [(-.18,"original_head_val_accuracy","Głowica sieci","#839ca9"),
                                  (.18,"linear_probe_val_accuracy","Nowa głowica z 500 etykiet","#16858b")]:
        vals=[100*reps[r][key] for r in selected]
        rects=ax.barh(y+delta,vals,height=.31,label=label,color=color)
        ax.bar_label(rects,labels=[report.num(v) for v in vals],padding=3,fontsize=8)
    ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(40,85);ax.set_xlabel("Walidacja [%]; oś zaczyna się od 40%")
    ax.grid(axis="x",alpha=.18);ax.set_axisbelow(True)
    ax.legend(loc="lower center",bbox_to_anchor=(.5,1.01),ncol=2,fontsize=7,frameon=False)
    save(fig,"representation_probe.png")

    fig,axes=plt.subplots(3,2,figsize=(7.5,8.4))
    for ax,run,title in zip(axes.flat,selected,names):
        source=ROOT/"outputs/representation_reference"/run/"embeddings.npz"
        with np.load(source) as a:xy,y=a["tsne"],a["validation_labels"]
        ax.scatter(xy[:,0],xy[:,1],c=y,cmap="tab10",vmin=0,vmax=9,s=4,alpha=.7,linewidths=0)
        ax.set_title(title,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
        data["sources"].append(source)
    from matplotlib.lines import Line2D
    labels=["samolot","auto","ptak","kot","jeleń","pies","żaba","koń","statek","ciężarówka"]
    handles=[Line2D([],[],marker="o",linestyle="none",color=plt.get_cmap("tab10")(i),label=n,markersize=4) for i,n in enumerate(labels)]
    fig.legend(handles=handles,loc="lower center",ncol=5,fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(assets/"validation_tsne.png",dpi=200);plt.close(fig)
    source=ROOT/"outputs/representation_full_pool/cnn_N50_seed0_ce_2v_pseudo_soft_80epochs/tsne_full_pool.png"
    shutil.copyfile(source,assets/"full_pool_soft.png");data["sources"].append(source)
    data["sources"].extend(sorted(assets.glob("*.png")))


def sections(data):
    p,table,note,image=report.p,report.table,report.note,report.image
    rows={r["run"]:r for r in data["rows"]}
    result=[]
    def add(title,body):
        title=title if not result else f"{len(result):02d}. {title}"
        result.append({"title":title,"html":f"<h1>{title}</h1>"+body})
    def rt(runs,titles):return report.result_table(data,runs,titles)
    def h(title,text):return "<h2>"+title+"</h2>"+p(text)
    def brief(text):return '<div class="algorithm"><b>W skrócie:</b> '+text+'</div>'

    add("Co dały nasze eksperymenty?",p("CIFAR-10 • MAŁY BUDŻET ETYKIET • 20.10.2026","kicker")
        +p("Raport dla doktoranta i promotora","lead")
        +p("Badaliśmy, jak wykorzystać obrazy bez etykiet, gdy znamy klasy tylko niewielkiej części zbioru. Poniżej opisujemy, co poszczególne metody robią z danymi, jakie uzyskaliśmy wyniki oraz co zmieniło się w reprezentacjach uczonych przez sieć.")
        +table(["Najważniejszy wynik","Co z niego wynika"],[
            ["CE: 49,25% → miękkie pseudoetykiety: 60,12%","Największy zysk daje wykorzystanie nieetykietowanych obrazów do uczenia klasyfikacji."],
            ["Miękkie pseudoetykiety, 80 epok: 65,34%","Dłuższy trening poprawił wynik mocniej niż kolejne dodatki do metody."],
            ["Miękkie pseudoetykiety + trójki B: 60,96%","Niewielki przyrost w jednym seedzie; brak potwierdzenia powtarzalności."],
            ["Active learning bez wyraźnej przewagi","Przebadane sposoby dobierania etykiet nie pokonały przekonująco losowania."],
            ["Lepsze reprezentacje, ale bez przełomu","Pseudoetykiety poprawiają embeddingi; trójki i contrastive poprawiają część ich miar, lecz niewiele zmieniają klasyfikację."]])
        +p("Liczby w tej tabeli to dokładności testowe. Poza jawnie opisanymi wyjątkami: 500 etykiet, 20 epok, seed0. Wszystkie 58 ukończonych treningów zachowano w załącznikach; część główna skupia się na wynikach potrzebnych do zrozumienia badań.","small")
        +note("To badanie pilotażowe: jeden zbiór i jedna mała CNN; większość dodatków ma tylko jeden seed. Trzy seedy sprawdzono dla CE, twardych i miękkich pseudoetykiet. Raport dokumentuje wyniki, nie stanowi jeszcze dowodu nowej metody."))

    add("Co było wspólne dla porównań?",h("Jakie dane widzi model?",
        "CIFAR-10 zawiera obrazy dziesięciu klas. Z puli treningowej wydzielamy 500 obrazów z etykietami — po 50 na klasę — oraz 48 500 obrazów, których etykiety ukrywamy. Osobne 1000 obrazów służy walidacji. Oficjalne 10 000 obrazów testowych służy ocenie końcowej. Walidacja nie trafia do treningu ani do puli wyboru active learningu.")
        +table(["Termin","Proste znaczenie"],[
            ["CE / supervised","Zwykłe uczenie klasyfikatora na obrazach z prawdziwymi etykietami."],
            ["Pseudoetykieta","Odpowiedź modelu użyta jako tymczasowy cel dla obrazu bez etykiety. Nie zużywa budżetu prawdziwych etykiet."],
            ["Widok / augmentacja","Przekształcona wersja tego samego obrazu. Słaba: losowy kadr i odbicie. Mocna: dodatkowo RandAugment, czyli silniejsze zmiany wyglądu."],
            ["Embedding","128 liczb opisujących obraz, wyliczonych przez CNN przed wyborem klasy."],
            ["Seed","Ziarno losowości. Zmienia m.in. podział danych, początkowe wagi i przebieg treningu."]])
        +h("Ta sama baza techniczna","Mała CNN ConvEncoder: trzy bloki splotowe, embedding 128D i klasyfikator dziesięciu klas. Łącznie <b>111 498 uczonych parametrów</b>. Schemat pokazuje przepływ jednego widoku; zapis wymiarów to kanały × wysokość × szerokość.")
        +image("cnn_architecture.png","Wszystkie Conv: filtry 3×3, stride 1, padding 1, z biasem. BN: BatchNorm; GAP: średnia przestrzenna każdego kanału. Między embeddingiem a głowicą nie ma dodatkowej aktywacji. Softmax jedynie zamienia logity na rozkład klas.",width=500)
        +p("Ten sam encoder obsługuje wszystkie widoki — „siamese” oznacza tu wspólne wagi, nie nową architekturę. Klasyfikator używa surowego z; odległości i podobieństwa liczymy po normalizacji L2. AE/VAE dodają dekoder tylko na czas uczenia, EMA — kopię nauczyciela. W kontroli bez BN pomijamy warstwy BN.","small")
        +p("Domyślnie 20 epok po 383 aktualizacje, SGD i malejący learning rate. Warianty 40/80 epok startują od początku, z harmonogramem rozłożonym na cały trening.","small")
        +h("Co oznacza uczciwe porównanie?","CE i metody półnadzorowane mają ten sam budżet etykiet, liczbę kroków i liczbę prezentacji obrazów etykietowanych. Metody półnadzorowane przetwarzają dodatkowo obrazy bez etykiet. Dlatego sprawdzamy korzyść z tych danych, ale nie twierdzimy, że koszt obliczeń jest równy. Trójki i contrastive korzystają z tych samych wyliczonych widoków, lecz dodają koszt porównywania embeddingów.")
        +p("Walidacja wymaga dodatkowych 1000 znanych etykiet; budżet 500 dotyczy wyłącznie treningu. To uczenie półnadzorowane przy stałych klasach, bez testowania adaptacji do nowych klas typowego dla wielu badań few-shot.","small"))

    add("Punkt wyjścia i twarde pseudoetykiety",h("1. Tylko nadzór: co robimy z danymi?",
        "Losujemy obrazy spośród 500 etykietowanych. Dla każdego tworzymy dwa słabe widoki, prosimy CNN o klasę i porównujemy odpowiedź z prawdziwą etykietą. Obrazy bez etykiet nie uczestniczą w tej metodzie. To nasza podstawowa kontrola, nazywana CE.")
        +brief("<b>CE:</b> L<sub>CE</sub> = średnia[−log p(y)] po obu widokach etykietowanych. p(y) to przewidywane prawdopodobieństwo prawdziwej klasy y. Minimalizujemy tę stratę.")
        +h("2. Twarde pseudoetykiety: co dokładamy?","Oprócz obrazów etykietowanych pobieramy obrazy bez etykiet. Dla każdego tworzymy słaby i mocny widok. Na słabym model przewiduje klasę. Jeśli jej pewność wynosi co najmniej 95%, uznajemy tę klasę za tymczasową etykietę i uczymy model, żeby rozpoznał ją także na mocnym widoku. Przy niższej pewności obraz nie wnosi straty pseudoetykietowania. Zwykły nadzór na 500 prawdziwych etykietach działa cały czas.")
        +brief("<b>Hard PL:</b> q = klasa argmax słabego widoku; w = 1 dla pewności ≥0,95, inaczej 0.<br>L = L<sub>CE</sub> + średnia[w · CE(q, p<sub>mocny</sub>)]. CE(q,p) mierzy błąd względem celu q; średnia obejmuje cały batch nieetykietowany, również przykłady z wagą 0. Cel i waga nie otrzymują gradientu.")
        +image("main_results.png","20 epok, 500 etykiet, seed0. Miękki wariant z wykresu objaśniamy na następnej stronie.")
        +rt([CE,HARD],["CE: tylko 500 prawdziwych etykiet","CE + twarde pseudoetykiety"])
        +note("Twarde pseudoetykiety podniosły test o 9,00 punktów procentowych. Jest to rdzeń mechanizmu FixMatch [1], lecz nie pełna reprodukcja jego treningu i architektury."))

    add("Miękkie pseudoetykiety: co się zmienia?",h("Oddzielamy dwie decyzje","Pierwsza: jaką odpowiedź chcemy uzyskać? Twardy cel mówi tylko „kot”. Miękki cel zachowuje rozkład ocen, np. „kot 80%, pies 15%, pozostałe 5%”. Druga: jak mocno dany obraz ma wpływać na uczenie? Twarda waga dopuszcza obraz lub go odrzuca; miękka zmniejsza jego wpływ stopniowo.")
        +p("W naszym miękkim wariancie — dalej soft PL — słaby widok dostarcza rozkładu klas, a mocny ma go odtworzyć. Rozkład dodatkowo wyostrzamy temperaturą 0,5: zwiększamy przewagę najbardziej prawdopodobnych klas. W pierwszej epoce obowiązuje próg 95%. Później wagę określamy na podstawie średniej i rozrzutu pewności z poprzedniej epoki: predykcje co najmniej tak pewne jak ta średnia mają pełną wagę, mniej pewne — malejącą. Cele powstają na nowo w każdym kroku; statystyki wag odświeżamy raz na epokę.")
        +brief("<b>Soft PL:</b> q = softmax(logity<sub>słaby</sub> / 0,5). Strata pozostaje L<sub>CE</sub> + średnia[w · CE(q, p<sub>mocny</sub>)], ale q jest rozkładem, a w maleje gładko według krzywej Gaussa poniżej średniej pewności poprzedniej epoki. Tylko wariant „miękka waga” zostawia twarde q. Cele i wagi są odłączone od gradientu.")
        +rt([HARD,PREFIX+"ce_2v_pseudo_soft_weight",SOFT],["Twardy cel i próg","Twardy cel + miękka waga","Miękki cel + miękka waga"])
        +p("Na seed0 same miękkie wagi dały prawie tyle samo co miękkie cele i wagi. Nie wykazaliśmy osobnej przewagi samego zmiękczenia celu. Miękkie ważenie jest pokrewne SoftMatch [2], a wyostrzanie rozkładu — MixMatch [3]; szczegóły naszych implementacji są inne.")
        +image("three_seeds.png","Trzy punkty na metodę: wyniki seedów 0, 1 i 2. Znacznik i odcinek: średnia i odchylenie standardowe, nie przedział ufności.",width=480)
        +note("Średni test: CE 48,84 ± 0,37%; twarde pseudoetykiety 57,39 ± 0,80%; miękkie cele i wagi 59,27 ± 0,94%. Miękki wariant wygrywa z twardym w każdej z trzech sparowanych prób, średnio o 1,88 p.p. To nasza najlepiej sprawdzona baza do dalszych pomysłów."))

    add("Dłuższy trening: największy dalszy przyrost",h("Co robimy z danymi?","Nie zmieniamy zbioru ani liczby etykiet. Powtarzamy ten sam sposób uczenia z miękkimi celami i wagami przez 40 lub 80 epok zamiast 20. Obrazy częściej przechodzą przez sieć, za każdym razem po losowych augmentacjach. To odrębne treningi od początku, z wolniej rozłożonym spadkiem learning rate — nie samo dopisanie epok do starego checkpointu.")
        +brief("Ta sama strata co w soft PL; liczba aktualizacji K = 383 · E dla E = 20, 40 lub 80 epok. W każdym runie learning rate maleje kosinusowo od 0,03 do okolic 0 w jego własnym horyzoncie K.")
        +image("training_horizon.png","Ten sam seed0 i zbiór 500 etykiet. Nie przeprowadzono CE80; brak punktu dla tej metody jest zamierzony.")
        +rt([SOFT,PREFIX+"ce_2v_pseudo_soft_40epochs",PREFIX+"ce_2v_pseudo_soft_80epochs"],["Miękkie PL: 20 epok","Miękkie PL: 40 epok","Miękkie PL: 80 epok"])
        +p("CE40 osiągnęło 49,34% testu wobec 49,25% CE20. Miękkie pseudoetykiety poprawiły się z 60,12% do 65,34%. Zmienił się także learning rate w kolejnych krokach, więc nie izolujemy samej liczby prezentacji obrazów.")
        +note("Wynik 65,34% kosztował cztery razy więcej aktualizacji niż wynik 20-epokowy. Nie porównujemy go jako przewagi przy tym samym koszcie nad dodatkami testowanymi przez 20 epok. Pełny nadzór S03 osiągnął 79,75% testu, ale znał 49 000 etykiet i miał inny protokół supervised."))

    add("Trójki: jak próbowaliśmy poprawić embeddingi",h("Co dokładamy do miękkich pseudoetykiet?","Dla wybranego obrazu — anchora — szukamy pozytywu, którego embedding chcemy przybliżyć, i negatywu, który chcemy odsunąć. Zwykła klasyfikacja i miękkie pseudoetykietowanie nadal działają. W nowym miksie używamy trzech źródeł relacji:")
        +table(["Rodzaj trójki","Co robimy z obrazami"],[
            ["Prawdziwe etykiety","Łączymy obrazy tej samej klasy jako anchor–pozytyw; obraz innej klasy jest negatywem. Preferujemy negatywy bliskie, ale nie bliższe od pozytywu; przy braku takich bierzemy najbliższy innej klasy."],
            ["Bez etykiet: własne augmentacje","Słaby i mocny widok tego samego obrazu tworzą anchor–pozytyw. Mocne widoki innych obrazów są negatywami. Ten składnik nie zna klas; może odpychać także dwa różne koty."],
            ["Prawdziwe i pewne pseudoetykiety","Dopuszczamy wszystkie prawdziwe etykiety i pseudoetykiety z pewnością co najmniej 95%. Wariant B szuka odległego pozytywu tej samej klasy i bliskiego negatywu innej klasy. Dla etykietowanego anchora pozytyw musi pochodzić z puli pseudoetykiet."]],"compact")
        +p("W B odległości mierzymy w znormalizowanych embeddingach. Pozytyw losujemy z najdalszej ćwiartki pasujących obrazów, a negatyw wybieramy najbliższy. Trójki z pseudoetykietami ważymy iloczynem pewności uczestników i włączamy od epoki 4. Nie wykonujemy dodatkowych forwardów CNN.")
        +brief("Kara jednej trójki: max(0, d(anchor, pozytyw) − d(anchor, negatyw) + 0,2), gdzie d to odległość euklidesowa. Całość: L = L<sub>soft PL</sub> + (L<sub>etykiety</sub> + L<sub>instancje</sub> + L<sub>pseudo</sub>)/30. Człon pseudo jest wyzerowany przez pierwsze 3 epoki.")
        +image("triplets.png","Nowa seria miningu, 20 epok i seed0. L: prawdziwa etykieta; U: obraz bez niej. Warianty opisano pod wykresem.",width=480)
        +p("A ogranicza relacje do L–U–L. B miesza etykiety i pseudoetykiety według odległości embeddingów. C wybiera trudne pary według pikseli augmentowanych obrazów. D dodaje do B maskę, która w instancyjnych negatywach pomija obrazy tej samej pewnej pseudo-klasy.","small")
        +note("B: 60,96% testu wobec 60,12% bez trójek, czyli +0,84 p.p.; walidacja +0,60 p.p. Wzrosła część miar reprezentacji, ale jeden seed nie potwierdza powtarzalnej przewagi."))

    add("Contrastive: porównujemy wiele par naraz",h("Co robimy z danymi?","Zostawiamy zwykły nadzór i miękkie pseudoetykiety. Z tych samych wyliczonych embeddingów tworzymy sieć relacji między parami obrazów. Dwa widoki własnego obrazu zawsze są parą dodatnią. Różne obrazy tej samej prawdziwej klasy także zbliżamy. W rozszerzeniu robimy to również dla zgodnych, pewnych pseudoetykiet — w tym dla par prawdziwa etykieta + pseudoetykieta.")
        +p("Pozostałe pary konkurują z dodatnimi w tym samym porównaniu. W przeciwieństwie do trójek nie wybieramy jednego konkretnego negatywu dla każdego pozytywu: uwzględniamy równocześnie inne obrazy z minibatcha. Relacje oparte na pseudoetykietach mają wagi zależne od pewności. Nie stosujemy osobnej głowicy projekcyjnej ani nowych forwardów CNN.")
        +p("Dodatkowy składnik działa od epoki 4 z wagą 0,05. Jeden zestaw relacji zapobiega równoczesnemu przyciąganiu i odpychaniu tej samej dodatniej pary przez osobne straty. Nadal możliwe są fałszywe negatywy wśród obrazów z odrzuconą pseudoetykietą.")
        +brief("L = L<sub>soft PL</sub> + 0,05 · L<sub>par</sub>. Dla anchora i liczymy P(j|i) przez softmax podobieństw kosinusowych / 0,2 po wszystkich innych widokach. L<sub>par</sub> to ważona średnia −log P(j|i) po jego pozytywach, następnie średnia po anchorach. Własna augmentacja ma wagę 1; para klasowa — iloczyn pewności, z progiem 0,95 dla pseudoetykiet.")
        +image("contrastive.png","20 epok, seed0. Oba warianty mają tę samą bazę supervised + miękkie pseudoetykiety.")
        +rt([SOFT,PREFIX+"contrastive_labeled_instance",CON],["Bez contrastive","Etykiety + własne instancje","Dodatkowo pary z pseudoetykiet"])
        +note("Wariant z pseudo-parami osiągnął 60,55% testu, ale 60,40% walidacji wobec 61,30% bez dodatku. Nie ma spójnej poprawy klasyfikacji. Geometria embeddingów zmieniła się korzystniej w części miar. Pokrewne rodziny: SupCon, SimCLR, CoMatch i SimMatch [5–8]."))

    add("Active learning: jak model prosił o etykiety",h("Co robimy inaczej z danymi?","Zaczynamy od 300 prawdziwych etykiet — po 30 na klasę. Po epokach 3, 6, 9, 12 i 15 model wybiera po 40 kolejnych obrazów. Dopiero po wyborze odsłaniamy ich prawdziwe etykiety i przenosimy obrazy do zbioru etykietowanego. Kończymy na 500 etykietach łącznie; ich liczba na klasę może być różna. Przez cały czas pozostałe obrazy mogą uczyć model przez pseudoetykiety.")
        +table(["Sposób wyboru","Jaki obraz chcemy oznaczyć?"],[
            ["Losowanie","Losuj 40 kandydatów bez zwracania. Kontrola z tym samym harmonogramem zdobywania etykiet."],
            ["Niepewność (entropy)","Policz H(p) = −Σ p<sub>k</sub> log p<sub>k</sub>; wybierz 40 najwyższych wartości. Model nie ma tu zdecydowanej odpowiedzi."],
            ["Niepewność + różnorodność","Weź do 400 najbardziej niepewnych kandydatów; kolejno wybieraj najdalszego od już oznaczonych i wcześniej wybranych embeddingów (odległość kosinusowa)."],
            ["BADGE","Dla każdej próbki przybliż gradient ostatniej warstwy, używając jej własnej klasy argmax. Wybierz największą normę, potem losuj kolejne proporcjonalnie do kwadratu odległości od dotychczas wybranych gradientów. Łączy siłę sygnału i różnorodność."],
            ["Gęstość i sąsiedzi","Rankingi: entropia × gęstość lub niezgodność JS z predykcją sąsiadów × gęstość. W miksie połowę zapytań losujemy. Osobny wariant wybiera połowę pewnych, niespójnych z sąsiadami obrazów, resztę według entropii."],
            ["Niezgodność widoków","Policz entropię średniej predykcji z widoków minus średnią ich entropii; wybierz 40 największych różnic."]])
        +h("Jak przebiega jedna runda?","Losujemy do 4096 kandydatów z aktualnej puli nieetykietowanej. Oceniamy ich bieżącą siecią, wybieramy 40 według zadanej strategii, odsłaniamy etykiety i kontynuujemy trening tej samej sieci. Nie restartujemy wag, optymalizatora ani learning rate. Walidacja i test nie są kandydatami, a selekcja nie zna ukrytych etykiet.")
        +brief("Po rundzie: L ← L ∪ wybrane obrazy; U ← U bez tych obrazów. Strata bazowa pozostaje hard PL lub soft PL, ewentualnie z trójkami. Kontrola frozen: weź końcowe 500 obrazów danej strategii → zresetuj sieć → ucz na wszystkich od pierwszego kroku.")
        +note("Porównujemy przede wszystkim strategie mające ten sam harmonogram i tę samą metodę bazową. Statyczne 500 etykiet od początku to dodatkowe odniesienie: model dostaje tam pełny nadzór wcześniej. AL z SSL i BADGE mają wcześniejsze odpowiedniki w literaturze [9,10,12]."))

    add("Active learning: co wyszło i czego nie wiemy",image("active_learning.png","20 epok, seed0, miękkie pseudoetykiety. „Końcowy wybór od początku” oznacza osobny trening od zera na dokładnie tych 500 obrazach, które wcześniej zebrało AL.")
        +h("Wybranie trudnych obrazów nie dało tu przewagi","Soft AL z losowaniem osiągnęło 59,12% testu, a z BADGE 58,22%, przy 60,12% statycznej bazy. Dodanie trójek do soft AL dało 57,94% z losowaniem i 58,36% z BADGE. W starszej bazie z twardymi pseudoetykietami: random 56,08%, entropy 54,91%, entropy + diversity 54,32%, BADGE 56,21%; pozostałe strategie 54,65–56,40%. Nie znaleźliśmy przekonująco skuteczniejszego doboru niż losowanie.")
        +h("Czy AL dostało za mało sygnału nadzorowanego?","Tak, względem statycznych 500 etykiet miało około 7,75% mniej prezentacji obrazów etykietowanych. Nowe etykiety pojawiały się też późno, przy mniejszym learning rate. Ale soft AL random i BADGE miały takie same prezentacje etykiet i koszt scoringu: 22 380 dodatkowych forwardów. Sam mniejszy nadzór nie tłumaczy więc różnicy między nimi.")
        +h("Sprawdziliśmy też sam końcowy wybór 500 obrazów","Po zakończeniu AL wzięliśmy zebrane przez nie etykiety i uczyliśmy nowe modele od początku, udostępniając wszystkie naraz. Końcowy zestaw random dał 58,94%, a BADGE 59,07% testu. Różnica tylko 0,13 p.p. nie wskazuje na wyraźnie lepszy wybór BADGE. Ta kontrola zmienia jednocześnie czas dostępu do etykiet i ich łączną ekspozycję; nie izoluje tych przyczyn osobno.")
        +note("Wniosek: zawiodły te strategie w tym pilocie, a nie „cały active learning”. Nie rozdzieliliśmy wpływu balansu klas, początkowych etykiet, terminów query, jakości embeddingów i ograniczenia kandydatów. To możliwe wyjaśnienia, nie ustalone przyczyny."))

    reps={r["run"]:r for r in data["reps"]}
    rep_runs=[CE,SOFT,PREFIX+"ce_2v_pseudo_soft_80epochs",TRIP,CON,"cnn_pilot_full_seed0"]
    rep_titles=["CE20","Soft PL20","Soft PL80","Soft + trójki B","Soft + contrastive/pseudo","Pełny nadzór (49k)"]
    add("Reprezentacje: czy obrazy są opisane lepiej?",h("Jak to mierzymy bez patrzenia tylko na rysunek?","Zamrażamy CNN i badamy jej embeddingi na tych samych 1000 obrazach walidacyjnych. Uczymy też od zera prosty liniowy klasyfikator na embeddingach tylko 500 etykietowanych obrazów. Taki „probe” sprawdza, ile informacji o klasie można łatwo odczytać z reprezentacji. Nie zmienia samej CNN.")
        +brief("Obraz → zamrożony encoder → wektor 128D → nowa głowica uczona tylko na 500 etykietach. Probe: standaryzacja cech + regresja logistyczna (C=1). Porównanie nieliniowe: RBF SVM lub MLP z 64 neuronami. Ocenę liczymy na 1000 osobnych obrazów walidacyjnych.")
        +image("representation_probe.png","Takie same 500 etykiet do nowych głowic we wszystkich przypadkach. Encoder pełnego nadzoru wcześniej widział 49 000 etykiet: to odniesienie diagnostyczne, z większym budżetem.",width=485)
        +table(["Model","Probe [%]","Czystość sąsiadów [%]","Silhouette"],[
            [title,report.pct(reps[r]["linear_probe_val_accuracy"]),report.pct(reps[r]["val_neighbor_purity_10"]),report.num(reps[r]["val_silhouette_cosine"],4)]
            for r,title in zip(rep_runs,rep_titles)],"compact")
        +p("<b>Czystość:</b> jaki procent dziesięciu najbliższych innych obrazów walidacyjnych ma tę samą prawdziwą klasę. <b>Silhouette:</b> czy obrazy są bliżej własnej klasy niż innych; wyższa wartość oznacza wyraźniejszy podział w tej mierze. Obie metryki liczymy w 128D, według odległości kosinusowej, a nie na rysunku t-SNE.","small")
        +note("Pseudoetykiety wyraźnie poprawiają także reprezentację. Trójki B zwiększają czystość z 47,72% do 49,61%, ale probe prawie się nie zmienia: 60,50% → 60,60%. Lepsze skupienia według jednej miary nie muszą oznaczać lepszego klasyfikatora."))

    add("t-SNE: porównanie tych samych nowych obrazów",image("validation_tsne.png","We wszystkich panelach te same 1000 obrazów walidacyjnych. Kolory oznaczają prawdziwe klasy; te obrazy nie były użyte do uczenia. Każdy model ma osobny układ współrzędnych.",width=485)
        +p("Widać porządkowanie klas, ale t-SNE może wyolbrzymiać wizualną separację. Nie odczytujemy z odległości między wyspami dokładności klasyfikatora i nie porównujemy ich powierzchni między modelami. Wnioski opieramy także na wynikach walidacji i metrykach z poprzedniej strony [11].","small")
        +p("Wykorzystano zapisane projekcje: t-SNE 2D, perplexity 30, seed0, 1000 iteracji, z wstępną redukcją liniową do 30 wymiarów. Ta redukcja jest tylko etapem obliczania projekcji; raport nie zawiera osobnych wykresów PCA.","small"))

    add("t-SNE: cała pula, a nie tylko znane etykiety",image("full_pool_soft.png","Miękkie pseudoetykiety, 80 epok. Jeden wspólny t-SNE dla 50 000 obrazów i osobne widoki jego podzbiorów. Oficjalny test nie jest włączony.",width=500)
        +h("Co pokazują cztery panele?","Góra lewo: wszystkie obrazy. Góra prawo: 500 z prawdziwymi etykietami w treningu. Dół lewo: 48 500 bez etykiet. Dół prawo: 1000 walidacyjnych. Tutaj współrzędne są wspólne, więc możemy sprawdzić, gdzie trafiają poszczególne pule. Znane 500 obrazów układa się czyściej niż obrazy nieetykietowane i nowe.")
        +h("Dlaczego ładne skupienia nie dały prawie 100%?","Model potrafi dobrze rozdzielać przykłady, które znał z nadzoru, a nadal mylić podobne klasy na pozostałych obrazach. Mieszanie klas w dużej puli oraz wyniki probe wskazują, że problem nie sprowadza się do dobrania lepszej ostatniej warstwy.")
        +note("Prawdziwe klasy nieetykietowanych obrazów posłużyły wyłącznie do pokolorowania analizy po treningu. Nie były używane do pseudoetykiet, relacji ani aktywnego wyboru etykiet. Rysunek nie jest dowodem na przeciek etykiet."))

    add("Pozostałe próby: co sprawdziliśmy bez przełomu",p("Te próby pozostają częścią dokumentacji, ale nie wymagają osobnego rozdziału dla każdego ustawienia. Wyniki są z pojedynczych seedów, o ile nie podano inaczej.")
        +table(["Podejście","Co robiliśmy z danymi i co wyszło"],[
            ["Starsze trójki i miks","Strata bazowa + ważone kary trójek; zmienialiśmy dobór relacji z etykiet, augmentacji i pseudoetykiet oraz negatywów. Nie dały ustalonej przewagi nad soft PL. Stary miękki miks: 60,57% testu."],
            ["AE i VAE","Mocny obraz → encoder → dekoder → odtworzenie kadru sprzed silnej augmentacji. Strata: CE + MSE rekonstrukcji, opcjonalnie hard PL; w VAE dodatkowo losowanie reprezentacji i β·KL do rozkładu normalnego. β rośnie do 0,01 przez 5 epok. AE/VAE: 48,42/48,66% testu; z hard PL: 58,14/58,08%, bez poprawy baz. To wspólne uczenie, nie dawne eksperymenty z pretreningiem VAE."],
            ["Sąsiedzi embeddingów","Embedding U → 15 najbliższych oznaczonych sąsiadów → ich ważony rozkład klas → porównanie z predykcją modelu. Wynik koryguje wagę PL albo wybór w AL. Lepsza detekcja części błędów nie poprawiła treningu. Statyczne wagi: 59,87% testu; warianty z AL: 57,86–59,05%."],
            ["Nauczyciel EMA i temperatura","Wagi nauczyciela ← α·stare wagi + (1−α)·wagi studenta, α narasta do 0,99. Od epoki 4 nauczyciel przewiduje q; uczy się tylko student. Zmienialiśmy też temperaturę celu na T=1, czyli bez wyostrzania. EMA T=0,5/1: studenci 59,19/59,63%; T=1 bez EMA: 60,07%. Brak poprawy. R53–R55."],
            ["BatchNorm i sam przepływ U","Kontrola: te same wejścia L/U do CNN, ale strata tylko CE na L → 48,05% testu; sam wpływ U na statystyki BN nie wyjaśnia zysku PL. Druga kontrola zastępuje BN identycznością: CE 43,93%, hard PL 54,63%. GroupNorm i LayerNorm nie wytrenowano porównawczo."],
            ["Inny klasyfikator na gotowym embeddingu","Dla soft80: nowa głowica liniowa 64,40%, RBF SVM 64,00%, mały MLP 64,10% walidacji. Nie znaleźliśmy dużego rezerwuaru poprawy samą zmianą głowicy."],
            ["Wcześniejsze kontrole supervised","S01: 100 etykiet, 29,77% testu; S02: 500 etykiet, 49,62%; S03: 49 000 etykiet, 79,75%. Miały jeden widok etykietowanego obrazu, więc nie zastępują kontroli CE20 głównej serii."]],"compact")
        +p("SVM dobierał ustawienia na 500 etykietach, ale encoder znał wcześniej całe te 500. Jego wewnętrzna walidacja nie jest niezależną oceną całego procesu. Dokładne konfiguracje i wszystkie wyniki, także pominięte z wykresów, są osadzone w PDF.","small"))

    add("Jak to się ma do literatury?",p("Większość użytych składników ma znane odpowiedniki. Samo połączenie ich w konfigurację nie jest jeszcze wykazaniem oryginalnego wkładu.")
        +table(["Nasza próba","Najbliższa rodzina i ważna różnica"],[
            ["Twarde pseudoetykiety","FixMatch [1,15]: pewna odpowiedź na słabym widoku uczy mocny widok. Nasz model, długość treningu i proporcja danych nie odtwarzają pełnej receptury."],
            ["Miękkie wagi","SoftMatch [2,16]: płynne ważenie pewności. U nas statystyki pochodzą z poprzedniej epoki; w SoftMatch są aktualizowane EMA co krok. Mamy inną szerokość wagi i brak Uniform Alignment, czyli wyrównywania wpływu klas w mechanizmie ważenia."],
            ["Miękkie cele","MixMatch [3] używa rozkładów i ich wyostrzania. Nie stosujemy jego pełnego uśredniania predykcji i MixUp. W SoftMatch „soft” dotyczy wag; cele mogą być twarde."],
            ["Nauczyciel EMA","Mean Teacher [4]: uśredniana kopia wag daje stabilniejsze cele. Nasz wariant to konkretna uproszczona realizacja tej idei, bez uzyskanej poprawy."],
            ["Uczenie relacji","SupCon i SimCLR [5,6] uczą przez pary; CoMatch i SimMatch [7,8] łączą SSL z informacją o podobieństwie. Nasze warianty nie odtwarzają ich pełnych architektur i pamięci."],
            ["Active learning","BADGE [9] i łączenie AL z MixMatch [10] poprzedzają ten pilot. Wrażliwość AL na protokół ocen opisuje [12]."],
            ["AE/VAE","Odtwarzanie uszkodzonych danych [13] i wariacyjny autoenkoder [14] to znane kierunki. Nasz wspólny trening rekonstrukcji i klasyfikacji nie jest nową zasadą uczenia."]])
        +h("Dlaczego FixMatch raportuje znacznie więcej?","Praca podaje 94,93% dla CIFAR-10 przy 250 etykietach [1]. Kod referencyjny domyślnie wykonuje około miliona aktualizacji i używa proporcji nieetykietowanych do etykietowanych 7:1 [15]. Nasz pilot: 7660 aktualizacji w 20 epokach albo 30 640 w 80, proporcja 1:1, inna sieć i ocena ostatniego studenta. Nie da się przypisać tej luki jednemu mechanizmowi pseudoetykiet.")
        +note("To mapa podobieństw, nie kompletny przegląd nowości. Wyniki pilota można opisać rzetelnie, ale publikacja metody wymaga jeszcze wskazania konkretnej luki i porównania z wiarygodnymi odpowiednikami."))

    add("Co z tego wynika dla dalszych badań?",h("Najbardziej obiecujący punkt wyjścia","Miękkie pseudoetykiety: poprawiają klasyfikację i embeddingi, a przewagę nad twardymi potwierdziliśmy w trzech seedach. Dłuższy trening dał dalszy zysk. Osobny wpływ miękkiego celu wobec miękkiej wagi pozostaje słabiej zbadany.")
        +h("Co wniosły próby z reprezentacją?","Pozwoliły odróżnić lepszą geometrię od lepszej generalizacji. Trójki i contrastive potrafią zmienić lokalne sąsiedztwo klas, lecz prosta nowa głowica nie uzyskała z tego dużego zysku. To użyteczna obserwacja diagnostyczna, jeszcze bez potwierdzenia w kolejnych seedach.")
        +h("Co z active learningiem?","Nie znaleźliśmy skutecznej przewagi wyboru aktywnego, także po kontroli końcowego zestawu etykiet. Wartość tego wątku nie polega obecnie na nowej skutecznej strategii, lecz na rozpoznaniu ograniczeń protokołu. Kolejny eksperyment powinien sprawdzać jedną przyczynę, np. terminy pozyskania etykiet, zamiast otwierać kolejną szeroką siatkę metod.")
        +h("Czego potrzeba przed wnioskami do publikacji?","Zamrozić jedną hipotezę i małą listę metod. Potwierdzić wyniki w kilku seedach, budżetach etykiet i zbiorach oraz z mocnym, odtworzonym baseline’em literaturowym. Dobór ustawień powinien opierać się na walidacji; test był dotąd oglądany wielokrotnie, więc dalsza ocena wymaga większej dyscypliny. Nie traktować pojedynczego najlepszego runu jako ustalonej przewagi.")
        +note("Wykonaliśmy 55 głównych treningów oraz 3 wcześniejsze supervised. To nowe lokalne eksperymenty. Repo powstało na podstawie access_v4.pdf; oryginalne logi doktoranta nie były dostępne. Nie deklarujemy odtworzenia opisanych tam 5760 eksperymentów.")
        +h("Gdzie są szczegóły?","W PDF osadzono trzy pliki: <b>konfiguracje_i_wyniki.json</b> — wszystkie 58 konfiguracji i wyników oraz audyty reprezentacji; <b>katalog_wynikow.csv</b> — wyniki, liczniki danych i kosztów; <b>manifest_zrodel.json</b> — źródła z sumami kontrolnymi. R01–R55 i S01–S03 odpowiadają pozycjom katalogu. Są też zwykłymi plikami obok PDF. Generator nie uruchamia treningów."))

    entries=[]
    for i,(author,title,url) in enumerate(report.REFERENCES,1):
        entries.append(p(f'<b>[{i}] {report.esc(author)}.</b> {report.esc(title)}<br><a href="{url}">{url}</a>',"reference"))
    add("Literatura i źródła wyników","".join(entries)
        +p("Wyniki własne: outputs/pilot_comparison.json; outputs/representation_reference/comparison.json; outputs/frozen_classifiers/report.json oraz config.json i metrics.json runów. Implementacje: fewshot_metric/pilot.py, mined_triplets.py, mixed_contrastive.py, active.py i active_selection.py. Data migawki i dostępu do literatury: 20.10.2026.","small"))
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8",line_buffering=True)
    OUT.mkdir(parents=True,exist_ok=True)
    report.REFERENCES=[REFERENCES[i-1] for i in REF_IDS]
    report.CSS += "\n.reference {font-size:8pt;line-height:1.2;margin:0 0 8pt;}\n.algorithm {font-size:8.5pt;line-height:1.25;background-color:#f0f5f7;border-left:2pt solid #839ca9;padding:6pt;margin:6pt 0;}\n"
    data=report.load_data();data["sources"].append(Path(__file__))
    charts(data)
    content=sections(data)
    report.write_attachments(data)
    path=report.render_report(content)
    report.verify_report(path,content)
    doc=report.fitz.open(path)
    assert len(doc)<=19, len(doc)
    text=" ".join(p.get_text() for p in doc)
    assert all(t in text for t in ["BADGE","58,94","59,07","7,75","Miękkie"])
    assert not list((OUT/"assets").glob("*pca*"))
    manifest=report.read(OUT/"manifest_zrodel.json")
    assert all(hashlib.sha256((ROOT/f["path"]).read_bytes()).hexdigest()==f["sha256"] for f in manifest["files"])
    with (OUT/"katalog_wynikow.csv").open(encoding="utf-8-sig") as f:assert len(list(csv.DictReader(f)))==58
    (OUT/"report_summary.json").write_text(json.dumps({"pages":len(doc),"completed_runs":58,
        "style":"Plain-language descriptions with brief algorithms and loss summaries; CNN diagram and comparison charts; no PCA figures",
        "generator":"scripts/build_research_overview.py"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    doc.close()
    import gc
    gc.collect()
    try:
        (OUT/"_report_raw.pdf").unlink(missing_ok=True)
    except PermissionError:
        pass
    print(path)


if __name__=="__main__":main()
