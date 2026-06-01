"""
Approximate multiplier „Approxx 10" – implementacja w Pythonie.

Źródło: Sabetzadeh, Moaiyeri, Ahmadinejad,
        "An Ultra-Efficient Approximate Multiplier With Error Compensation
         for Error-Resilient Applications",
        IEEE TCAS-II, vol. 70, no. 2, Feb. 2023.

Architektura (8-bit × 8-bit → 16-bit):

  Pełny iloczyn cząstkowy pp[i][j] = a_i AND b_j; pozycja bitu: 2^(i+j).

  1. Constant-truncated region (bity 7:0 wyniku)
     ┌─ Right nibble (kolumny 0-2):  wartość stała = 6 (binary "110")
     │   – uśredniona wartość kolumn 0-2: E[k]=0.25→0, E[k]=0.5→1, E[k]=0.75→1
     └─ Left nibble  (kolumny 3-7):  wyzerowane (przybliżone kompresory → 0)

     Łączna stała dla bitów 7:0 = 0b00000110 = 6.

  2. Exact part (kolumny 8-14)
     Sumuje dokładnie wszystkie pp[i][j] z i+j ≥ 8.

  3. Error Compensation Module (ECM)
     Dwie 4-wejściowe bramki OR na iloczynach cząstkowych z kolumny 7
     (najważniejsza kolumna stałego regionu, waga 2^7 = 128):
       OR1: pp[0][7], pp[1][6], pp[2][5], pp[3][4]   (a[0:3] × b[4:7])
       OR2: pp[4][3], pp[5][2], pp[6][1], pp[7][0]   (a[4:7] × b[0:3])
     Każde wyjście OR podaje carry do kompresorów kolumny 8 (+256 jeśli = 1).

     Uzasadnienie: błędy z left-nibble są zawsze ujemne (ED = −liczba_jedynek);
     OR wykrywa czy cokolwiek było niezerowe i kompensuje najbardziej prawdopodobny
     przypadek błędu (10.5% szansy na każdą jedynkę w 4-wejściowym kompresorze).
"""


def exact_multiply(a: int, b: int) -> int:
    """Dokładne mnożenie – zwykły operator Pythona, traktuje a i b jako liczby 8-bit."""
    return (a & 0xFF) * (b & 0xFF)
    # a & 0xFF → maska 11111111; ucina bity powyżej pozycji 7, gwarantuje zakres 0–255
    # b & 0xFF → to samo dla b
    # * → standardowe mnożenie; Python nie ogranicza wyniku, zwraca pełne 0–65 025


def approx10_multiply(a: int, b: int) -> int:
    """
    Przybliżone mnożenie według architektury Approxx 10.

    Wejście: a, b – liczby całkowite 8-bit bez znaku (0–255).
    Wyjście: 16-bitowy wynik przybliżony.

    Odpowiedniki sprzętowe (rys. 1 z artykułu):
      upper          ←→  blok „Exact part" (Dadda + RCA)
      + 6            ←→  blok „Constant-truncated region" (hardwired bits)
      or1, or2       ←→  blok „ECM" (dwie bramki OR4)
      (or1+or2) << 8 ←→  carry wstrzyknięty do kompresorów kolumny 8
    """

    # ── Wejścia układu: 8-bitowe szyny a[7:0] i b[7:0] ──────────────────────
    a &= 0xFF
    # Sprzęt: fizyczny pin wejściowy ma 8 linii; & 0xFF modeluje tę szerokość szyny.
    b &= 0xFF
    # Identycznie dla b; bez tego Python użyłby wszystkich bitów liczby całkowitej.

    # ── BLOK 1: Exact part – drzewo redukcji Dadda dla kolumn 8–14 ───────────
    # Odpowiada prawej (górnej) połowie schematu z Fig. 1 artykułu.
    # W sprzęcie: half adder, 2× full adder, 3× kompresor 4:2 (etap 1),
    #             2× full adder, 4× kompresor 4:2 (etap 2), RCA na wyjściu.
    # W Pythonie modelujemy wynik końcowy tej logiki przez bezpośrednie sumowanie pp.

    # Górny trójkąt macierzy PP: 1+2+…+7 = 28 iloczynów cząstkowych (kolumny 8–14).
    # Dolna granica j = max(8−i, 0): dla i=0 zakres jest pusty (bo 0+j ≤ 7 zawsze),
    # dla i=7 pokrywa j=1..7 – implementuje dokładnie strukturę Dadda górnego trójkąta.
    upper = 0
    # upper: akumulator; na początku zero, będzie zbierał wartości wszystkich pp z kolumn 8–14.

    for i in range(8):
        # i: numer bitu wejścia a, odpowiada numerowi wiersza w macierzy PP (a[0]..a[7]).
        # W sprzęcie: każdy wiersz to osobna warstwa bramek AND zasilana jedną linią a[i].

        ai = (a >> i) & 1
        # a >> i: przesuwa a w prawo o i pozycji, bit i trafia na pozycję 0.
        # & 1: maska wyciągająca tylko najniższy bit → ai ∈ {0, 1}.
        # Sprzęt: linia sygnałowa a[i] podłączona do jednego wejścia każdej bramki AND w wierszu i.

        if ai == 0:
            # Jeśli a[i] = 0, to pp[i][j] = 0 AND b[j] = 0 dla każdego j.
            # Cały wiersz i macierzy PP jest zerowy – nie trzeba nic dodawać.
            # Sprzęt: bramki AND z jednym wejściem = 0 zawsze dają 0; kompilator usuwa je statycznie.
            continue
            # Przeskocz do następnego i bez wchodzenia w wewnętrzną pętlę.

        for j in range(max(8 - i, 0), 8):
            # j: numer bitu wejścia b, odpowiada kolumnie w macierzy PP (b[0]..b[7]).
            # max(8−i, 0): dolna granica j taka, że i+j ≥ 8 (exact part zaczyna się od kolumny 8).
            # Dla i=1: j startuje od 7 (tylko pp[1][7] trafia do kolumny 8).
            # Dla i=7: j startuje od 1 (pp[7][1]..pp[7][7] – siedem elementów).
            # Sprzęt: tylko te bramki AND są fizycznie obecne w bloku „Exact part".

            bj = (b >> j) & 1
            # Analogicznie do ai: wyciąga bit b[j] ze słowa b.
            # Sprzęt: linia sygnałowa b[j] – drugie wejście bramki AND dla pp[i][j].

            if bj:
                # pp[i][j] = ai AND bj; ponieważ ai = 1 (sprawdziliśmy wyżej),
                # pp[i][j] = 1 wtedy i tylko wtedy gdy bj = 1.
                # Sprzęt: wyjście bramki AND = 1 → ten iloczyn cząstkowy jest aktywny.

                upper += 1 << (i + j)
                # 1 << (i+j): liczba 2^(i+j) – wartość pozycyjna iloczynu pp[i][j].
                # Dodajemy tę wartość do akumulatora.
                # Sprzęt: drzewo adderów (kompresory 4:2 + RCA) wykonuje tę samą sumę,
                # ale równolegle i w jednym cyklu zegarowym.

    # ── BLOK 2: Constant-truncated region – bity 7:0 wyjścia ─────────────────
    # Odpowiada lewej (dolnej) połowie schematu z Fig. 1.
    # W sprzęcie: bity 7:0 rejestru wyjściowego są dosłownie przewodami:
    #   bit 0 → GND (logiczne 0)   bit 1 → VDD (logiczne 1)
    #   bit 2 → VDD (logiczne 1)   bity 3–7 → GND (logiczne 0)
    # Żaden sumator nie istnieje dla tych bitów – to czyste okablowanie.

    # 0b00000110 = 6: zaokrąglona wartość oczekiwana kolumn 0–2:
    #   E[col0] = 1×(1/4) = 0.25 → bit0 = 0
    #   E[col1] = 2×(1/4) = 0.50 → bit1 = 1
    #   E[col2] = 3×(1/4) = 0.75 → bit2 = 1
    # Bity 7:3 = 0 (zerowe kompresory left nibble).
    # Skutek uboczny: dla a=0 lub b=0 upper=0, więc wynik = 6 ≠ 0 – świadoma konsekwencja sprzętowa.
    result = upper + 6
    # result: zmienna przechowująca bieżący wynik 16-bitowy.
    # upper to bity 15:8 (exact part); +6 dokłada stałe bity 7:0.
    # To jedyne „dodawanie" – nie ma tu sumatora kolumn 0–7, tylko wpisanie stałej.

    # ── BLOK 3: ECM – dwie 4-wejściowe bramki OR ─────────────────────────────
    # Odpowiada blokowi „ECM" z Fig. 1 i opisu §III artykułu.
    # Sprzętowo: NAND4 + inwerter = 10 tranzystorów na bramkę OR, 20 łącznie (§III artykułu).
    # Wejścia: iloczyny cząstkowe z kolumny 7 (i+j = 7), najważniejszej kolumny stałego regionu.
    # Cel: wykryć czy cokolwiek z left nibble byłoby niezerowe → dodać carry korekcyjny.

    # OR1 – górna połowa przekątnej kolumny 7 (a[0:3] AND b[4:7])
    or1 = (
        (((a >> 0) & 1) & ((b >> 7) & 1))
        # pp[0][7] = a[0] AND b[7]; bramka AND → wejście 1 bramki OR1.
        # (a >> 0) & 1 = bit 0 wejścia a; (b >> 7) & 1 = bit 7 wejścia b.
        | (((a >> 1) & 1) & ((b >> 6) & 1))
        # pp[1][6] = a[1] AND b[6]; bramka AND → wejście 2 bramki OR1.
        | (((a >> 2) & 1) & ((b >> 5) & 1))
        # pp[2][5] = a[2] AND b[5]; bramka AND → wejście 3 bramki OR1.
        | (((a >> 3) & 1) & ((b >> 4) & 1))
        # pp[3][4] = a[3] AND b[4]; bramka AND → wejście 4 bramki OR1.
    )
    # or1 ∈ {0, 1}: 1 jeśli przynajmniej jeden z czterech pp w górnej połowie kolumny 7 jest 1.
    # Sprzęt: wyjście bramki OR1 = jeden bit sygnałowy; idzie jako carry do kompresorów kolumny 8.

    # OR2 – dolna połowa przekątnej kolumny 7 (a[4:7] AND b[0:3])
    or2 = (
        (((a >> 4) & 1) & ((b >> 3) & 1))
        # pp[4][3] = a[4] AND b[3]; bramka AND → wejście 1 bramki OR2.
        | (((a >> 5) & 1) & ((b >> 2) & 1))
        # pp[5][2] = a[5] AND b[2]; bramka AND → wejście 2 bramki OR2.
        | (((a >> 6) & 1) & ((b >> 1) & 1))
        # pp[6][1] = a[6] AND b[1]; bramka AND → wejście 3 bramki OR2.
        | (((a >> 7) & 1) & ((b >> 0) & 1))
        # pp[7][0] = a[7] AND b[0]; bramka AND → wejście 4 bramki OR2.
    )
    # or2 ∈ {0, 1}: analogicznie do or1, ale dla dolnej połowy kolumny 7.
    # Sprzęt: wyjście bramki OR2 = drugi bit carry do kompresorów kolumny 8.

    # ── Wstrzyknięcie carry ECM do kolumny 8 ─────────────────────────────────
    # Korekcja skokowa: ECM ∈ {0, 256, 512}; E[korekcja] ≈ 350,
    # co przesuwa średnie ED z −442 (bez ECM) do −92 (z ECM).
    result += (or1 + or2) << 8
    # (or1 + or2): suma carry = 0, 1 lub 2 (ile bramek OR dało 1).
    # << 8: przesuwa na pozycję bitu 8, czyli mnożenie przez 2^8 = 256.
    # Sprzęt: or1 i or2 idą jako dwa bity carry do wejść kompresorów 4:2 na kolumnie 8.
    # Efekt: result += 0×256, 1×256 lub 2×256 w zależności od wejść.

    # ── Rejestr wyjściowy 16-bit ──────────────────────────────────────────────
    return result & 0xFFFF
    # & 0xFFFF: maska 16-bitowa (1111111111111111); ucina ewentualne bity powyżej pozycji 15.
    # Sprzęt: wynik trafia do 16-bitowego rejestru wyjściowego; bity powyżej 15 są fizycznie nieobecne.
    # Bez ECM: result ≤ 65 025+6 = 65 031 (mieści się w 16 bitach).
    # Z ECM:   result ≤ 65 031+512 = 65 543 > 65 535 → maska jest potrzebna.


def approx10_multiply_no_ecm(a: int, b: int) -> int:
    """Approxx 10 bez modułu ECM – exact upper + stała 6, bez bramek OR.

    Używana wyłącznie jako punkt odniesienia dla ecm_comparison.png.
    Nie odpowiada żadnemu samodzielnie opisanemu układowi w artykule –
    to wariant pomocniczy do wizualizacji wkładu ECM.
    """
    a &= 0xFF                            # maska wejścia a do 8 bitów
    b &= 0xFF                            # maska wejścia b do 8 bitów
    upper = 0                            # akumulator exact part (identyczny jak w approx10_multiply)
    for i in range(8):                   # iteracja po wierszach macierzy PP
        ai = (a >> i) & 1                # bit a[i]
        if ai == 0:                      # cały wiersz zerowy → pomiń
            continue
        for j in range(max(8 - i, 0), 8):  # tylko kolumny 8–14 (exact part)
            if (b >> j) & 1:             # pp[i][j] = 1 → dodaj do akumulatora
                upper += 1 << (i + j)   # waga pozycyjna 2^(i+j)
    return (upper + 6) & 0xFFFF
    # Brak or1, or2 i wstrzyknięcia carry – ECM jest wyłączony.
    # Wynik jest zawsze ≤ dokładnego (ED ≤ 0), bo lower_part jest zawsze ≥ 0.
