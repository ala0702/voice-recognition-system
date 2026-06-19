# Dokumentacja eksperymentów — System biometrycznej weryfikacji mówcy

## Spis treści
1. [Opis systemu](#opis-systemu)
2. [Struktura danych testowych](#struktura-danych-testowych)
3. [Metryki oceny](#metryki-oceny)
4. [Eksperyment 1 — Punkt odniesienia (Baseline)](#eksperyment-1)
5. [Eksperyment 2 — Skalowanie amplitudy](#eksperyment-2)
6. [Eksperyment 3 — Zmiana częstotliwości próbkowania](#eksperyment-3)
7. [Eksperyment 4 — Szum gaussowski (biały)](#eksperyment-4)
8. [Eksperyment 5 — Szumy środowiskowe (dźwięki tła)](#eksperyment-5)
9. [Eksperyment 6 — Kompresja audio kodekami stratymi](#eksperyment-6)
10. [Eksperyment 7 — Pogłos akustyczny](#eksperyment-7)

---

## Opis systemu

System realizuje zadanie **weryfikacji mówcy** — użytkownik podaje swoją tożsamość i nagranie głosowe, a system decyduje, czy to on naprawdę mówi, czy ktoś się pod niego podszywa.

### Model: ECAPA-TDNN

Sercem systemu jest model **ECAPA-TDNN** (*End-to-End Conv-Attention* Time Delay Neural Network) wytrenowany na zbiorze VoxCeleb. Model ten przetwarza cechy mowy i zwraca **wektor embeddingu** — 192-wymiarową reprezentację liczbową głosu mówcy, która zawiera jego charakterystykę biometryczną.

### Ekstrakcja cech

Przed podaniem sygnału do modelu, z surowego nagrania WAV (16 kHz) wyodrębniane są **logarytmiczne filtry Mel (Log-Mel Filterbanks)**:
- 80 filtrów Mel
- Okno analizy: 25 ms (400 próbek przy 16 kHz)
- Krok: 10 ms (160 próbek)
- Transformacja: log(spektrogram Mel + 1e-6)

Mel-skala naśladuje nieliniową czułość ludzkiego ucha na częstotliwości — jest gęstsza w niskich częstotliwościach, gdzie mowa niesie więcej informacji.

### Rejestracja (enrollment)

Każdy mówca jest wcześniej **zarejestrowany** w systemie. Podczas rejestracji z kilku nagrań treningowych wyznaczane są embeddingi, które następnie uśrednia się w jeden **profil biometryczny** (192-wymiarowy wektor). Profil zapisywany jest w bazie `voice_profiles.npy`.

### Weryfikacja i decyzja

Podczas weryfikacji:
1. Z nagrania testowego wyznaczany jest embedding.
2. Obliczane jest **podobieństwo cosinusowe** między embeddingiem testowym a profilem zarejestrowanej tożsamości.
3. Jeśli wynik ≥ próg (`threshold`), dostęp jest przyznawany (genuine); w przeciwnym razie odmawiany (impostor).

Podobieństwo cosinusowe mierzy kąt między dwoma wektorami:

```
cos(A, B) = (A · B) / (||A|| × ||B||)
```

Wartość 1.0 = wektory identyczne, 0.0 = prostopadłe, -1.0 = przeciwne.

---

## Struktura danych testowych

### Foldery

```
data/
  enrollment/<id_mówcy>/    — nagrania do rejestracji (wiele plików WAV)
  test_genuine/<id_mówcy>/  — nagrania testowe (pliki test_1.wav ... test_10.wav)
  background_sounds/        — pliki WAV z szumami tła (Eksperyment 5)
```

### Pliki testowe

Dla każdego mówcy w folderze `test_genuine`:
- `test_1.wav` do `test_5.wav` — **nagrania genuine** (prawdziwe próby; ten mówca faktycznie mówi i o sobie twierdzi)
- `test_6.wav` do `test_10.wav` — **nagrania impostora** (te pliki służą do symulacji włamania; głos danego mówcy jest podawany jako inny mówca)

### Budowanie prób testowych (trial)

Każda **próba (trial)** składa się z:
- `claimed_id` — tożsamość, którą system ma zweryfikować
- `audio_path` — ścieżka do nagrania testowego
- `label` — 1 = genuine (prawdziwy), 0 = impostor (podszywacz)

**Próba genuine**: mówca X twierdzi, że jest X i podaje swoje nagranie (test_1–5). System powinien wpuścić.

**Próba impostor**: nagranie mówcy X jest podane do weryfikacji tożsamości Y (losowo wybranego innego mówcy). System powinien odrzucić.

---

## Metryki oceny

Wszystkie eksperymenty korzystają z tego samego zestawu metryk zaimplementowanego w `metrics.py`.

### AUC (Area Under the ROC Curve)

Pole pod krzywą ROC (Receiver Operating Characteristic). Wartość od 0.5 (losowe zgadywanie) do 1.0 (perfekcyjny separator). Mierzy ogólną zdolność systemu do odróżnienia genuine od impostor, niezależnie od progu.

### EER (Equal Error Rate — Stopa błędów równych)

Punkt na krzywej, w którym **FAR = FRR** — system popełnia tyle samo błędów po jednej i drugiej stronie. Im niższy EER, tym lepszy system. Wyznaczany metodą interpolacji. Przy okazji wyznaczany jest **próg EER (EER threshold)** — wartość progu, przy której FAR ≈ FRR.

### FAR (False Acceptance Rate — Wskaźnik fałszywej akceptacji)

Odsetek prób impostorskich, które system **błędnie zaakceptował** (błąd II rodzaju, niebezpieczny z punktu widzenia bezpieczeństwa).

```
FAR = liczba impostorów wpuszczonych / wszystkie próby impostorskie
```

### FRR (False Rejection Rate — Wskaźnik fałszywego odrzucenia)

Odsetek prób genuine, które system **błędnie odrzucił** (błąd I rodzaju, wpływa na wygodę użytkownika).

```
FRR = liczba prawdziwych użytkowników odrzuconych / wszystkie próby genuine
```

### Wykresy

Dla każdego eksperymentu generowane są:
- **Rozkład wyników** — histogramy podobieństw dla genuine i impostor (powinny być rozdzielone)
- **Krzywa ROC** — FAR vs TAR (1-FRR) z zaznaczonym EER
- **Krzywa DET** — FAR vs FRR w skali logarytmicznej (typowa dla systemów biometrycznych)
- **FAR/FRR vs próg** — jak błędy zmieniają się w zależności od wartości progu

---

## Eksperyment 1

### Punkt odniesienia (Baseline)

**Plik:** `scripts/experiment_1.py`

#### Cel

Ustalenie bazowej wydajności systemu na czystych, niezakłóconych nagraniach. Wyniki tego eksperymentu stanowią punkt porównania dla wszystkich kolejnych eksperymentów.

#### Dobór danych

- Katalog testowy: `data/test_genuine/`
- Wszyscy mówcy w folderze (tylko ci, którzy mają profil w bazie)
- Pliki genuine: `test_1.wav` – `test_5.wav` (5 plików na mówcę)
- Pliki impostor: `test_6.wav` – `test_10.wav` (5 plików na mówcę, tożsamość losowana z puli innych mówców)
- Ziarno losowości: `RANDOM_SEED = 42` (gwarantuje powtarzalność)

#### Co się dzieje

System działa dokładnie tak jak podczas normalnej weryfikacji: wczytuje nagranie, extrahuje embedding i porównuje cosinusowo z profilem. Próg decyzyjny ustawiony na `0.25` (permisywny; w tym eksperymencie interesuje nas przede wszystkim wyznaczenie optymalnego progu EER).

#### Kluczowy wynik

Wyznaczony próg EER = **0.7301** — używany w kolejnych eksperymentach jako stały punkt odniesienia przy porównywaniu FAR/FRR między grupami.

---

## Eksperyment 2

### Skalowanie amplitudy

**Plik:** `scripts/experiment_2.py`

#### Cel

Zbadanie, jak zmiana **głośności (amplitudy)** nagrania wpływa na trafność weryfikacji. W praktyce użytkownik może nagrywać głos przy różnych odległościach od mikrofonu, z różnym poziomem wzmocnienia, lub sprzęt audio może różnie kalibrować sygnał.

#### Co to jest amplituda?

Amplituda sygnału audio to jego "siła" — wartości próbek oscylują wokół zera, a maksymalna odchyłka to amplituda. Przemnożenie całego sygnału przez stałą wartość (`factor`) skaluje amplitudę:
- `factor > 1` → głośniejszy sygnał
- `factor < 1` → cichszy sygnał
- `factor = 1` → bez zmian

Skalowanie amplitudy **nie zmienia treści językowej**, ale może wpływać na wyodrębniane cechy (np. log-Mel filterbanki są wrażliwe na bezwzględne wartości energii w oknach).

#### Dobór danych

- Te same próby co w Eksperymencie 1 (ta sama logika budowania prób)
- Każdej próbie losowo przypisywany jest jeden z trzech czynników skalowania: `[25, 1, 0.04]`
  - `×25` — sygnał 25× głośniejszy niż oryginał (bardzo głośno)
  - `×1` — oryginalna głośność (punkt kontrolny)
  - `×0.04` — sygnał 25× cichszy niż oryginał (bardzo cicho)
- Losowanie niezależne od wyboru prób (ziarno `RANDOM_SEED + 1`)

#### Co się dzieje

```python
factor = amp_rng.choice([25, 1, 0.04])
signal_scaled = signal * factor
test_emb = extract_embedding(signal_scaled)
```

Nagranie jest wczytywane jako macierz próbek (float), przemnażane przez czynnik, a następnie przetwarzane przez pipeline embeddingu.

#### Analiza wyników

Eksperyment mierzy wydajność osobno dla każdego czynnika (prefix `factor_25_`, `factor_1_`, `factor_0.04_`), a następnie rysuje wykres słupkowy FAR/FRR per czynnik przy progu z Eksperymentu 1.

---

## Eksperyment 3

### Zmiana częstotliwości próbkowania (Downsampling)

**Plik:** `scripts/experiment_3.py`

#### Cel

Sprawdzenie wpływu **obniżenia częstotliwości próbkowania** sygnału audio na jakość weryfikacji. W systemach telekomunikacyjnych (telefony, VoIP) audio często jest próbkowane z niższą częstotliwością niż standardowe 16 kHz.

#### Co to jest częstotliwość próbkowania?

Częstotliwość próbkowania (sample rate) mówi, ile razy na sekundę zapisywana jest wartość ciągłego sygnału analogowego. Standardem dla mowy jest **16 000 Hz (16 kHz)**, co oznacza 16 000 próbek na sekundę. Twierdzenie Nyquista mówi, że można wiernie odtworzyć częstotliwości do połowy częstotliwości próbkowania — przy 16 kHz do 8 kHz. Ludzki głos koncentruje się w paśmie 80–8000 Hz, więc 16 kHz jest wystarczające.

Obniżenie częstotliwości próbkowania (decimation) redukuje liczbę próbek przez wybranie co N-tej próbki lub użycie filtra interpolacji:
- Współczynnik `×2` → 16000 → **8000 Hz** (telefonia analogowa)
- Współczynnik `×5` → 16000 → **3200 Hz** (wąskopasmowa transmisja)
- Współczynnik `×10` → 16000 → **1600 Hz** (ekstremalna degradacja)

#### Dwie metody downsamplingu

##### Metoda naiwna (`naive_downsample`)
```python
signal[::factor]  # co N-ta próbka
```
Prosta selekcja próbek. **Nie stosuje filtru antyaliasingowego** — może powstawać aliasing (fałszywe składowe częstotliwościowe), gdy w sygnale są składowe powyżej nowej częstotliwości Nyquista.

##### Metoda właściwa (`proper_downsample`)
```python
scipy.signal.resample_poly(signal, 1, factor)
```
Polyphase resampling: przed decimacją stosuje **filtr dolnoprzepustowy** (anti-aliasing filter), który usuwa składowe powyżej nowej częstotliwości Nyquista. Daje wynik bez aliasingu.

#### Dobór danych

- **200 prób** losowo wybranych i przetasowanych (z całej puli prób genuine/impostor)
- Ziarno: `RANDOM_SEED = 42`
- **Te same 200 prób** przetestowane dla każdego czynnika i obu metod (uczciwe porównanie)

#### Dlaczego dla każdego współczynnika budowany jest nowy ekstraktor Mel?

Po downsamplingiem sygnał ma nową częstotliwość próbkowania — np. 8000 Hz zamiast 16000 Hz. Gdyby użyć starego ekstraktora (skonfigurowanego na 16000 Hz), interpretowałby on każdą próbkę jako trwającą 1/16000 s, podczas gdy w rzeczywistości trwa ona 1/8000 s. Skutki:

- `win_length = 400 próbek` przy 16 kHz = **25 ms** — poprawnie
- `win_length = 400 próbek` przy 8 kHz = **50 ms** — dwa razy za długie okno

Model był trenowany na cechach obliczonych z okien 25 ms / co 10 ms. Użycie okien o innych długościach czasowych oznacza, że cechy podawane do modelu mają inny charakter niż te, na których uczył się rozpoznawać mówców — porównanie embeddingów staje się nierzetelne.

Nowy ekstraktor przelicza okna ze **stałej wartości w sekundach** na liczbę próbek odpowiednią dla nowej częstotliwości:
```python
def build_mel_extractor(sample_rate):
    win_length = int(0.025 * sample_rate)   # zawsze 25 ms, niezależnie od SR
    hop_length = int(0.010 * sample_rate)   # zawsze 10 ms
    n_fft = max(win_length, 2 * 80)
    ...
```
Przy 8000 Hz: `win_length = 200 próbek` (0.025 × 8000), co znowu odpowiada 25 ms. Dzięki temu eksperyment mierzy tylko wpływ utraty informacji spowodowanej obniżeniem pasma — nie zaś artefakt wynikający ze złej konfiguracji ekstraktorów.

---

## Eksperyment 4

### Szum gaussowski (biały)

**Plik:** `scripts/experiment_4.py`

#### Cel

Zbadanie odporności systemu na **addytywny szum gaussowski (AWGN)** — najbardziej podstawowy model zakłóceń, który modeluje szum termiczny mikrofonu, szum kwantyzacji, lub ogólne zakłócenia elektroniczne.

#### Co to jest szum gaussowski?

Szum biały gaussowski to sygnał losowy, w którym każda próbka jest niezależnie losowana z rozkładu normalnego N(0, σ). Cechuje go:
- **Brak korelacji** między próbkami w czasie
- **Płaskie widmo** — jednakowa moc na wszystkich częstotliwościach
- Jest to matematyczny model "idealnych" zakłóceń

#### Co to jest SNR?

**SNR (Signal-to-Noise Ratio — stosunek sygnału do szumu)** to miara jakości sygnału:
```
SNR [dB] = 10 × log₁₀(moc_sygnału / moc_szumu)
```
Im wyższy SNR, tym czystszy sygnał:
- **40 dB** — bardzo czyste nagranie (szum ledwo słyszalny)
- **20 dB** — umiarkowane zakłócenia (wyraźnie słyszalny szum, ale mowa zrozumiała)
- **10 dB** — silny szum (mowa ledwo zrozumiała)

#### Generowanie szumu

```python
signal_power = np.mean(signal ** 2)
noise_power = signal_power / (10 ** (snr_db / 10))
noise = rng.normal(0, np.sqrt(noise_power), len(signal))
noisy_signal = signal + noise
```

Moc szumu jest wyliczana dynamicznie na podstawie mocy konkretnego nagrania, co zapewnia **spójny SNR** dla każdej próbki, niezależnie od głośności oryginału.

#### Dobór danych

- **100 prób** losowo wybranych i przetasowanych
- Te same próby dla każdego poziomu SNR (uczciwe porównanie)
- Szum generowany niezależnie dla każdej próby (różne realizacje losowe)

---

## Eksperyment 5

### Szumy środowiskowe (dźwięki tła)

**Plik:** `scripts/experiment_5.py`

#### Cel

Zbadanie odporności systemu na **rzeczywiste dźwięki tła** (środowiskowe), w odróżnieniu od szumu białego z Eksperymentu 4. W praktycznych zastosowaniach biometrycznych użytkownik może nagrywać głos w zatłoczonym biurze, kawiarni, komunikacji miejskiej itp.

#### Różnica względem Eksperymentu 4

| Cecha | Eksperyment 4 | Eksperyment 5 |
|---|---|---|
| Typ szumu | Gaussowski (biały) | Rzeczywiste nagrania tła |
| Widmo | Płaskie (wszystkie częstotliwości) | Nierównomiernie (dźwięki naturalne) |
| Korelacja w czasie | Brak | Obecna (np. muzyka, rozmowy) |
| Źródło | Generowane matematycznie | Pliki WAV z `data/background_sounds/` |

#### Poziomy SNR

```python
SNR_LEVELS_DB = [20, 10, 0]
```
Testowane wartości są **niższe niż w Eksperymencie 4** (bardziej ekstremalne warunki):
- **20 dB** — tło słyszalne, ale mowa dominuje
- **10 dB** — wyraźne zakłócenia tłem
- **0 dB** — tło i mowa mają **równą moc** (skrajna degradacja)

#### Mieszanie sygnałów

```python
def mix_at_snr(speech, background, snr_db):
    speech_power = np.mean(speech ** 2)
    bg_power = np.mean(background ** 2)
    scale = np.sqrt(speech_power / (bg_power * 10 ** (snr_db / 10)))
    return speech + scale * background
```

Tło jest skalowane tak, by osiągnąć zadany SNR względem mocy mowy. Jeśli nagranie tła jest za krótkie, jest **powtarzane** (tiled) do potrzebnej długości. Jeśli ma inną częstotliwość próbkowania, jest **resamplingowane** do 16 kHz.

#### Dobór danych

- **100 prób** losowo wybranych (te same dla każdego poziomu SNR)
- Do każdej próby **losowo wybierany jest plik tła** z katalogu `background_sounds/` (z powtórzeniami — ten sam plik może trafić do kilku prób)

---

## Eksperyment 6

### Kompresja audio kodekami stratymi

**Plik:** `scripts/experiment_6.py`

#### Cel

Ocena wpływu **kompresji stratnej** (lossy audio compression) na weryfikację mówcy. Przesyłanie głosu przez internet lub telefon prawie zawsze wiąże się z kompresją, która nieodwracalnie usuwa część informacji z sygnału.

#### Co to jest kompresja stratna?

Kodeki stratne (lossy codecs) zmniejszają rozmiar pliku audio kosztem jakości dźwięku. Działają na zasadzie **modelu psychoakustycznego** — usuwają składniki, które ludzkie ucho słabiej percypuje (np. ciche dźwięki maskowane przez głośniejsze). Wynik jest percepcyjnie podobny do oryginału, ale matematycznie różny — i ta różnica może wpłynąć na biometrię głosu.

#### Testowane kodeki

| Kodek | Rozszerzenie | Testowane przepływności |
|---|---|---|
| **MP3** (MPEG Layer III) | `.mp3` | 8, 32, 128 kbps |
| **AAC** (Advanced Audio Coding) | `.aac` | 8, 32, 128 kbps |
| **Opus** | `.opus` | 6, 16, 64 kbps |

**Bitrate (przepływność)** mierzy ilość danych na sekundę audio:
- Niski bitrate (np. 8 kbps) = silna kompresja, wyraźna degradacja jakości
- Wysoki bitrate (np. 128 kbps) = słaba kompresja, jakość zbliżona do oryginału
- MP3 128 kbps uważany jest za jakość "transparentną" dla zwykłych słuchaczy

Opus to nowoczesny kodek stosowany m.in. w WebRTC i Discord; jego niskie przepływności (6–16 kbps) są typowe dla transmisji głosowych.

#### FFmpeg

FFmpeg to darmowe, otwartoźródłowe narzędzie wiersza poleceń do przetwarzania multimediów — de facto standard w branży do konwersji, kompresji i edycji plików audio i wideo. W skrypcie wywoływany jest przez Pythona jako podproces (`subprocess.run`), a ścieżka do jego pliku wykonywalnego pobierana jest automatycznie przez bibliotekę `imageio_ffmpeg`, która dołącza FFmpeg jako zależność pakietu — nie trzeba go instalować ręcznie w systemie.

#### Procedura: kompresja i dekompresja

Każde nagranie przechodzi pełny cykl round-trip przez FFmpeg:
```
WAV ─[encode]─> MP3/AAC/Opus ─[decode]─> WAV (16 kHz, mono)
```
Wynikowy zdekodowany sygnał trafia do pipeline embeddingu. Symuluje to warunki, w których głos był transmitowany przez skompresowany kanał, a następnie odtworzony.

#### Dobór danych

- **100 prób** losowo wybranych (te same dla wszystkich kodeków i przepływności)
- Wyniki zapisywane per kodek i per przepływność (`mp3_128kbps_`, `aac_8kbps_` itp.)

---

## Eksperyment 7

### Pogłos akustyczny (Reverberation)

**Plik:** `scripts/experiment_7.py`

#### Cel

Zbadanie wpływu **pogłosu akustycznego** na weryfikację mówcy. Pogłos jest obecny w każdym zamkniętym pomieszczeniu — dźwięk odbija się od ścian, sufitu i podłogi, docierając do mikrofonu z opóźnieniem i tworząc charakterystyczne "echo".

#### Co to jest pogłos (reverberation)?

Gdy osoba mówi w pomieszczeniu, mikrofon odbiera nie tylko bezpośrednią falę dźwiękową, ale też fale odbite (reflections) od powierzchni. Sumaryczny efekt to **pogłos** — "rozciągnięcie" i "zamglenie" dźwięku.

#### Parametr T60

**T60** (reverberation time) to czas (w sekundach), po którym energia dźwięku maleje o **60 dB** (czyli do 1/1 000 000 wartości wyjściowej). Jest głównym parametrem opisującym akustykę pomieszczenia:

| T60 [s] | Przykładowe otoczenie |
|---|---|
| 0.2 | Mała, wytłumiona kabina nagraniowa |
| 0.5 | Typowy pokój mieszkalny z meblami |
| 1.0 | Sala konferencyjna, biuro open-space |
| 2.0 | Kościół, duże pomieszczenie |
| 5.0 | Hala sportowa, duży kościół |
| 8.0 | Katedra, tunel — wartość ekstremalnie wysoka |

#### Room Impulse Response (RIR)

Akustykę pomieszczenia modeluje się przez **odpowiedź impulsową pomieszczenia** (Room Impulse Response, RIR). W tym eksperymencie RIR generowane jest syntetycznie za pomocą modelu eksponencjalnego zaniku:

```python
def generate_rir(t60, sample_rate, rng):
    duration = t60 * 2
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    decay = np.exp(-6.91 * t / t60)     # wykładniczy zanik energii
    rir = rng.standard_normal(n) * decay  # szum biały przemnożony przez zanik
    rir /= np.max(np.abs(rir))           # normalizacja
    return rir
```

Wzór `exp(-6.91 × t / T60)` pochodzi z definicji T60: po czasie T60 amplituda maleje o 10^(-3) = -60 dB, co daje współczynnik `ln(10^(-3)) = -6.91`.

#### Nakładanie pogłosu

Pogłos nakładany jest przez **splot (konwolucję)** sygnału mowy z RIR:
```python
reverbed = fftconvolve(signal, rir, mode="full")[:len(signal)]
```
Konwolucja w dziedzinie czasu (tu: szybka metoda przez FFT) symuluje fizyczny proces odbić dźwięku w pomieszczeniu.

#### Dobór danych

- **100 prób** losowo wybranych (te same dla wszystkich wartości T60)
- Ten sam RIR dla wszystkich prób przy danym T60 (jeden RIR per wartość T60)
- Dla celów odsłuchowych generowane są przykładowe nagrania w `results/experiment_7/audio_samples/`

#### Dobór wartości T60

Zakres `[0.2, 0.5, 1.0, 2.0, 5.0, 8.0]` s celowo obejmuje zarówno warunki realistyczne (0.2–1.0 s), jak i ekstremalnie trudne (5.0–8.0 s), co pozwala prześledzić pełne spektrum degradacji wydajności systemu.

---

---

## Eksperyment 8

### Weryfikacja przy wysokim poziomie szumu (P3 — Wyzwanie 1)

**Plik:** `scripts/experiment_8.py`

#### Cel

Zbadanie wpływu silnego szumu gaussowskiego na skuteczność weryfikacji mówcy oraz ocena skuteczności algorytmicznej redukcji szumu jako metody poprawy odporności systemu. Eksperyment realizuje pierwsze wyzwanie specyfikacji P3: *"nagrania z dużym poziomem hałasu, tzn. takie, w których amplituda hałasu wynosi od 70% do 120% amplitudy sygnału"*.

#### Czym jest szum amplitudowy?

Szum gaussowski (biały) to losowe zakłócenie o rozkładzie normalnym dodawane do każdej próbki sygnału audio. W odróżnieniu od eksperymentu 4 (gdzie szum był parametryzowany przez SNR w dB), tutaj poziom szumu wyrażony jest przez **stosunek amplitudy szumu do amplitudy sygnału**:

```
noise_RMS = amplitude_ratio × signal_RMS
```

gdzie `RMS` (Root Mean Square) to miara głośności sygnału:

```
signal_RMS = sqrt(mean(signal²))
```

Przekładając na SNR (Signal-to-Noise Ratio):

```
SNR [dB] = 20 × log10(1 / amplitude_ratio)
```

| Amplitude ratio | SNR | Interpretacja |
|---|---|---|
| 70% | +3.1 dB | szum wyraźnie słyszalny, mowa zrozumiała |
| 85% | +1.4 dB | szum głośny, mowa utrudniona |
| 100% | 0.0 dB | szum równie głośny co sygnał |
| 110% | −0.8 dB | szum głośniejszy niż sygnał |
| 120% | −1.6 dB | dolna granica zrozumiałości mowy |

#### Jak szum jest dodawany do sygnału?

```python
def add_amplitude_noise(signal, amplitude_ratio, rng):
    signal_rms = np.sqrt(np.mean(signal ** 2))   # głośność oryginału
    noise = rng.normal(0, amplitude_ratio * signal_rms, len(signal))  # losowy szum
    return (signal + noise).astype(np.float32)   # suma: głos + szum
```

Parametr `rng` to generator liczb losowych z deterministycznym seedem — ten sam seed dla danego poziomu szumu gwarantuje że **sekcja bez denoizingu i sekcja z denoizingiem otrzymują identyczny zaszumiony sygnał**. To czysty ablation design: jedyną zmienną między sekcjami jest zastosowanie przetwarzania wstępnego.

#### Dobór danych

- Katalog testowy: `data/test_genuine/`
- 102 zarejestrowanych mówców
- Pliki genuine: `test_1.wav` – `test_5.wav` (473 triałów)
- Pliki impostor: `test_6.wav` – `test_10.wav` (457 triałów)
- **Łącznie: 930 triałów** × 5 poziomów szumu = 4650 trudnych próbek
- Seed szumu: `RANDOM_SEED + int(amplitude_ratio × 100)` — osobny seed per ratio

#### Sekcje eksperymentu

**Sekcja 1 — Baseline bez denoizingu**

Zaszumiony sygnał podawany bezpośrednio do ekstrakcji embeddingu. Pokazuje skalę degradacji systemu wraz ze wzrostem poziomu szumu. Przy SNR = 0 dB model ECAPA-TDNN wytrenowany na czystych nagraniach VoxCeleb nie jest w stanie skutecznie rozdzielić cech mówcy od szumu — wyznaczany embedding reprezentuje mieszaninę obu.

**Sekcja 2 — Redukcja szumu niestacjonarnego (`stationary=False`)**

```python
import noisereduce as nr
signal_denoised = nr.reduce_noise(y=signal, sr=16000, stationary=False)
```

Algorytm `noisereduce` działa na spektrogramie STFT sygnału:
1. Estymuje profil szumu adaptacyjnie w każdej ramce czasowej
2. Oblicza maskę spektralną — dla każdej częstotliwości i ramki ocenia stosunek szum/sygnał
3. Mnoży spektrogram przez maskę tłumiąc składowe zdominowane przez szum
4. Rekonstruuje sygnał przez odwrotną STFT

Tryb `stationary=False` aktualizuje profil szumu w czasie — lepiej radzi sobie z lokalną zmiennością amplitudy mowy.

**Sekcja 3 — Redukcja szumu stacjonarnego (`stationary=True`)**

Ten sam algorytm w trybie stacjonarnym — jeden globalny profil szumu estymowany dla całego nagrania. Testowany jako alternatywa, ponieważ biały szum gaussowski jest matematycznie stacjonarny. W praktyce okazał się mniej skuteczny niż tryb niestacjonarny.

#### Wyniki

| Poziom szumu | Baseline EER | NR non-stationary EER | NR stationary EER |
|---|---|---|---|
| Czyste audio (Exp. 1) | **7.82%** | — | — |
| 70% | 36.79% | **27.91%** | 32.14% |
| 85% | 40.04% | **32.98%** | 35.89% |
| 100% | 43.55% | **34.88%** | 38.95% |
| 110% | 43.33% | **38.05%** | 40.26% |
| 120% | 44.86% | **38.90%** | 40.70% |

#### Kluczowe obserwacje

- **Degradacja bez denoizingu:** EER wzrasta z 7.82% do 36–45% (+29–37pp). Przy progu operacyjnym 0.7301 FRR wzrasta do ~90–96% — system jest praktycznie bezużyteczny bez przetwarzania wstępnego.
- **Skuteczność NR non-stationary:** poprawa o 6–9pp EER względem baseline przy każdym poziomie szumu.
- **NR stationary gorszy niż NR non-stationary:** mimo że biały szum jest teoretycznie stacjonarny, tryb adaptacyjny lepiej radzi sobie z lokalną zmiennością mowy.
- **Wymaganie P3 (+5pp FRR):** nie zostało spełnione przy żadnym poziomie szumu — przy SNR ≤ 0 dB pełne przywrócenie jakości sygnału metodami post-processingu nie jest możliwe. Rozwiązanie istotnie ogranicza degradację systemu.

#### Pliki wynikowe

```
results/experiment_8/
├── baseline_{70,85,100,110,120}pct_metrics.txt     ← metryki bez denoizingu
├── enhanced_{70,85,100,110,120}pct_metrics.txt     ← metryki NR non-stationary
├── stationary_{70,85,100,110,120}pct_metrics.txt   ← metryki NR stationary
├── noise_frr_progression.png                        ← wykres liniowy FRR vs ratio
└── noise_frr_grouped_bars.png                       ← wykres słupkowy porównawczy
```

---

## Eksperyment 10

### Weryfikacja mówcy w dialogu wieloosobowym — diaryzacja (P3 — Wyzwanie 3)

**Plik:** `scripts/experiment_10.py`

#### Cel

Zbadanie wpływu obecności wielu mówców w nagraniu uwierzytelniającym na skuteczność systemu weryfikacji oraz ocena metod diaryzacji jako rozwiązania tego problemu. Eksperyment realizuje trzecie wyzwanie P3: *"Zamiast pojedynczego mówcy mamy do czynienia z dialogami lub rozmową grupy osób."*

#### Problem wielomówcowości

Klasyczny system weryfikacji ECAPA-TDNN zakłada jeden mówca na nagranie. Podany dialog powoduje że model wyznacza **jeden 192-wymiarowy embedding** uśredniający cechy akustyczne wszystkich obecnych mówców. Ten uśredniony wektor ma niższe podobieństwo cosinusowe do profilu referencyjnego claimed speakera — system częściej odrzuca prawdziwych użytkowników (wyższe FRR).

#### Konstrukcja dialogów testowych

Dialogi syntetyzowano metodą **konkatenacji sygnałów** według wzorca A-B-A:

```
dialogue = [pierwsza_połowa(A)] + [całe_nagranie(B)] + [druga_połowa(A)]
```

```python
def build_dialogue(seg_a1, seg_b, seg_a2):
    return np.concatenate([seg_a1, seg_b, seg_a2])
```

Dwa typy triałów:
- **Genuine:** claimed speaker A jest obecny w dialogu (~2/3 długości)
- **Impostor:** claimed speaker nie występuje w dialogu — dialog zawiera dwóch obcych mówców

**Zbiór testowy:** 861 triałów (437 genuine / 424 impostor) dla 102 mówców.

#### Sekcje eksperymentu

**Sekcja 1 — Brak diaryzacji (punkt odniesienia)**

Cały dialog traktowany jako jeden sygnał → jeden embedding → porównanie z profilem. Demonstruje problem: wyznaczony embedding nie reprezentuje dobrze żadnego z mówców.

**Sekcja 2 — Diaryzacja k-means na embeddingach ECAPA-TDNN**

Kluczowa obserwacja: segmenty tego samego mówcy tworzą skupiska w przestrzeni 192-wymiarowej embeddingów. Algorytm:

1. Podział dialogu na segmenty o długości **2 sekund**
2. Każdy segment → embedding ECAPA-TDNN (192-dim)
3. **K-means (k=2)** grupuje segmenty w 2 klastry odpowiadające 2 mówcom
4. Dla każdego klastra obliczane jest podobieństwo cosinusowe centroidu do profilu claimed speakera
5. Wybrany zostaje klaster o **najwyższym podobieństwie** do profilu
6. Segmenty wybranego klastra konkatenowane → jeden sygnał → weryfikacja

```python
km = KMeans(n_clusters=2, n_init=10, random_state=42)
cluster_labels = km.fit_predict(embeddings)

# Wybór klastra najbliższego claimed speakerowi
for c in range(2):
    centroid = embeddings[cluster_labels == c].mean(axis=0)
    sim = cosine_similarity(centroid, claimed_profile)
    if sim > best_sim:
        best_cluster = c

# Konkatenacja segmentów z wybranego klastra
extracted = np.concatenate([segments[i] for i in range(n) 
                            if cluster_labels[i] == best_cluster])
```

**Sekcja 3 — pyannote.audio (opcjonalna)**

Profesjonalny model diaryzacji (pyannote/speaker-diarization-3.1). Wymaga tokena HuggingFace i `pip install pyannote.audio`. W tym eksperymencie pominięta z powodu braku dostępu do tokena.

#### Wyniki

| Metoda | EER | AUC | Zmiana vs baseline |
|---|---|---|---|
| Czyste audio (Exp. 1) | **7.82%** | 0.9655 | — |
| Dialog bez diaryzacji | 23.11% | 0.8414 | +15.3pp |
| K-means diaryzacja | **12.36%** | **0.9319** | **+4.5pp ✅** |

#### Kluczowe obserwacje

- **Problem potwierdzony:** dialog bez diaryzacji powoduje wzrost EER o 15.3pp — embedding uśredniający głosy dwóch mówców wyraźnie degraduje weryfikację.
- **K-means skuteczny:** EER spada z 23.11% do 12.36% — poprawa o 10.75pp. Wzrost względem czystego baseline wynosi tylko **4.5pp**, co spełnia wymaganie P3 (+5pp).
- **AUC wzrasta z 0.84 do 0.93** — system bez diaryzacji częściowo traci zdolność dyskryminacji genuine/impostor; k-means ją przywraca.
- **Metoda nienadznorowana:** k-means nie wymaga żadnych dodatkowych danych ani etykiet — jedyną informacją jest profil biometryczny claimed speakera używany do wyboru klastra.

#### Pliki wynikowe

```
results/experiment_10/
├── no_diarization_metrics.txt            ← sekcja 1
├── kmeans_diarization_metrics.txt        ← sekcja 2
└── comparison_diarization_methods.png   ← wykres porównawczy
```

---

## Podsumowanie porównawcze eksperymentów

| Nr | Zakłócenie | Parametr | Wartości |
|---|---|---|---|
| 1 | Brak (baseline) | — | — |
| 2 | Skalowanie amplitudy | Czynnik skalowania | ×0.04, ×1, ×25 |
| 3 | Obniżenie częstotliwości próbkowania | Częstotliwość [Hz] | 8000, 3200, 1600 |
| 4 | Szum gaussowski | SNR [dB] | 40, 20, 10 |
| 5 | Dźwięki tła (rzeczywiste) | SNR [dB] | 20, 10, 0 |
| 6 | Kompresja kodekami | Kodek + bitrate [kbps] | MP3/AAC/Opus × kilka przepływności |
| 7 | Pogłos akustyczny | T60 [s] | 0.2, 0.5, 1.0, 2.0, 5.0, 8.0 |
| 8 | Szum amplitudowy (P3) | Amplitude ratio | 70%, 85%, 100%, 110%, 120% |
| 10 | Dialog wieloosobowy (P3) | Metoda diaryzacji | brak / k-means / pyannote |

Eksperymenty 8 i 10 używają tego samego progu EER = **0.7301** z Eksperymentu 1.
