# Rotor Imbalance Detection with a CRNN and Physics-Based Features

In this project, we use a CRNN model to identify how much imbalance weight is attached to a rotor, using vibration and microphone signals. We give the model physics-based features from signal processing (order tracking and synchronous demodulation with a tachometer) next to the spectrograms. We also test whether adding a microphone to the vibration sensors helps, especially when one sensor is noisy.

**Main result** (8 weights, 1 s of signal, 72 test recordings): the vibration model reaches **89.5% accuracy** and keeps **83.5%** when the vibration signal has very strong noise (0 dB SNR). The best model on clean data (early fusion) reaches **90.7%**, and **98% of its predictions are within ±5 g** of the true weight.

---

## Data

We use the **MAFAULDA** dataset (Machinery Fault Database, UFRJ), from its Kaggle mirror ([uysalserkan/fault-induction-motor-dataset](https://www.kaggle.com/datasets/uysalserkan/fault-induction-motor-dataset)). We did not record this data.

- 382 recordings: normal (0 g) and seven imbalance weights (6, 10, 15, 20, 25, 30, 35 g) → **8 classes**
- Each recording is 5 s at 50 kHz, at one constant speed between about 12 and 60 Hz
- 8 channels: a tachometer (one pulse per revolution), two 3-axis accelerometers on the bearings, and one microphone

---

## Method

### 1. Split by recording
Windows cut from the same recording are almost identical, so splitting by window would leak information into the test set. We split by **recording**: for each weight, we sort the recordings by speed and assign them in turn to train, validation and test (3 : 1 : 1). Every set contains all weights and the whole speed range (235 / 75 / 72 recordings). Windows are cut after the split, and normalization uses training statistics only.

### 2. Preprocessing
- Remove drift (detrend), apply an anti-aliasing filter, and downsample: vibration to 5 kHz, microphone to 25 kHz
- Cut each recording into 1 s windows (0.5 s overlap), 9 windows per recording

### 3. Speed and angle from the tachometer
The tachometer gives one pulse per revolution, so we calculate the rotor speed and the rotor angle at every sample.

While checking this, we found that the speeds in the file names are about **2.4% higher** than the measured speed. They are rounded to steps of 0.2048 Hz (= 51,200 / 250,000), which means they were calculated for a 51.2 kHz sampling rate, not 50 kHz. We use the **measured** speed everywhere.

### 4. Two types of input
1. **Spectrograms (STFT):** they show the general vibration pattern across frequencies and how it changes over time. For vibration, we use **order tracking**: the frequency axis is converted to multiples of the rotor speed (1×, 2×, 3×), so the peaks stay in the same place at every speed.
2. **Synchronous demodulation features:** using the rotor angle from the tachometer, we calculate the **amplitude and phase** of the vibration at 1×, 2× and 3× the rotor speed for each channel. Amplitude tells us how strong the vibration is; phase tells us at which rotor angle it peaks. This is the same measurement used to balance rotors in industry.

<img width="978" height="605" alt="speedbeforeafter" src="https://github.com/user-attachments/assets/3634536c-84af-4713-b4c8-e1af3718924a" />


### 5. Model
- **CRNN:** 4 convolution blocks read the spectrogram, and a GRU reads it over time. The synchronous features and the speed join the GRU at every time step and the final layers.
- **Two fusion types:**
  - **Early fusion:** vibration and microphone go into one model together.
  - **Late fusion:** one model for vibration, one for the microphone; their output probabilities are combined with a weighted average (the weight is chosen on the validation set).
- **Noise augmentation:** during training, 25% of windows get noise on the vibration and 25% on the microphone (0 to 30 dB SNR).
- **Training:** early stopping on validation loss, 3 random seeds, and the 3 models are averaged (ensemble).

### 6. Evaluation
- **Physics baselines** (no neural network) to check whether deep learning is needed
- **95% confidence intervals** by bootstrap over the test recordings
- **Paired comparisons** between models on the same test recordings
- **Noise test:** noise added to one sensor at a time, at 20, 10 and 0 dB SNR

---

## Results

Test set, 72 recordings (648 windows). CRNN results are the ensemble of 3 seeds. Accuracy with 95% confidence interval (bootstrap over test recordings).

| Model | Clean | Vibration noise 0 dB | Microphone noise 0 dB |
|---|---|---|---|
| Physics baseline: 1× amplitude + speed | 0.793 [0.72, 0.87] | – | – |
| Physics baseline: synchronous features (gradient boosting) | 0.847 [0.77, 0.92] | – | – |
| CRNN, vibration only | 0.895 [0.84, 0.95] | **0.835** [0.77, 0.89] | 0.895 |
| CRNN, microphone only | 0.344 [0.27, 0.42] | – | 0.304 |
| CRNN, early fusion | **0.907** [0.86, 0.96] | 0.778 [0.71, 0.85] | 0.883 |
| CRNN, late fusion | 0.898 [0.85, 0.95] | 0.827 [0.77, 0.88] | 0.890 |

<img width="886" height="407" alt="physicsbaselinesgrayvscrnn" src="https://github.com/user-attachments/assets/705bc688-8442-4abb-b365-9b46629a1752" />

<img width="1203" height="477" alt="noisevibrationmicrophone" src="https://github.com/user-attachments/assets/aad5a036-4a6e-40f6-a992-1cdf8faef78d" />


### What we found
1. **The CRNN is better than the simple physics baseline** (+10 points, statistically significant). Compared with the best physics baseline, it is +5 points, but this difference is not statistically significant with 72 test recordings.
2. **Synchronous demodulation features helped.** They raised the best physics baseline from 0.79 to 0.85. In the CRNN, adding them together with the measured speed (v3 → v4) improved accuracy under 0 dB vibration noise from 0.75 to 0.84. These are two separate runs, so this comparison is not a paired statistical test.
3. **The microphone did not help.** Alone it reaches 34% (chance is 12.5%), but adding it to vibration did not improve accuracy significantly: late fusion changed it by less than 1 point, even with noisy vibration.
4. **Early fusion is less robust.** With 0 dB noise on the vibration, early fusion was 5.7 points worse than vibration alone (statistically significant). When sensors are mixed inside one model, noise on one sensor affects the whole model.
5. **The noise augmentation worked.** Without it (v2), 10 dB of vibration noise dropped the vibration model's accuracy to 0.47. With it, accuracy stays at 0.84 (v3) and 0.86 (v4) (mean of 3 seeds).

<img width="979" height="450" alt="alldifferences" src="https://github.com/user-attachments/assets/bf0647d1-4dd1-48ae-8764-7030adc080b6" />


### Accuracy per weight
<img width="1096" height="494" alt="earlylatetrueclass" src="https://github.com/user-attachments/assets/6f8d731c-cb71-4b57-aa42-8a43aaec1e30" />


- With late fusion, 0, 6, 10, 30 and 35 g are detected at 93–100%, and 15 g at 89%.
- Most errors are between **20 g and 25 g**, and almost all errors are to the neighboring weight. For both fusion models, 98% of predictions are within ±5 g, and the average error is about 0.6 g.
- Averaging the 9 windows of each recording barely changed accuracy (vibration model: 0.895 → 0.903). This suggests the errors come mostly from whole recordings, not from single windows.

---

## How the project developed

| Version | Change | Vibration CRNN accuracy (mean of 3 seeds) |
|---|---|---|
| v1 | Spectrograms at 5 kHz | 0.814 |
| Check | Microphone at 25 kHz instead of 5 kHz: the microphone went from 19% to 33%, so its information is above 1 kHz | – |
| v2 | Speed as an input | 0.833 |
| v3 | Order tracking, noise augmentation, ensemble, confidence intervals | 0.844 |
| v4 | Synchronous demodulation, measured speed | **0.874** |

---

## Deployment size

| Model | Parameters | Size (float32) | Time per 1 s window (CPU) |
|---|---|---|---|
| Vibration only | 203k | 795 KB | 4.8 ms |
| Early fusion | 204k | 795 KB | 4.5 ms |

Times are for the model only (batch of 1, office PC CPU); computing the spectrogram and the synchronous features adds a few milliseconds. The model is small and runs in about 0.5% of real time, so it is realistic for an edge device. The int8 size (about 200 KB) is an estimate; quantization was not tested.

---

## Limitations
- **20 g vs 25 g:** each weight appears to come from a single mounting, so differences in mounting cannot be separated from the effect of the weight. This is the most likely reason these two weights are confused.
- **One lab machine:** the results are not tested on other machines. The phase values also depend on how the sensors and the tachometer are mounted.
- **Small test set:** 72 recordings give wide confidence intervals (about ±5 points).
- **Sampling rate:** the dataset describes 50 kHz, but the file names suggest 51.2 kHz. If 51.2 kHz is correct, all frequencies in Hz are 2.4% too low. Classification is not affected.
- **Not tested separately:** the effect of order tracking alone, and a plain CRNN without any physics inputs.

---

## Files
- `download_mafaulda.py`: downloads the normal and imbalance recordings from Kaggle and saves them as `.npy` files with a manifest
- `project2_pipeline_v4.py`: the full pipeline (split, preprocessing, features, training, evaluation, figures)
- `test_1x_vector.py`: the first test of synchronous demodulation features, on the validation set
- `figures/`: all figures in this README

**Requirements:** Python 3, NumPy, SciPy, pandas, scikit-learn, PyTorch, Matplotlib.

