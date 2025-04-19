# hw3-assign

Student Name : Yeshwanth Narige   
ID : 700764035

# Deep‑Learning Examples

This repository contains four self‑contained TensorFlow Keras projects demonstrating:

1. **Basic Autoencoder** for MNIST reconstruction  
2. **Denoising Autoencoder** for noise removal on MNIST  
3. **LSTM‑based Text Generation** (Shakespeare)  
4. **Sentiment Classification** (IMDB reviews)  

---

## 📁 Project Structure

```
.
├── ae_basic/               # Q1: Basic Autoencoder  
│   ├── train.py            # definition & training script  
│   └── visualize.py        # original vs. reconstructed  
│
├── ae_denoise/             # Q2: Denoising Autoencoder  
│   ├── train.py            # noisy→clean training  
│   └── compare.py          # noisy vs. denoised vs. clean  
│
├── rnn_textgen/            # Q3: Text Generation with LSTM  
│   ├── train.py            # data prep & model training  
│   └── generate.py         # sampling with temperature  
│
└── rnn_sentiment/          # Q4: Sentiment Classification  
    ├── train.py            # IMDB data prep & model  
    └── evaluate.py         # confusion matrix & report  
```

---

## 🔧 Requirements

- Python 3.8+  
- TensorFlow 2.x  
- NumPy, Matplotlib  
- scikit‑learn (for Q4)  

Install via:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

---

## 1. Basic Autoencoder (Q1)

**Objective:** Encode 28×28 MNIST images to a 32‑dim latent vector and reconstruct them.

- **Architecture**  
  - Encoder: 784 → 32 (ReLU)  
  - Decoder: 32 → 784 (Sigmoid)  
- **Loss:** Binary cross‑entropy  
- **Usage:**  
  ```bash
  cd ae_basic
  python train.py      # trains model on MNIST
  python visualize.py  # plots original vs. reconstructed
  ```
- **Insights:**  
  - Smaller latent dims (16) blur details  
  - Larger dims (64) give sharper reconstructions but risk overfitting  

---

## 2. Denoising Autoencoder (Q2)

**Objective:** Train the same 784 – 32 – 784 autoencoder, but feed it **noisy** MNIST and teach it to recover the **clean** images.

- **Noise:** Gaussian, μ=0, σ=0.5  
- **Training:** Noisy inputs → Clean targets  
- **Usage:**  
  ```bash
  cd ae_denoise
  python train.py      # trains denoising AE & basic AE
  python compare.py    # visualize noisy vs. denoised vs. basic AE
  ```
- **Comparison:**  
  - Basic AE reproduces noise if fed noisy input  
  - Denoising AE learns to filter out noise, producing cleaner digits  

---

## 3. LSTM Text Generation (Q3)

**Objective:** Train a character‑level LSTM on Shakespeare’s sonnets to predict the next character, then sample new text.

- **Data:** `shakespeare.txt` (downloaded automatically)  
- **Model:**  
  1. Embedding → 256-d  
  2. LSTM → 1024 units  
  3. Dense → vocab size  
- **Generation:**  
  - Seed with a prompt (e.g. “ROMEO: “)  
  - Temperature scaling to control creativity  
- **Usage:**  
  ```bash
  cd rnn_textgen
  python train.py       # trains the RNN (20 epochs)
  python generate.py    # sample new text with chosen temperature
  ```
- **Tip:**  
  - **T<1** → more conservative, repetitive  
  - **T>1** → more random, creative (but risk gibberish)  

---

## 4. Sentiment Classification (Q4)

**Objective:** Classify IMDB movie reviews as positive or negative using an LSTM.

- **Data:** `tensorflow.keras.datasets.imdb` (top 10 000 words)  
- **Preprocessing:** pad/truncate each review to 200 tokens  
- **Model:**  
  1. Embedding → 128‑d  
  2. LSTM → 64 units + Dropout  
  3. Dense → 1 (sigmoid)  
- **Evaluation:**  
  - Confusion matrix  
  - Precision, recall, F1‑score via `sklearn.metrics`  
- **Usage:**  
  ```bash
  cd rnn_sentiment
  python train.py      # trains & validates model
  python evaluate.py   # computes confusion matrix & report
  ```
- **Precision–Recall Trade‑off:**  
  - **Precision**: Of predicted positives, how many are true?  
  - **Recall**: Of true positives, how many did we catch?  
  - Adjusting the decision threshold lets you balance false positives vs. false negatives depending on application needs.

---

## 📖 References

- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)  
- [Imdb Dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)  
 
