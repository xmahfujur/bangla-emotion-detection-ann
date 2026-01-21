# bangla-emotion-detection-ann
# 🇧🇩 Bengali Text Emotion Detection using ANN

An Artificial Neural Network (ANN) based Natural Language Processing (NLP) project that detects emotions from **Bengali (Bangla) text**.  
The system classifies text into emotions such as **Happy, Sad, Angry, and Neutral** using **TF-IDF features** and a **feedforward neural network**.

---

## 📌 Features
- ✅ Emotion detection from Bangla text
- ✅ ANN-based multiclass classification
- ✅ Supports multiple emotions
- ✅ Clean preprocessing for Bengali language
- ✅ Easy-to-extend architecture

---

## 🧠 Emotions Supported
- 😊 Happy  
- 😢 Sad  
- 😠 Angry  
- 😐 Neutral  

---

## 🧩 Project Workflow
```

Bengali Text
↓
Text Cleaning & Normalization
↓
TF-IDF Vectorization
↓
Artificial Neural Network (ANN)
↓
Predicted Emotion

```

---

## 🛠 Tech Stack
- **Python**
- **TensorFlow / Keras**
- **Scikit-learn**
- **NLTK**
- **Pandas & NumPy**
- **Matplotlib**

---

## 📂 Project Structure
```

bangla-emotion-detection-ann/
│
├── data/
│   └── bangla_emotion.csv
│
├── preprocess.py
├── train_ann.py
├── evaluate.py
├── predict.py
│
├── requirements.txt
└── README.md

````

---

## 📊 Dataset
The dataset consists of labeled Bengali text sentences with emotion categories.

**Sample format:**
```csv
text,emotion
আজ আমি খুব খুশি,Happy
মনটা আজ খুব খারাপ,Sad
সে আমাকে খুব রাগিয়ে দিয়েছে,Angry
আজ আবহাওয়া স্বাভাবিক,Neutral
````

You can use:

* Public Bengali emotion datasets
* Or create your own labeled dataset (recommended)

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/bangla-emotion-detection-ann.git
cd bangla-emotion-detection-ann
```

### 2️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 🔹 Train the Model

```bash
python train_ann.py
```

### 🔹 Evaluate the Model

```bash
python evaluate.py
```

### 🔹 Predict Emotion

```bash
python predict.py
```

**Example Input:**

```
আজ মনটা খুব খারাপ
```

**Output:**

```
Emotion: Sad 😢
```

---

## 📈 Model Architecture

* Input Layer (TF-IDF Features)
* Hidden Layer (128 neurons, ReLU)
* Hidden Layer (64 neurons, ReLU)
* Output Layer (Softmax)

**Optimizer:** Adam
**Loss Function:** Categorical Cross-Entropy

---

## 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 🎓 Learning Outcomes

* Bengali NLP preprocessing
* Feature extraction using TF-IDF
* ANN model design and training
* Multiclass emotion classification
* Model evaluation and deployment basics

---

## 🔮 Future Improvements

* 🔄 Upgrade ANN to LSTM or Transformer
* 🌐 Build a Web App (Flask / Streamlit)
* 📱 Mobile app integration
* 🧠 More emotion classes
* 📊 Larger dataset for higher accuracy

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!
Feel free to fork this repository and submit a pull request.

---

## 📜 License

This project is for **educational and research purposes only**.

---

## 👤 Author

**Mahfujur Rahman**
Aspiring AI Engineer | University Student
🇧🇩 Bangladesh
