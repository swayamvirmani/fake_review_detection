# 🧠 Fake Review Detection
A machine learning-powered web app that classifies e-commerce product reviews as **Real** or **Fake** with high confidence, hype score tracking, and user-friendly Gradio interface.

> 🔍 Built with ❤️ by **Ayush Sharma** and **Vranda Garg**

---

## 🚀 Features

- ✅ Real vs Fake classification using VotingClassifier (LogReg + Random Forest)
- 📊 Shows confidence score
- ✨ Flags overly hyped reviews
- ⚠️ Highlights low-confidence predictions
- 🌐 Interactive UI using Gradio
- 📎 Clean and ready for live showcase / demo day

---

## 📸 Demo Screenshots

### Main Interface
![Fake Review Detector UI](/demo.png)

### Result Examples
| Real Review (High Confidence) | Fake Review (High Confidence) | Hyped Review Detection |
|:----------------------------:|:-----------------------------:|:----------------------:|
| ![Real Review Example](/real.png) | ![Fake Review Example](/fake.png) | ![Hyped Review Example](/low.png) |

---

## 🔧 Tech Stack

- `Python 3.10+`
- `Gradio` for frontend
- `Scikit-learn` for ML models
- `NLTK` for NLP
- `Pandas` for preprocessing

---

## 📂 Project Structure

```
FakeReviewDetection/
├── model.py   # Main Gradio app
├── updated_fake_reviews_dataset.csv       # Labeled reviews dataset
├── requirements.txt                       # Package dependencies
└── README.md                              # This file
```

---

## 🛠️ How to Run (via Conda)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/CyberBoyAyush/FakeReviewDetection.git
cd FakeReviewDetection
```

### 2️⃣ Create and Activate Conda Environment

```bash
conda create -n fake-review-env python=3.10
conda activate fake-review-env
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Launch the App

```bash
python model.py
```

Then open the local link in your browser (usually [http://127.0.0.1:7860](http://127.0.0.1:7860)).

---

## 📄 Dataset

The `updated_fake_reviews_dataset.csv` file contains labeled product reviews (CG = fake, OR = real), cleaned and mapped for model training. Make sure this file is in the same folder as the script.

---

## 🧪 Sample Reviews

Try these in the app:

```
Setup was smooth and the device works as expected. Would buy again.
OMG THIS CHANGED MY LIFE!!! I’ve told EVERYONE. Get it NOW!!!
Simple and effective — no complaints. Quiet and reliable.
```

---

## 📌 Authors

- 👨‍💻 Ayush Sharma [@CyberBoyAyush](https://github.com/CyberBoyAyush)
- 👩‍💻 Vranda Garg

---

## 📢 License

This project is for educational/demo purposes. No commercial use or resale of dataset permitted.

---

## ⭐️ Show Some Love

If this project helped you, give it a ⭐️ on GitHub!
