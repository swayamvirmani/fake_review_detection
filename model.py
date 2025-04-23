import gradio as gr
import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === Setup ===
nltk.download('stopwords')
nltk.download('wordnet')

HYPE_WORDS = {
    'amazing', 'incredible', 'unbelievable', 'perfect', 'obsessed', 'must', 'now',
    'life-changing', 'crazy', 'awesome', 'insane', 'never', 'best', 'ever',
    'omg', 'buy', 'love', 'wow'
}

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(lemmatized)

def hype_score(text):
    return sum(word in HYPE_WORDS for word in text.lower().split())

# === Load & Prepare Data ===
df = pd.read_csv("updated_fake_reviews_dataset.csv")
df = df.rename(columns={"text_": "review"}) if "text_" in df.columns else df
df['label'] = df['label'].str.lower().map({'cg': 'fake', 'or': 'real', 'fake': 'fake', 'real': 'real'})
df.dropna(subset=["review", "label"], inplace=True)
df['review'] = df['review'].astype(str).apply(preprocess)
df = df[df['review'].str.strip() != ""]

# === Vectorize & Train Model ===
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['review'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=1000, class_weight='balanced')
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model = VotingClassifier(estimators=[('lr', lr), ('rf', rf)], voting='soft')
model.fit(X_train, y_train)

# === Accuracy ===
y_pred = model.predict(X_test)
model_accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

# === Prediction Logic ===
def analyze(text):
    cleaned = preprocess(text)
    if not cleaned.strip():
        return "❌ Invalid input", "Please enter meaningful text."

    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    confidence = max(model.predict_proba(vec)[0]) * 100
    hype = hype_score(text)

    result = "✅ Real" if prediction == "real" else "❌ Fake"
    extra = f"Confidence: {round(confidence, 2)}%\nHype Score: {hype}"

    if confidence < 70:
        extra += "\n⚠️ Low model confidence. Please verify manually."
    if prediction == "real" and hype >= 4:
        extra += "\n🚩 High hype detected in real review. Possibly suspicious."

    return result, extra

# === Gradio UI ===
with gr.Blocks(title="Fake Review Detector by Ayush & Vranda") as demo:
    gr.Markdown(
        f"""
        <div style='text-align:center; padding-bottom: 0.8em'>
            <h1 style="font-size: 2.2em; color: #333;">🧠 FAKE REVIEW DETECTOR</h1>
            <p style="font-size: 1.1em;">
                Paste a product review to check if it's <b style="color:green">Real</b> or <b style="color:red">Fake</b><br>
                <span style="color:gray;">🧪 Model Accuracy: <b>{model_accuracy}%</b> (Voting Classifier)</span>
            </p>
        </div>
        """
    )

    with gr.Row():
        review_input = gr.Textbox(
            lines=5,
            placeholder="📝 Type or paste a product review here...",
            label="Your Review"
        )

    predict_btn = gr.Button("🔍 Analyze Review", variant="primary")
    with gr.Row():
        label_output = gr.Text(label="🧾 Prediction")
        explain_output = gr.Text(label="📊 Details")

    predict_btn.click(analyze, inputs=review_input, outputs=[label_output, explain_output])

    gr.Markdown("---")

    gr.Markdown(
        """
        <div style="text-align: center; font-size: 0.9em; padding-top: 1em;">
            🛠️ Built with ❤️ by <b>Ayush Sharma</b> & <b>Vranda Garg</b>
        </div>
        """,
        elem_id="footer"
    )

demo.launch()