import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# === Config ===
nltk.download('stopwords')
nltk.download('wordnet')

HYPE_WORDS = {
    'amazing', 'incredible', 'unbelievable', 'perfect', 'obsessed', 'must', 'now', 'life-changing',
    'crazy', 'awesome', 'insane', 'never', 'best', 'ever', 'omg', 'buy', 'love', 'wow'
}

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# === Preprocessing ===
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(lemmatized)

def hype_score(text):
    return sum(word in HYPE_WORDS for word in text.lower().split())

# === Load Dataset ===
file_path = "updated_fake_reviews_dataset.csv"
if not os.path.exists(file_path):
    print("❌ Dataset not found.")
    exit()

df = pd.read_csv(file_path)
df = df.rename(columns={"text_": "review"}) if 'text_' in df.columns else df
df['label'] = df['label'].str.lower().map({'cg': 'fake', 'or': 'real', 'fake': 'fake', 'real': 'real'})
df.dropna(subset=["review", "label"], inplace=True)
df['review'] = df['review'].astype(str).apply(preprocess)
df = df[df['review'].str.strip() != ""]

# === TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), analyzer='word')
X = vectorizer.fit_transform(df['review'])
y = df['label']
tfidf_matrix = vectorizer.transform(df['review'])

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Models ===
lr = LogisticRegression(max_iter=1000, class_weight='balanced')
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

model = VotingClassifier(estimators=[('lr', lr), ('rf', rf)], voting='soft')
print("⚙️ Training...")
model.fit(X_train, y_train)

# === Evaluation ===
y_pred = model.predict(X_test)
print("\n✅ Evaluation")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print(classification_report(y_test, y_pred))

# === Prediction Loop ===
print("\n💬 Type a product review to check. Type 'exit' to quit.\n")

while True:
    review_input = input("📝 Review: ")
    if review_input.strip().lower() == 'exit':
        print("👋 Exiting.")
        break

    processed = preprocess(review_input)
    if not processed.strip():
        print("⚠️ Review is empty after preprocessing.\n")
        continue

    review_vector = vectorizer.transform([processed])
    prediction = model.predict(review_vector)[0]
    confidence = max(model.predict_proba(review_vector)[0]) * 100
    hype = hype_score(review_input)

    print(f"🔍 Prediction: {'✅ Real' if prediction == 'real' else '❌ Fake'} (Confidence: {round(confidence, 2)}%)")
    print(f"✨ Hype Words: {hype}")

    if confidence < 70:
        print("⚠️ Low model confidence. Review may require human check.")
    if prediction == 'real' and hype >= 4:
        print("🚩 High hype detected in 'Real' prediction. Possibly suspicious.")

    print()