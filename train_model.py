import os
import re
import string
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Download stopwords if not already downloaded
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text.strip()

# Load IMDB dataset from given directory (expects 'pos' and 'neg' subfolders)
def load_imdb_data(data_dir):
    data = {"review": [], "sentiment": []}
    for sentiment in ["pos", "neg"]:
        folder_path = os.path.join(data_dir, sentiment)
        for filename in os.listdir(folder_path):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                review = file.read()
                data["review"].append(clean_text(review))
                data["sentiment"].append(1 if sentiment == "pos" else 0)
    return pd.DataFrame(data)

# Set paths (modify these paths to your local directories)
train_path = r"aclImdb\train"
test_path = r"aclImdb\test"

# Load datasets
df_train = load_imdb_data(train_path)
df_test = load_imdb_data(test_path)

# Remove duplicates & missing values
df_train.drop_duplicates(inplace=True)
df_train.dropna(inplace=True)
df_test.drop_duplicates(inplace=True)
df_test.dropna(inplace=True)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(6, 4))
sns.countplot(x=df_train["sentiment"])
plt.title("Class Distribution (Train Data)")
plt.show()

print("\nDataset Info:")
print(df_train.info())
print("\nSentiment Distribution:\n", df_train["sentiment"].value_counts())

# Split train data (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(df_train["review"], df_train["sentiment"], test_size=0.2, random_state=42)

# TF-IDF Vectorization (for traditional models)
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(df_test["review"])

# Handle class imbalance using SMOTE (for TF-IDF based training)
# Note: We create new variables so that the original y_train remains unchanged for LSTM
smote = SMOTE(random_state=42)
X_train_tfidf_res, y_train_tfidf_res = smote.fit_resample(X_train_tfidf, y_train)

# Train Logistic Regression with GridSearchCV
param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000, class_weight="balanced"), param_grid, cv=3, scoring="accuracy")
grid_search.fit(X_train_tfidf_res, y_train_tfidf_res)
best_lr_model = grid_search.best_estimator_

# Train Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf_res, y_train_tfidf_res)

# Tokenization & Padding for LSTM (using original X_train and X_val)
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=200, padding="post")
X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=200, padding="post")
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(df_test["review"]), maxlen=200, padding="post")

# Build and Train LSTM Model
lstm_model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=200),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])
lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
lstm_history = lstm_model.fit(X_train_seq, y_train, epochs=5, batch_size=64, validation_data=(X_val_seq, y_val))

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# Initialize list to store model evaluation results
comparison_results = []

# Evaluate Logistic Regression & Naive Bayes
models = {"Logistic Regression": best_lr_model, "Naive Bayes": nb_model}
for name, model in models.items():
    y_val_pred = model.predict(X_val_tfidf)
    y_test_pred = model.predict(X_test_tfidf)
    
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(df_test["sentiment"], y_test_pred)
    comparison_results.append({"Model": name, "Validation Accuracy": val_acc, "Test Accuracy": test_acc})
    
    # Save classification report
    with open(f"{name}_classification_report.txt", "w") as f:
        f.write(f"Validation Accuracy: {val_acc}\n")
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write("Validation Report:\n")
        f.write(classification_report(y_val, y_val_pred))
        f.write("\nTest Report:\n")
        f.write(classification_report(df_test["sentiment"], y_test_pred))
    
    # Plot confusion matrix for this model
    plot_confusion_matrix(y_val, y_val_pred, name)

# Evaluate LSTM Model
lstm_val_pred = (lstm_model.predict(X_val_seq) > 0.5).astype("int32")
lstm_test_pred = (lstm_model.predict(X_test_seq) > 0.5).astype("int32")
lstm_val_acc = accuracy_score(y_val, lstm_val_pred)
lstm_test_acc = accuracy_score(df_test["sentiment"], lstm_test_pred)
comparison_results.append({"Model": "LSTM", "Validation Accuracy": lstm_val_acc, "Test Accuracy": lstm_test_acc})

# Save LSTM Classification Report
with open("LSTM_classification_report.txt", "w") as f:
    f.write(f"Validation Accuracy: {lstm_val_acc}\n")
    f.write(f"Test Accuracy: {lstm_test_acc}\n")
    f.write("Validation Report:\n")
    f.write(classification_report(y_val, lstm_val_pred))
    f.write("\nTest Report:\n")
    f.write(classification_report(df_test["sentiment"], lstm_test_pred))

# Plot confusion matrix for LSTM
plot_confusion_matrix(y_val, lstm_val_pred, "LSTM")

# Plot LSTM training accuracy & loss curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(lstm_history.history["accuracy"], label="Train Accuracy")
plt.plot(lstm_history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("LSTM Model Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lstm_history.history["loss"], label="Train Loss")
plt.plot(lstm_history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("LSTM Model Loss")
plt.legend()
plt.show()

# Save Comparison Table as CSV
comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv("Accuracy-Reprts/model_comparison_results.csv", index=False)
print("\nComparison Results:\n", comparison_df)

# Save Models & Vectorizer
joblib.dump(best_lr_model, "Models/logistic_regression_model.pkl")   # Saved using Joblib
joblib.dump(nb_model, "Models/naive_bayes_model.pkl")                  # Saved using Joblib
joblib.dump(vectorizer, "Models/tfidff_vectorizer.pkl")                 # Saved using Joblib

# Save tokenizer as JSON
tokenizer_json = tokenizer.to_json()
with open("Models/tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer_json)

# Save LSTM model using Keras native save (HDF5 format)
lstm_model.save("Models/lstm_sentiment_model.h5")

print("\n✅ All models, reports, plots, and comparison results saved successfully!")          