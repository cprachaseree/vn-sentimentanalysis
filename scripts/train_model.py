from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import pickle
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))

def load_dataset(data_dir=os.path.join(script_dir, "..", "data", "sentiment_analysis_viet_raw"), file_parts=["negative_data.csv", "neutral_data.csv", "positive_data.csv"]):
    train_texts, val_texts = [], []
    train_labels, val_labels = [], []
    for file_part in file_parts:
        file_path = os.path.join(data_dir, file_part)
        df = pd.read_csv(file_path).dropna()
        X = list(df["Review"])
        y = list(df["Label"].apply(lambda x : x+1))
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
        train_texts.extend(X_train)
        val_texts.extend(X_val)
        train_labels.extend(y_train)
        val_labels.extend(y_val)
    return train_texts, val_texts, train_labels, val_labels

def tokenize_sent(sent):
    return ViTokenizer.spacy_tokenize(sent)[0]

def get_vectorizer(train_texts):
    vectorizer = TfidfVectorizer(tokenizer=tokenize_sent, ngram_range=(1,1))
    vectorizer.fit(train_texts)
    return vectorizer

def get_feats(vectorizer, texts):
    return vectorizer.transform(texts)

def train_classifier(train_feats, train_labels, val_feats, val_labels, classifier=None):
    if classifier == "nb":
        classifier = MultinomialNB()
    else:
        classifier = SVC()
    classifier.fit(train_feats, train_labels)
    predictions = classifier.predict(val_feats) 
    print("Accuracy: ", accuracy_score(val_labels, predictions))
    print("Confusion:\n", confusion_matrix(val_labels, predictions))
    print(classification_report(val_labels, predictions))
    return classifier

def save_model(vectorizer, classifier, output_dir=os.path.join(script_dir, "models")):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    pickle.dump(vectorizer, open(os.path.join(output_dir, "vectorizer.pickle"), "wb"))
    pickle.dump(classifier, open(os.path.join(output_dir, "classifier.pickle"), "wb"))

def main():
    train_texts, val_texts, train_labels, val_labels = load_dataset()
    vectorizer = get_vectorizer(train_texts)
    train_feats = get_feats(vectorizer, train_texts)
    
    val_feats = get_feats(vectorizer, val_texts)
    if len(sys.argv) == 2 and sys.argv[1] == "nb":
        model = "nb"
    else:
        model = None
    classifier = train_classifier(train_feats, train_labels, val_feats, val_labels, classifier=model)
    save_model(vectorizer, classifier)

if __name__ == "__main__":
    main()
