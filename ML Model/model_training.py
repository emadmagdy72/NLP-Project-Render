# model_training.py

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from data_preprocessing import preprocess_text

def train_model(df):
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['text'])
    y = df['dialect']

    le = LabelEncoder()
    y = le.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    with open('tfidf.pkl', 'wb') as tfidf_file:
        pickle.dump(tfidf, tfidf_file)
    
    with open('label_encoder.pkl', 'wb') as le_file:
        pickle.dump(le, le_file)
    
    return accuracy

def load_model():
    model = pickle.load(open('model.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    le = pickle.load(open('label_encoder.pkl', 'rb'))
    return model, tfidf, le

def predict(text, model, tfidf, le):
    text = preprocess_text(text)
    text_transformed = tfidf.transform([text])
    prediction = model.predict(text_transformed)
    return le.inverse_transform(prediction)[0]
