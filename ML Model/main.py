# main.py

from data_fetching import fetch_data
from data_preprocessing import preprocess_dataframe
from model_training import train_model, load_model, predict

def main():
    database_file = 'dialects_database.db'
    df = fetch_data(database_file)
    df = preprocess_dataframe(df)
    
    accuracy = train_model(df)
    print(f"Model trained with accuracy: {accuracy}")
    
    model, tfidf, le = load_model()
    sample_text = 'كغو عليك'
    prediction = predict(sample_text, model, tfidf, le)
    print(f"Prediction for '{sample_text}': {prediction}")

if __name__ == "__main__":
    main()
