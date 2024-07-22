from data_fetching import fetch_data
from data_preprocessing import preprocess_dataframe
from model_training import train_model, load_model, predict

def main():
    file_path = 'dialects_database.db' 
    df = fetch_data(file_path)
    df = preprocess_dataframe(df)
    
    model = train_model(df)
    
    model, tokenizer = load_model()
    sample_text = 'your sample text here'
    prediction = predict(sample_text, model, tokenizer)
    print(f"Prediction for '{sample_text}': {prediction}")

if __name__ == "__main__":
    main()
