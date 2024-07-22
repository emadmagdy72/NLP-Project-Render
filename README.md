# Dialect Identification Project

This project demonstrates the process of dialect identification using natural language processing techniques. It consists of several Python scripts for data handling, preprocessing, model training, and prediction.

## Demo 
 [https://renderapp-nlp.onrender.com/]( #Click here to watch the demo) <!-- Replace with the actual link to the demo -->


## Scripts Overview

### 1. Data Fetching Script (`data_fetching.py`)

- Connects to a SQLite database (`dialects_database.db`).
- Retrieves data from multiple tables.
- Merges data into a single DataFrame for further processing.

### 2. Data Preprocessing Script (`data_preprocessing.py`)

- Contains functions to clean and preprocess text data.
- Operations include removing emojis, converting text to lowercase, removing numbers and extra whitespace, and applying Arabic stopwords removal.

### 3. Model Training Script (`model_training.py`)

- Involves the training of a machine learning model (Logistic Regression).
- Includes functions for text vectorization using TF-IDF and label encoding.
- Trains the model and saves the trained models (`model.pkl`, `tfidf.pkl`, `label_encoder.pkl`).

### 4. Main Execution Script (`main.py`)

- Integrates the above functionalities in a sequential manner:
  1. Fetches data using `fetch_data` from `data_fetching.py`.
  2. Preprocesses the data using `preprocess_dataframe` from `data_preprocessing.py`.
  3. Trains a model using `train_model` from `model_training.py`.
  4. Loads the trained model and necessary transformers using `load_model` from `model_training.py`.
  5. Makes predictions using `predict` from `model_training.py` based on sample input.

## Usage

To run the project:

1. Ensure `dialects_database.db` is accessible and contains relevant data.
2. Install necessary dependencies: `pandas`, `nltk`, `scikit-learn`, `tensorflow`, etc.
3. Execute `main.py`:

   ```bash
   python main.py
