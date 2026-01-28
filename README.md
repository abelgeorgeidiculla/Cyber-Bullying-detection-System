# Multi-Label Cyberbullying Detection System

## 📌 Project Overview

This project is a Machine Learning application designed to detect and categorize various forms of cyberbullying in text. Unlike simple binary classifiers, this system uses a **multi-label classification** approach to identify specific types of toxicity, including insults, threats, identity hate, and harassment based on gender, religion, or ethnicity.

The solution consists of:

- A **synthetic dataset generator** to create balanced training data.

- A **Scikit-Learn training pipeline** using Logistic Regression and TF-IDF.

- A **Flask web application** for real-time user interaction and prediction.

## 🚀 Features

- **Multi-Label Detection**: Classifies text into one or more categories: `insult`, `threat`, `identity_hate`, `gender`, `religion`, `ethnicity`.

- **Robust NLP**: Handles realistic text noise like typos and emojis.

- **Web Interface**: User-friendly Flask app for easy testing.

- **Custom Dataset**: Includes a script to generate synthetic labeled data using `Faker`.

## 🛠️ Tech Stack

- **Language**: Python 3.x

- **Web Framework**: Flask

- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib

- **Data Generation**: Faker, NLTK

## 📂 Project Structure

```

├── app.py # Main Flask application ├── train\_model.py # ML pipeline: loads data, trains model, saves .pkl ├── generate\_dataset.py # Generates synthetic cyberbullying dataset ├── predict.py # CLI script for testing predictions ├── requirements.txt # Python dependencies └── final\_model.pkl # Trained model (generated after training)

```

## ⚙️ Installation

```bash
git clone <repository-url>
cd cyber-bullying-detection-system

```

1. **Install Dependencies** It is recommended to use a virtual environment.

   Bash

   ```
   pip install -r requirements.txt

   ```

   *Note: The data generation script also requires *******`faker`******* and *******`nltk`*******.*

   Bash

   ```
   pip install faker nltk

   ```

## 🏃‍♂️ Usage Guide

### Step 1: Generate the Dataset

If you don't have the dataset yet, generate a synthetic one (5000 samples).

Bash

```
python generate_dataset.py

```

*Output: Saves *******`cyberbullying_dataset_5000.csv`*******.*

### Step 2: Train the Model

Train the multi-label logistic regression model.

> **Note:** Ensure the CSV filename in `train_model.py` matches your generated file (e.g., rename `cyberbullying_dataset_5000.csv` to `cyberbullying_dataset.csv`).

Bash

```
python train_model.py

```

*Output: Saves the trained model as *******`final_model.pkl`*******.*

### Step 3: Run the Web Application

Start the Flask server to interact with the model.

Bash

```
python app.py

```

Open your browser and navigate to: `http://127.0.0.1:5000/`

### Step 4: Test via Command Line

You can also test a list of predefined sentences using the prediction script.

Bash

```
python predict.py

```

## 🧠 Model Details

The model uses a **Pipeline** containing:

1. **TF-IDF Vectorizer**: Extracts features from text (1-2 n-grams, max 10,000 features).

2. **OneVsRestClassifier**: Wraps a **Logistic Regression** model to handle multi-label classification (predicting multiple categories for a single text).

3. **Calibration**: Uses `CalibratedClassifierCV` for better probability estimates.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements.
