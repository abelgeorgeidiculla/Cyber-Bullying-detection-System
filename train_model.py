import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

# 🚀 Load dataset
print(" Loading dataset...")
df = pd.read_csv('cyberbullying_dataset_5000.csv')

# ✅ Extract features and labels
X = df['text']

# FIX: Explicitly list your target categories so metadata isn't included
label_columns = ['insult', 'threat', 'identity_hate', 'gender', 'religion', 'ethnicity']
y = df[label_columns]

# 🎯 Train-test split
print("🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 🧠 Build pipeline
print("🔧 Building pipeline...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', OneVsRestClassifier(
        CalibratedClassifierCV(estimator=LogisticRegression(solver='liblinear'), cv=3)
    ))
])

# 🔍 Define hyperparameters for Grid Search
params = {
    'clf__estimator__estimator__C': [0.1, 1, 10],
    'clf__estimator__estimator__penalty': ['l1', 'l2']
}

# 🔥 Train with GridSearchCV
print("🚀 Starting training...")
grid_search = GridSearchCV(pipeline, param_grid=params, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# ✅ Best model
final_pipeline = grid_search.best_estimator_

# 🧠 Evaluate
print("📊 Evaluating model...")
y_pred = final_pipeline.predict(X_test)

print("\n🎯 Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n🔍 F1 Score (micro):", f1_score(y_test, y_pred, average='micro'))
print("\n📝 Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# 💾 Save model
print("💾 Saving model to 'final_model.pkl'...")
joblib.dump(final_pipeline, 'final_model.pkl')

print("✅ Model training and saving complete!")


