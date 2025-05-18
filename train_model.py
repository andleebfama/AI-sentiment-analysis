# 1. Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# 2. Load Dataset
df = pd.read_csv('Train.csv')  # Make sure path sahi ho!

# 3. Separate Features and Labels
X = df['text']
y = df['label']

# 4. Vectorize Text Data
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Max 5000 features
X_vectorized = vectorizer.fit_transform(X)

# 5. Split into Train and Validation (optional)
X_train, X_val, y_train, y_val = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# 6. Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Evaluate (optional)
score = model.score(X_val, y_val)
print(f"Validation Accuracy: {score:.2f}")

# 8. Save the Model and Vectorizer
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully!")
