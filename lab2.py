import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

# Read dataset from CSV file
df = pd.read_csv("data.csv")

# Remove RID column (not needed)
df = df.drop("RID", axis=1)

# Encode categorical columns
le = LabelEncoder()
for c in df.columns:
    df[c] = le.fit_transform(df[c])

# Features and Target
X = df.drop("CLASS:BUYS_COMPUTER", axis=1)
y = df["CLASS:BUYS_COMPUTER"]

# Train Naive Bayes model
model = CategoricalNB()
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Display predictions
print("Predictions:")
print(y_pred)

# Display accuracy
print("Accuracy:", accuracy_score(y, y_pred))

# Display class probabilities
print("\nClass Probabilities:")
print(model.predict_proba(X))