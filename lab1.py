import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Read dataset
df = pd.read_csv("data.csv")

# Remove Roll column
df = df.drop("Roll", axis=1)

# Features and Target
X = df.drop("Buys_Computer", axis=1)
y = df["Buys_Computer"]

# Encode each feature column separately
for col in X.columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Encode target separately
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Train Decision Tree
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X, y)

plt.figure(figsize=(18, 10), dpi=200)

plot_tree(
    model,
    feature_names=X.columns,
    class_names=target_encoder.classes_,   # or ["No", "Yes"] if you're not using target_encoder
    filled=True,
    rounded=True,
    fontsize=12,
    proportion=False,
    precision=2
)

plt.tight_layout()
plt.show()