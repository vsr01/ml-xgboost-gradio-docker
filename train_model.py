# train_model.py
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import os

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
os.makedirs("output", exist_ok=True)
joblib.dump(model, "output/iris_model.pkl")

# Save feature names
os.makedirs("model", exist_ok=True)
with open("model/features.txt", "w") as f:
    for name in iris.feature_names:
        f.write(f"{name}\n")