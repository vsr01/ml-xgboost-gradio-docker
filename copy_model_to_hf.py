# copy_model_to_hf.py

import shutil
import os

SOURCE = "output/iris_model.pkl"
DEST = "../iris-model/iris_model.pkl"  # Adjust if your HF model repo lives elsewhere

if not os.path.exists(SOURCE):
    raise FileNotFoundError(f"Trained model not found at: {SOURCE}")

os.makedirs(os.path.dirname(DEST), exist_ok=True)
shutil.copy2(SOURCE, DEST)

print(f"✅ Model copied from '{SOURCE}' to '{DEST}'")
