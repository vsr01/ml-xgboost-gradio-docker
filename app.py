# app.py
import joblib
import gradio as gr
import numpy as np
from sklearn.datasets import load_iris

# Load model
model = joblib.load("output/iris_model.pkl")
iris = load_iris()

def predict(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    probs = model.predict_proba(features)[0]
    confidences = {iris.target_names[i]: f"{100 * p:.2f}%" for i, p in enumerate(probs)}
    return confidences

inputs = [
    gr.Slider(4.0, 8.0, step=0.1, label="Sepal Length (cm)"),
    gr.Slider(2.0, 4.5, step=0.1, label="Sepal Width (cm)"),
    gr.Slider(1.0, 7.0, step=0.1, label="Petal Length (cm)"),
    gr.Slider(0.1, 2.5, step=0.1, label="Petal Width (cm)"),
]

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Label(num_top_classes=3),
    title="Iris Species Classifier",
    description="Adjust the sliders to predict the iris flower species."
)

if __name__ == "__main__":
    demo.launch()
