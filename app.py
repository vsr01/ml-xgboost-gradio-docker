import gradio as gr
import joblib
import numpy as np

# Load model and features
model = joblib.load("output/iris_model.pkl")
with open("model/features.txt") as f:
    features = [line.strip() for line in f]

target_names = ["Setosa", "Versicolor", "Virginica"]

# Prediction function with confidence
def predict_with_confidence(sepal_length, sepal_width, petal_length, petal_width):
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    probs = model.predict_proba(X)[0]
    pred_index = np.argmax(probs)
    pred_label = target_names[pred_index]
    
    label_output = f"Predicted: {pred_label}"
    prob_output = {target_names[i]: float(p) for i, p in enumerate(probs)}
    
    return label_output, prob_output

# Inputs: sliders using dynamic feature names
inputs = [
    gr.Slider(4.0, 8.0, step=0.1, label=features[0]),
    gr.Slider(2.0, 4.5, step=0.1, label=features[1]),
    gr.Slider(1.0, 7.0, step=0.1, label=features[2]),
    gr.Slider(0.1, 2.5, step=0.1, label=features[3])
]

# Outputs: predicted label and confidence bar chart
outputs = [
    gr.Text(label="Prediction"),
    gr.Label(label="Confidence Scores")
]

# Launch Gradio app (bind to all interfaces for VS Code container access)
if __name__ == "__main__":
    gr.Interface(
        fn=predict_with_confidence,
        inputs=inputs,
        outputs=outputs,
        title="Iris Species Predictor with Confidence",
        description="Adjust the sliders to classify the iris flower and view model confidence."
    ).launch(server_name="0.0.0.0", server_port=7860)
