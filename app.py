import gradio as gr
import joblib

# Load the saved models and vectorizer
LR = joblib.load("logistic_model.pkl")
GB = joblib.load("gradientboost_model.pkl")
RF = joblib.load("randomforest_model.pkl")
vectorization = joblib.load("vectorizer.pkl")

# Function to label output
def output_label(n):
    return "Fake News" if n == 0 else "Real News"

# Prediction function
def predict_fake_news(news):
    try:
        transformed = vectorization.transform([news])
        result = {
            "Logistic Regression": output_label(LR.predict(transformed)[0]),
            "Gradient Boost": output_label(GB.predict(transformed)[0]),
            "Random Forest": output_label(RF.predict(transformed)[0])
        }
        return result
    except Exception as e:
        return {"error": str(e)}

# Gradio interface
iface = gr.Interface(
    fn=predict_fake_news,
    inputs=gr.Textbox(lines=6, placeholder="Enter news text here..."),
    outputs="json",
    title="Fake News Detection",
    description="Enter a news article and get predictions from multiple models."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
