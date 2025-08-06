# Fake News Detection

A simple and effective web application that classifies news text as **Fake** or **Real** using Machine Learning models. The app is built with Python and deployed using Gradio on Hugging Face Spaces.

## Features

- Input a news article or headline
- Get predictions from:
  - Logistic Regression
  - Gradient Boosting
  - Random Forest
- Clean user interface using Gradio
- Deployed on Hugging Face for public access

## Demo

🔗 [Live App on Hugging Face Spaces](https://huggingface.co/spaces/lavanyabafila2/fake-news-detector)

## Models Used

- **Logistic Regression**
- **Random Forest Classifier**

These models were trained using a labeled dataset of real and fake news. Text features are extracted using a pre-trained `TfidfVectorizer`.

## Project Structure

├── app.py # Main Gradio app
├── logistic_model.pkl # Trained Logistic Regression model
├── gradientboost_model.pkl # Trained Gradient Boosting model
├── randomforest_model.pkl # Trained Random Forest model
├── vectorizer.pkl # TfidfVectorizer for transforming input text
├── requirements.txt # Python dependencies

## How to Run Locally
```bash
git clone https://github.com/Lavanyabafila/FakeNews-Detection.git
cd FakeNews-Detection

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```
License
This project is licensed under the MIT License.



