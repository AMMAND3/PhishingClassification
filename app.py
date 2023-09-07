from flask import Flask, render_template, request
import os
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from text_transformers import TextLengthExtractor
from counter import NumericCounter

# Initialization
app = Flask(__name__)

# Download necessary resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Text preprocessing utilities
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Load the combined model
model = joblib.load('phish_classifier_model_combined.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        message = request.form.get('message')
        processed_message = preprocess_text(message)
        prediction = model.predict([processed_message])
        prediction = 'Possible Phishing' if prediction[0] == 'spam' else 'Safe Mail'
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
