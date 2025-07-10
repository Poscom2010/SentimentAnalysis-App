import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Make sure you have the necessary NLTK data downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    stop_words = set(stopwords.words("english"))  # Stopwords
    cleaned_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(cleaned_tokens)  # Rejoin into string
