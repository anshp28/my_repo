from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
import spacy

# We import spacytextblob but DO NOT instantiate it manually
# Just add it by name to the pipeline
from spacytextblob.spacytextblob import SpacyTextBlob

# Initialize sentiment analyzers
analyzer = SentimentIntensityAnalyzer()
classifier = TextClassifier.load("sentiment")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Add spacytextblob component by name if not already present
if "spacytextblob" not in nlp.pipe_names:
    nlp.add_pipe("spacytextblob", last=True)

# TextBlob
def analyze_sentiment_textblob(text):
    polarity = TextBlob(text).sentiment.polarity
    print(f"TextBlob Polarity Score: {polarity}")
    if polarity > 0:
        return "Positive 😄"
    elif polarity < 0:
        return "Negative 😠"
    else:
        return "Neutral 😐"

# VADER
def analyze_sentiment_vader(text):
    score = analyzer.polarity_scores(text)["compound"]
    print(f"VADER Compound Score: {score}")
    if score > 0.05:
        return "Positive 😄"
    elif score < -0.05:
        return "Negative 😠"
    else:
        return "Neutral 😐"

# Flair
def analyze_sentiment_flair(text):
    sentence = Sentence(text)
    classifier.predict(sentence)
    label = sentence.labels[0]
    print(f"Flair Output: {label}")
    if "POSITIVE" in label.value.upper():
        return "Positive 😄"
    elif "NEGATIVE" in label.value.upper():
        return "Negative 😠"
    else:
        return "Neutral 😐"

# spaCy (with spacytextblob)
def analyze_sentiment_spacy(text):
    doc = nlp(text)
    # The spacytextblob extension should be registered now
    if not hasattr(doc._, "polarity"):
        print("spaCyTextBlob extension not set properly.")
        return "Error"
    polarity = doc._.polarity
    print(f"spaCy TextBlob Polarity Score: {polarity}")
    if polarity > 0:
        return "Positive 😄"
    elif polarity < 0:
        return "Negative 😠"
    else:
        return "Neutral 😐"

# Main CLI
def analyze_input():
    text = input("Enter a sentence: ")
    print("\nChoose Analysis Method:")
    print("1. TextBlob Analysis")
    print("2. VADER Sentiment Analysis")
    print("3. Flair Sentiment Analysis")
    print("4. spaCy Sentiment Analysis")
    choice = input("Enter your choice: ").strip()

    if choice == "1":
        print(f"TextBlob Sentiment: {analyze_sentiment_textblob(text)}")
    elif choice == "2":
        print(f"VADER Sentiment: {analyze_sentiment_vader(text)}")
    elif choice == "3":
        print(f"Flair Sentiment: {analyze_sentiment_flair(text)}")
    elif choice == "4":
        print(f"spaCy Sentiment: {analyze_sentiment_spacy(text)}")
    else:
        print("Invalid Choice")

if __name__ == "__main__":
    analyze_input()
