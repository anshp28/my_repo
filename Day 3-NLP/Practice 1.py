from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

def analyze_sentiment_textblob(text):
    sentiment = TextBlob(text).sentiment.polarity
    print(sentiment)
    if sentiment > 0:
        return "Positive 😄"
    elif sentiment < 0:
        return "Negative 😠"
    else:
        return "Neutral 😐"

def analyze_sentiment_vader(text):
    analyzer=SentimentIntensityAnalyzer()
    sentiment=analyzer.polarity_scores(text)["compound"]

    if sentiment > 0.05:
        return "Positive 😄"
    elif sentiment < -0.05:
        return "Negative 😠"
    else:
        return "Neutral 😐"

def analyze_sentiment_flair(text):
    classifier=TextClassifier.load("sentiment")
    sentence = Sentence(text)
    classifier.predict(sentence)
    print(sentence.labels)

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")
def analyze_sentiment_spacy(text):

    doc = nlp(text)
    return doc._.polarity

def analyze_input():
    text=input("Enter a sentence: ")
    print("Choose Analysis Method :")
    print("1. Text Blob Analysis")
    print("2. Vader Sentiment Analysis")
    print("3. flair Sentiment Analysis")
    print("4. spacy Sentiment Analysis")
    choice=input("Enter your choice: ").strip()
    if choice == "1":
        print(f"Text Blob Sentiment : {analyze_sentiment_textblob(text)}")
    elif choice == "2":
        print(f"Text Blob Sentiment : {analyze_sentiment_vader(text)}")
    elif choice == "3":
        print(f"Text Blob Sentiment : {analyze_sentiment_flair(text)}")
    elif choice == "4":
        print(f"Text Blob Sentiment : {analyze_sentiment_spacy(text)}")
    else:
        print("Invalid Choice")

analyze_input()


