import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import string
import json
from pathlib import Path
import logging
from typing import Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import plotly.express as px
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Configure the page
st.set_page_config(page_title="Spam Detection System", layout="wide")

# Initialize NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk_data()

# Configure logging
def setup_logging() -> logging.Logger:
    log_file = Path("spam_detector.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Text preprocessing with caching
@st.cache_data
def preprocess_texts_with_cache(texts: pd.Series, stop_words: Set[str]) -> pd.Series:
    ps = PorterStemmer()
    
    def preprocess_text(text: str) -> str:
        try:
            text = text.lower()
            words = nltk.word_tokenize(text)
            words = [word for word in words if word.isalpha()]
            words = [ps.stem(word) for word in words if word not in stop_words]
            return " ".join(words)
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text

    return texts.apply(preprocess_text)

@st.cache_resource
class PhoneBlocker:
    def __init__(self):
        self.blocked_numbers: Set[str] = set()

    def validate_phone_number(self, number: str) -> bool:
        if not number:
            return False
        cleaned_number = re.sub(r'[^\d]', '', number)
        return bool(re.match(r'^\d{10}$', cleaned_number))

    def block_number(self, number: str) -> bool:
        if not self.validate_phone_number(number):
            return False
        cleaned_number = re.sub(r'[^\d]', '', number)
        self.blocked_numbers.add(cleaned_number)
        return True

    def unblock_number(self, number: str) -> bool:
        if number in self.blocked_numbers:
            self.blocked_numbers.remove(number)
            return True
        return False

@st.cache_resource
class SpamDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()
        self.is_trained = False
        self.model_metrics = {}
        self.df = None  # Store the training data for lookup

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        try:
            if 'phone_number' not in df.columns:
                df['phone_number'] = None
            
            required_columns = {'text', 'labels'}
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Found: {df.columns}")
            
            df = df.drop_duplicates().dropna(subset=['text', 'labels'])
            stop_words = set(stopwords.words('english'))
            df['processed_text'] = preprocess_texts_with_cache(df['text'], stop_words)
            
            self.df = df  # Store the dataframe for later lookup
            return df, True
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            st.error(f"Data preparation error: {e}")
            return None, False

    def train(self, df: pd.DataFrame) -> bool:
        try:
            X = self.vectorizer.fit_transform(df['processed_text'])
            y = (df['labels'].str.lower() == 'spam').astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)
            
            self.model_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def predict(self, text: str) -> Tuple[int, float]:
        if not text.strip():
            raise ValueError("Empty message provided")
            
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            stop_words = set(stopwords.words('english'))
            processed_text = preprocess_texts_with_cache(
                pd.Series([text]), 
                stop_words
            ).iloc[0]
            
            X = self.vectorizer.transform([processed_text])
            prediction = self.classifier.predict(X)[0]
            probability = self.classifier.predict_proba(X)[0][prediction]
            
            return prediction, probability
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise

    def find_phone_number(self, text: str) -> str:
        """Find the phone number associated with the given text in the training data."""
        if self.df is not None:
            match = self.df[self.df['text'] == text]
            if not match.empty:
                return match.iloc[0]['phone_number']
        return None

def load_data() -> pd.DataFrame:
    """Load the training data from CSV file"""
    try:
        file_path = Path("sms-spam-updated-phone-numbers.csv")
        df = pd.read_csv(file_path)
        st.success("Training data loaded successfully!")
        return df
    except FileNotFoundError:
        st.error("Training data file not found")
        st.info("Please ensure 'sms-spam-updated-phone-numbers.csv' is in the same directory as the script.")
        return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def display_confusion_matrix(cm):
    """Create and configure confusion matrix plot"""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual"),
        x=['Not Spam', 'Spam'],
        y=['Not Spam', 'Spam'],
        color_continuous_scale='Blues',
        text_auto=True  # Automatically add text annotations
    )
    fig.update_traces(texttemplate="%{z}", textfont={"size": 16})  # Customize text
    return fig

def main():
    st.title("📱 SMS Spam Detection System")
    st.markdown("This system helps identify spam messages and manage blocked phone numbers.")

    phone_blocker = PhoneBlocker()
    spam_detector = SpamDetector()

    # Sidebar for blocked numbers management
    with st.sidebar:
        st.header("📞 Blocked Numbers Management")
        new_number = st.text_input("Enter phone number to block (10 digits)")
        if st.button("Block Number"):
            if phone_blocker.block_number(new_number):
                st.success(f"Blocked number: {new_number}")
            else:
                st.error("Invalid phone number format")
        
        st.subheader("Currently Blocked Numbers")
        for number in phone_blocker.blocked_numbers:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.code(number)
            with col2:
                if st.button("Unblock", key=f"unblock_{number}"):
                    if phone_blocker.unblock_number(number):
                        st.success("Number unblocked")
                        st.rerun()

    # Main content tabs
    tab1, tab2 = st.tabs(["💬 Message Analysis", "📊 Model Metrics"])

    with tab1:
        df = load_data()
        
        if df is not None:
            with st.expander("Preview Training Data"):
                st.dataframe(df.head())
            
            if not spam_detector.is_trained:
                with st.spinner("Training model..."):
                    prepared_df, success = spam_detector.prepare_data(df)
                    if success and prepared_df is not None:
                        if spam_detector.train(prepared_df):
                            st.success("Model trained successfully! 🎉")
                        else:
                            st.error("Failed to train model.")
                    else:
                        st.error("Failed to prepare data.")

        st.subheader("Analyze Message")
        message_text = st.text_area("Enter message to analyze")

        if st.button("Analyze Message") and message_text:
            if not spam_detector.is_trained:
                st.warning("⚠️ Please wait for the model to finish training!")
            else:
                try:
                    prediction, confidence = spam_detector.predict(message_text)
                    
                    result_container = st.container()
                    with result_container:
                        if prediction == 1:
                            st.error("🚨 SPAM Detected!")
                            # Find the phone number associated with the text
                            phone_number = spam_detector.find_phone_number(message_text)
                            if phone_number:
                                if phone_blocker.block_number(phone_number):
                                    st.warning(f"📵 Automatically blocked phone number: {phone_number}")
                            else:
                                st.warning("No associated phone number found in the dataset.")
                                # Allow user to manually enter the phone number
                                manual_phone_number = st.text_input(
                                    "Enter associated phone number (optional)",
                                    key="manual_phone_number"
                                )
                                if manual_phone_number:
                                    if phone_blocker.block_number(manual_phone_number):
                                        st.success(f"📵 Manually blocked phone number: {manual_phone_number}")
                                    else:
                                        st.error("Invalid phone number format")
                        else:
                            st.success("✅ Legitimate Message")
                        
                        st.markdown("### Confidence Score")
                        st.progress(confidence)
                        st.text(f"{confidence:.2%}")
                except Exception as e:
                    st.error(f"Error analyzing message: {str(e)}")

    with tab2:
        if spam_detector.is_trained:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{spam_detector.model_metrics['accuracy']:.2%}")
            with col2:
                st.metric("Precision", f"{spam_detector.model_metrics['precision']:.2%}")
            
            cm = spam_detector.model_metrics['confusion_matrix']
            fig = display_confusion_matrix(cm)
            st.plotly_chart(fig)
        else:
            st.info("Model metrics will appear here after training")

if __name__ == "__main__":
    main()
