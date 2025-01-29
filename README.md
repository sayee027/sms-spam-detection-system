# SMS Spam Detection System with Phone Number Blocking 🔍📱

## Overview
An AI-powered SMS spam detection system that classifies messages as **spam** or **legitimate** using Natural Language Processing (NLP) and Machine Learning (ML). The system also allows users to automatically/manually block phone numbers associated with spam messages. Built with Python and deployed via Streamlit.

---

## Technologies Used 🛠️
- **Python**
- **Streamlit** (Web Interface)
- **Scikit-learn** (ML Model: `MultinomialNB`)
- **NLTK** (Text Preprocessing)
- **Pandas** (Data Handling)
- **Plotly** (Visualizations)

---

## Key Features ✨

### 1. Data Collection & Preprocessing
- **Dataset**: Uses a custom SMS dataset (`sms-spam-updated-phone-numbers.csv`) with `text`, `labels`, and `phone_number` columns.
- **Cleaning**:
  - Handles missing/duplicate values.
  - Label encoding for spam (`1`) vs. legitimate (`0`).
- **Text Preprocessing**:
  - Lowercasing, tokenization, and removal of special characters/punctuation.
  - Stopword removal and stemming using `PorterStemmer`.

### 2. Model Training & Evaluation
- **Algorithm**: `Multinomial Naive Bayes` classifier with `TF-IDF` vectorization.
- **Performance Metrics**:
  - Accuracy: Up to **98%**
  - Precision: **99%** (Minimizes false positives)
  - Interactive confusion matrix visualization.

### 3. Phone Number Blocking
- **Automatic Blocking**: Blocks numbers linked to spam messages in the dataset.
- **Manual Blocking**: Users can manually enter/block suspicious numbers (10-digit validation).
- **Real-Time Management**: Sidebar interface to view/remove blocked numbers.

### 4. Web Interface (Streamlit)
- **Input**: Users can paste SMS messages for real-time spam checks.
- **Output**: 
  - Spam/legitimate prediction with confidence score.
  - Automatic/manual phone number blocking.
- **Metrics Dashboard**: Displays model accuracy, precision, and confusion matrix.

---

## Demo 🚀
Try the live demo: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sms-spam-detection-system-tsotl4zm79sxaz7xxvrtks.streamlit.app/#analyze-message)
![](https://via.placeholder.com/600x400?text=Screenshot+of+Streamlit+Interface) *(Replace with actual screenshot)*

---

## Installation & Usage 💻

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sms-spam-detection.git
cd sms-spam-detection
