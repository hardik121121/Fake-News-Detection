import streamlit as st
import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    true = pd.read_csv('True.csv')
    fake = pd.read_csv('Fake.csv')

    true['label'] = 1
    fake['label'] = 0

    news = pd.concat([fake, true], axis=0).sample(frac=1).reset_index(drop=True)
    news = news.drop(['title', 'subject', 'date'], axis=1)

    def wordopt(text):
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d', '', text)
        text = re.sub(r'\n', '', text)
        return text

    news['text'] = news['text'].apply(wordopt)

    return news

# Train models and cache them
@st.cache_resource
def train_models(news):
    x = news['text']
    y = news['label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(xv_train, y_train)

    # Decision Tree Classifier
    dtc = DecisionTreeClassifier()
    dtc.fit(xv_train, y_train)

    # Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(xv_train, y_train)

    # Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(n_estimators=100)
    gbc.fit(xv_train, y_train)

    # Save models using joblib for faster reloading
    joblib.dump(lr, 'lr_model.pkl')
    joblib.dump(dtc, 'dtc_model.pkl')
    joblib.dump(rfc, 'rfc_model.pkl')
    joblib.dump(gbc, 'gbc_model.pkl')
    joblib.dump(vectorization, 'vectorization.pkl')

    # Generate and display evaluation metrics
    pred_lr = lr.predict(xv_test)
    pred_dtc = dtc.predict(xv_test)
    pred_rfc = rfc.predict(xv_test)
    pred_gbc = gbc.predict(xv_test)

    st.write("**Model Evaluation Metrics:**")
    st.write("**Logistic Regression:**")
    st.text(classification_report(y_test, pred_lr))
    st.write("**Decision Tree:**")
    st.text(classification_report(y_test, pred_dtc))
    st.write("**Random Forest:**")
    st.text(classification_report(y_test, pred_rfc))
    st.write("**Gradient Boosting:**")
    st.text(classification_report(y_test, pred_gbc))

    return vectorization, lr, dtc, rfc, gbc

# Predict manually
def manual_testing(news_text, vectorization, lr, dtc, rfc, gbc):
    def wordopt(text):
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d', '', text)
        text = re.sub(r'\n', '', text)
        return text

    def output_label(n):
        return "❌ Fake News" if n == 0 else "✅ True News"

    testing_news = pd.DataFrame({"text": [news_text]})
    testing_news["text"] = testing_news["text"].apply(wordopt)

    new_xv_test = vectorization.transform(testing_news['text'])

    pred_lr = lr.predict(new_xv_test)
    pred_dtc = dtc.predict(new_xv_test)
    pred_rfc = rfc.predict(new_xv_test)
    pred_gbc = gbc.predict(new_xv_test)

    return {
        "Logistic Regression": output_label(pred_lr[0]),
        "Decision Tree": output_label(pred_dtc[0]),
        "Random Forest": output_label(pred_rfc[0]),
        "Gradient Boosting": output_label(pred_gbc[0]),
    }

# App UI
st.title("📰 Fake News Detection")

st.write("Welcome to the **Fake News Detection App**! Enter a news article below to analyze if it is real or fake using various machine learning models.")

# Load pre-trained models
try:
    vectorization = joblib.load('vectorization.pkl')
    lr = joblib.load('lr_model.pkl')
    dtc = joblib.load('dtc_model.pkl')
    rfc = joblib.load('rfc_model.pkl')
    gbc = joblib.load('gbc_model.pkl')
except:
    # If models are not found, train them
    news_data = load_and_preprocess_data()
    vectorization, lr, dtc, rfc, gbc = train_models(news_data)

# News Input
news_article = st.text_area("📝 Enter News Article:", height=200)

if st.button("Analyze"):
    if news_article.strip():
        predictions = manual_testing(news_article, vectorization, lr, dtc, rfc, gbc)
        st.subheader("Analysis Results:")

        for model, prediction in predictions.items():
            st.write(f"**{model}:** {prediction}")
    else:
        st.error("Please enter some news text to analyze.")

st.sidebar.header("About the App 📖")
st.sidebar.write(
    """
    - **Goal**: Detect fake news with high accuracy.  
    - **Models Used**:  
        - Logistic Regression  
        - Gradient Boosting Classifier  
        - Random Forest Classifier  
        - Decision Tree Classifier  
    """
)
st.sidebar.info("✨ Developed with ❤️ and ML magic!")
