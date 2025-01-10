# 📰 Fake News Detection App

Welcome to the **Fake News Detection App**! This application leverages machine learning models to help you analyze news articles and classify them as **True** or **Fake**. With the power of **Natural Language Processing (NLP)** and **ML Algorithms**, this app can assist in combating misinformation and promoting trustworthy news.

## 📌 Features

- **Multi-Model Analysis**: Use multiple machine learning models for news classification:
  - Logistic Regression 🧠
  - Decision Tree 🌳
  - Random Forest 🌲
  - Gradient Boosting ⚡
  
- **Real-Time News Classification**: Enter a news article, and the app will analyze it in real-time to classify it as **True** ✅ or **Fake** ❌.

- **Evaluation Metrics**: View detailed performance metrics for each model, including accuracy, precision, recall, and F1 score.

- **Easy-to-Use Interface**: Simply input the news article and click the **Analyze** button to see results.

---

## 🛠️ Technologies Used

- **Streamlit**: For building the interactive UI.
- **scikit-learn**: For training and evaluating the machine learning models.
- **Pandas**: For data manipulation.
- **TfidfVectorizer**: For text vectorization (converting text into numerical form).
- **Joblib**: For saving and loading trained models.
- **Regular Expressions (Regex)**: For preprocessing and cleaning the news data.

---

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### 2. Install Dependencies

Ensure you have **Python 3.x** installed, then create a virtual environment and install the necessary packages:

```bash
# Create a virtual environment
python -m venv venv
# Activate the virtual environment
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows

# Install required packages
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

Your Fake News Detection app will now be running locally at `http://localhost:8501`.

---

## 🔍 How It Works

### 1. **Data Preprocessing**
   - We load two datasets, one containing **True** news and the other **Fake** news.
   - The text is cleaned using various techniques like:
     - Lowercasing the text.
     - Removing URLs and HTML tags.
     - Removing punctuation and numbers.
     - Stripping newlines and extra spaces.

### 2. **Model Training**
   - We train four different machine learning models using the cleaned text data:
     - **Logistic Regression**: A simple yet powerful model for binary classification.
     - **Decision Tree**: A tree-based model for decision making.
     - **Random Forest**: An ensemble method using multiple decision trees.
     - **Gradient Boosting**: A boosting algorithm to improve predictive accuracy.
   
   - After training, we save the models and the vectorizer (for transforming new text input) using **joblib**.

### 3. **Prediction**
   - Users input news articles into the app. The text is preprocessed, transformed using the saved vectorizer, and passed through each of the models for classification.
   - The app outputs whether the news article is **True** or **Fake** based on each model's prediction.

---

## 📊 Model Evaluation

After training the models, we evaluate their performance using metrics such as:

- **Accuracy**: The proportion of correctly predicted news articles.
- **Precision**: The percentage of predicted fake articles that are actually fake.
- **Recall**: The percentage of actual fake articles that are correctly identified.
- **F1 Score**: A weighted average of precision and recall.

### **Model Performance**
- **Logistic Regression**: 🏆 High Accuracy
- **Decision Tree**: 🌳 Balanced Results
- **Random Forest**: 🌲 High Precision
- **Gradient Boosting**: ⚡ Strong Recall

---

## 🧑‍💻 Usage

1. **Input News Article**: Type or paste any news article in the text box provided.
2. **Click 'Analyze'**: Press the button to classify the article.
3. **View Results**: See predictions from each model. Whether the article is **True** ✅ or **Fake** ❌ will be displayed.

---

## ⚙️ Running the App Locally

After setting up the environment and dependencies, you can test the application locally. Just run:

```bash
streamlit run fake_news_detection.py
```

This will launch the app in your browser.

---

## 📚 About the Dataset

The dataset consists of two parts:
- **True News**: Real articles (labeled as `1`).
- **Fake News**: Misinformation articles (labeled as `0`).

You can use this app to classify news articles based on their content.

---

## 🏆 Contributing

We welcome contributions to improve this project! If you have suggestions, bug fixes, or new features to add, feel free to fork this repository and create a pull request. 

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 💬 Feedback

If you have any feedback or questions, feel free to reach out! You can contact the developer at (mailto:hardikarora483@gmail.com).

---

## ✨ Developed with ❤️ and ML Magic
```
