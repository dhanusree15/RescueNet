# 🚨 RescueNet - Disaster Message Classifier

**RescueNet** is a machine learning-powered disaster response system that classifies emergency-related messages into multiple relevant aid categories. It enables emergency responders to efficiently triage incoming text data and route it to the appropriate response teams in real time.

---

## 📘 Project Overview

This project is part of a disaster response pipeline that processes multilingual social media and emergency messages. It uses natural language processing (NLP) to identify the type of assistance needed—such as medical help, water, food, shelter, etc.—and classifies each message into one or more of **36 disaster-related categories**.

---

## 🧠 Core Components

### 🔹 1. Data Preprocessing & ETL
- Extracts and merges `disaster_messages.csv` and `disaster_categories.csv`.
- Cleans and transforms raw message data.
- Stores the cleaned dataset into a SQLite database (`messages.db`).

> Implemented in: `etl_pipeline.ipynb`

---

### 🔹 2. Machine Learning Pipeline
- Uses **TF-IDF Vectorization** with custom text tokenization (stopword removal + stemming).
- Implements a **MultiOutputClassifier** using **Logistic Regression** to handle multilabel classification.
- Model is trained, evaluated, and saved as a `.pkl` file.

> Implemented in: `train_classifier.py` and `ml_pipeline.ipynb`

---

### 🔹 3. Web Application & API
- Built using **Flask** and **Plotly**, the app allows users to:
  - Enter a disaster-related message.
  - View its predicted categories.
  - Explore visualizations of message distribution and top categories.

> Implemented in: `run.py`  
> Hosted on: `localhost:3001`

---

## 📊 Key Features

- 🔎 **Multilabel Classification**: A single message can be tagged with multiple labels like *water*, *shelter*, *medical help*, etc.
- 📈 **Interactive Visual Dashboard**:
  - Distribution of messages by genre (direct, news, social).
  - Most common aid-related categories.
  - Word frequency histogram.
- ✂️ **Custom NLP Tokenization**: Cleans input using regex, removes stopwords, and stems words before vectorization.

---

## 📂 Dataset Summary

- Source: [Figure Eight Disaster Response Data](https://appen.com/)
- Files:
  - `disaster_messages.csv`
  - `disaster_categories.csv`
- Stored as: `messages.db` (SQLite)

---

## 🧪 Model Evaluation

Each category is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**  
Evaluation metrics are printed for each category in the terminal during model training.

---

## 🗂️ Project Structure

RescueNet/
├── app/
│ ├── templates/
│ │ ├── index.html
│ │ └── result.html
│ └── run.py # Flask web app
├── data/
│ ├── disaster_messages.csv
│ ├── disaster_categories.csv
│ └── messages.db # SQLite database
├── models/
│ ├── train_classifier.py # ML pipeline training script
│ └── classifier.pkl # Saved ML model
├── notebooks/
│ ├── etl_pipeline.ipynb
│ ├── ml_pipeline.ipynb
│ └── dashboard_visuals.ipynb



---

## 📬 Acknowledgments

- Developed as a solution for emergency message classification under real-world disaster response scenarios.
- Special thanks to **Figure Eight** for providing the original labeled dataset.

---

