# 🏨 Hotel Price Category Classification

A machine learning project that predicts the **price category** of hotels using structured hotel information collected through **web scraping**. The project implements a complete machine learning pipeline, including data extraction, preprocessing, feature engineering, model training, evaluation, and offline prediction.

---

## 📌 Project Overview

The goal of this project is to build a **multi-class classification model** that predicts one of the following hotel price categories:

- 💰 Budget
- 🏨 Standard
- ⭐ Premium
- 👑 Luxury

The categories are generated from the hotel price using quartile-based categorization to create a balanced classification problem.

---

## 📂 Project Structure

```text
HOTEL_PRICE_CATEGORY/
│
├── data/
│   ├── city/                    # Raw scraped data by city
│   └── merged/                  # Final merged dataset
│
├── logs/                        # Log files
│
├── models/
│   ├── baseline/                # Baseline trained models
│   └── engineered/              # Models after feature engineering
│
├── notebooks/
│   ├── data_extraction.ipynb
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   ├── baseline_modeling.ipynb
│   └── offline_testing.ipynb
│
├── scripts/
│   ├── data_extraction.py
│   ├── feature_engineering.py
│   └── baseline_modeling.py
│
├── src/
│   ├── data_extraction.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── logger.py
│   ├── pipeline.py
│   └── preprocess.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

The dataset consists of **1,258 hotel records** collected through web scraping.

| Feature | Description |
|----------|-------------|
| title | Hotel name |
| price | Hotel room price (USD) |
| adults | Maximum number of adults |
| city | Hotel location |
| distance_to_center | Distance from city center (km) |
| description | Hotel description |
| bf_has_included | Breakfast included (True/False) |
| price_category | Target variable |

---

# 🎯 Target Classes

The target variable consists of four classes:

| Category | Description |
|----------|-------------|
| Budget | Low-priced hotels |
| Standard | Mid-range hotels |
| Premium | High-quality hotels |
| Luxury | Luxury hotels |

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- BeautifulSoup4
- Requests
- Matplotlib
- Joblib

---

# 🤖 Machine Learning Models

The following models were evaluated:

- Logistic Regression
- Decision Tree
- One-vs-One Classifier
- One-vs-Rest Classifier

---

Logs are stored in the `logs/` directory.

---

# 💾 Saved Models

Trained models are stored inside:

```
models/
├── baseline/
└── engineered/
```

Models are saved using **Joblib**.

---

# 🚀 Running the Project

## 1. Clone the repository

```bash
git clone https://github.com/agus787-dev/hotel_price_category.git
cd hotel-price-category
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Extract hotel data

```bash
python scripts/data_extraction.py
```

---

## 4. Train baseline models

```bash
python scripts/baseline_modeling.py
```

---

## 5. Perform feature engineering

```bash
python scripts/feature_engineering.py
```

---

## 6. Test with new hotel data

Open

```
notebooks/offline_testing.ipynb
```

and predict the category of an unseen hotel.

---

# 👨‍💻 Author

**Machine Learning Project**

Hotel Price Category Classification using Supervised Machine Learning.