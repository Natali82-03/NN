import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time
import requests

# Настройки страницы
st.set_page_config(page_title="Одобрение кредита", layout="centered")

# Стиль для отображения результата
RESULT_STYLE = """
<style>
    .result-box {
        font-size: 24px;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    .approved {
        background-color: #d4edda;
        color: #155724;
    }
    .denied {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
"""

st.markdown(RESULT_STYLE, unsafe_allow_html=True)

@st.cache_resource
def load_data_and_model():
    # URL CSV-файла на GitHub (замените на вашу ссылку)
    url = 'https://raw.githubusercontent.com/yourusername/yourrepo/main/L_Score.csv' 
    
    # Загрузка данных
    df = pd.read_csv(url, header=None)

    # Примерные названия столбцов (предположим, что порядок таков):
    columns = [
        "age", "gender", "education", "income", "dependents", "housing",
        "monthly_rent", "loan_purpose", "interest_rate", "dti_ratio",
        "loan_term", "credit_score", "loan_approved", "zip_code"
    ]

    df.columns = columns

    # Удаление лишних колонок
    df = df.drop(columns=["zip_code"])

    # Преобразование целевой переменной
    df["loan_approved"] = df["loan_approved"].map({"Yes": 1, "No": 0})

    # Выделение признаков и целевой переменной
    X = df.drop("loan_approved", axis=1)
    y = df["loan_approved"]

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обработка категориальных переменных
    categorical_features = ["gender", "education", "housing", "loan_purpose"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Создание пайплайна с XGBClassifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])

    # Обучение модели
    model.fit(X_train, y_train)

    return model

# Загрузка модели
with st.spinner('Загрузка модели...'):
    model = load_data_and_model()

# Интерфейс приложения
st.title("✅ Кредитный скоринг")
st.subheader("Введите данные клиента для оценки одобрения кредита")

# Ввод данных
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Возраст", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Пол", options=["male", "female"])
    education = st.selectbox("Образование", options=["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    income = st.number_input("Доход ($)", min_value=0, value=50000)
    dependents = st.number_input("Количество иждивенцев", min_value=0, value=0)

with col2:
    housing = st.selectbox("Тип жилья", options=["OWN", "MORTGAGE", "RENT"])
    monthly_rent = st.number_input("Ежемесячный платеж ($)", min_value=0, value=1500)
    loan_purpose = st.selectbox("Цель кредита", options=[
        "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", 
        "MEDICAL", "PERSONAL", "VENTURE"
    ])
    credit_score = st.number_input("Кредитный рейтинг", min_value=300, max_value=850, value=650)

# Предсказание
if st.button("🔍 Проверить одобрение"):
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "education": education,
        "income": income,
        "dependents": dependents,
        "housing": housing,
        "monthly_rent": monthly_rent,
        "loan_purpose": loan_purpose,
        "credit_score": credit_score,
        # Другие поля заполним условно или средними значениями
        "interest_rate": 10.0,
        "dti_ratio": 0.2,
        "loan_term": 3.0
    }])

    with st.spinner('Рассчитываем результат...'):
        time.sleep(1)  # Имитация задержки
        prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.markdown('<div class="result-box approved">✅ Кредит одобрен</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box denied">❌ Кредит не одобрен</div>', unsafe_allow_html=True)
else:
    st.info("Нажмите кнопку ниже для проверки статуса кредита.")

# Боковая панель
with st.sidebar:
    st.header("ℹ️ Информация")
    st.write("Это приложение использует модель машинного обучения XGBoost для предсказания одобрения кредита.")
    st.markdown("---")
    st.markdown("📊 **Источник данных:** L_Score.csv")
    st.markdown("🧠 **Модель:** XGBClassifier")