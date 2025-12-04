import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from datetime import datetime

# ===============================
# 0) Page config + Background
# ===============================
st.set_page_config(page_title="🌦️ Weather Dashboard", layout="wide")

# CSS شامل للتنسيق والظهور الواضح للنصوص داخل وخارج البوكسات
st.markdown(
    """
    <style>
    /* خلفية الصفحة */
    .stApp {
        background-color: #f4f4f9 !important;
        color: #000000 !important;
    }

    /* كل النصوص تبقى أسود */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #000000 !important;
    }

    /* نصوص الـ labels (عناوين الحقول) */
    label {
        color: #000000 !important;
        font-weight: 600;
        font-size: 16px;
        margin-bottom: 6px;
        display: block;
    }

    /* الكروت */
    .card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.15);
        color: #000000 !important;
    }

    /* تحسين selectbox */
    div[data-baseweb="select"] > div {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 12px 15px;
        color: #000000 !important;
        min-height: 60px;
        line-height: 30px;
        font-size: 16px;
        border: 1px solid #ccc;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* لون النص داخل selectbox */
    div[data-baseweb="select"] > div > div {
        color: #000000 !important;
        line-height: normal;
    }

    /* تحسين input الوقت والتاريخ */
    input[type="time"], input[type="date"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 12px 15px;
        color: #000000 !important;
        font-size: 16px;
        border: 1px solid #ccc;
        min-height: 60px;
        line-height: normal;
        white-space: nowrap;
    }

    /* زر Predict */
    div.stButton > button {
        background-color: #4a90e2;
        color: #ffffff !important;
        font-size: 18px;
        padding: 12px 25px;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #6ab0ff;
        color: #ffffff !important;
    }

    /* لون قيمة الميترك */
    .stMetricValue {
        color: #000000 !important;
    }
    .stMetricLabel {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# 1) Load Data + Model + Scalers
# ===============================
df = pd.read_csv("weather_data.csv")
model = load_model("my_model.h5", compile=False)

with open('all_scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

X_scaler = scalers["X_scaler"]
Temperature_scaler = scalers["Temperature_scaler"]
Humidity_scaler = scalers["Humidity_scaler"]
Prec_scaler = scalers["Prec_scaler"]
Wind_scaler = scalers["Wind_scaler"]

# ===============================
# 2) Recommendation Function
# ===============================
def energy_recommendations(temp, humidity, rain, wind):
    recs = []

    # Temperature
    if temp >= 30:
        recs.append(("🔥 حرارة مرتفعة", "استخدم التكييف بكفاءة أو استعمل مروحة لتوفير الكهرباء"))
    elif temp <= 10:
        recs.append(("❄️ حرارة منخفضة", "استعمل تدفئة معتدلة لتوفير الطاقة"))
    else:
        recs.append(("🌤️ درجة حرارة معتدلة", "افتح النوافذ للتهوية الطبيعية لتوفير الكهرباء"))

    # Humidity
    if humidity >= 70:
        recs.append(("💧 رطوبة عالية", "استخدم مزيلات الرطوبة بكفاءة وقلّل استهلاك الكهرباء"))
    elif humidity <= 30:
        recs.append(("🌵 رطوبة منخفضة", "استخدم أجهزة الترطيب بكمية معتدلة لتوفير الطاقة"))
    else:
        recs.append(("🌿 الرطوبة معتدلة", "لا حاجة لتشغيل أجهزة إضافية"))

    # Rain
    if rain > 0:
        recs.append(("☔ هطول أمطار", "يمكن استخدام مصادر طبيعية للتهوية دون الحاجة للكهرباء"))
    else:
        recs.append(("🌞 الطقس جاف", "استغل التهوية الطبيعية لتقليل استهلاك الكهرباء"))

    # Wind
    if wind >= 15:
        recs.append(("💨 رياح قوية", "استغل نسيم الرياح لتقليل استخدام المراوح والتكييف"))
    else:
        recs.append(("🌬️ الرياح هادئة", "استخدم المراوح أو التكييف بكفاءة عند الحاجة"))

    return recs

# ===============================
# 3) Streamlit UI
# ===============================
st.markdown("<h1 style='text-align:center;'>🌦️ Weather Prediction Dashboard 🌦️</h1>", unsafe_allow_html=True)
st.markdown("---")

# Location selector
locations = sorted(df['Location'].unique())
user_location = st.selectbox("📍 اختر المكان:", locations)

# Date selector (only 2024)
dates_2024 = pd.date_range(start="2024-01-01", end="2024-12-31").to_pydatetime().tolist()
user_date = st.selectbox("📅 اختر التاريخ (سنة 2024):", dates_2024)

# Time selector
user_time = st.time_input("⏰ اختر الوقت:")
user_datetime = datetime.combine(user_date, user_time)

# Predict button
if st.button("🔮 Predict Weather"):

    # ===============================
    # Prepare input features
    # ===============================
    location_columns = [
        "Location_Chicago","Location_Dallas","Location_Houston",
        "Location_Los Angeles","Location_New York","Location_Philadelphia",
        "Location_Phoenix","Location_San Antonio","Location_San Diego","Location_San Jose"
    ]
    location_vector = [0]*len(location_columns)
    loc_col_name = f"Location_{user_location}"
    if loc_col_name in location_columns:
        idx = location_columns.index(loc_col_name)
        location_vector[idx] = 1

    day = user_datetime.day
    month = user_datetime.month
    year = user_datetime.year
    hour = user_datetime.hour

    month_sin = np.sin(2*np.pi*month/12)
    month_cos = np.cos(2*np.pi*month/12)
    day_sin = np.sin(2*np.pi*day/31)
    day_cos = np.cos(2*np.pi*day/31)

    input_features = [day, month, year, hour, month_sin, month_cos, day_sin, day_cos] + location_vector
    input_scaled = X_scaler.transform([input_features])
    model_input = input_scaled.reshape(1, 1, input_scaled.shape[1])

    # ===============================
    # Make prediction
    # ===============================
    pred_scaled = model.predict(model_input)
    temp = Temperature_scaler.inverse_transform(pred_scaled[:, 0].reshape(-1, 1))[0][0]
    humidity = Humidity_scaler.inverse_transform(pred_scaled[:, 1].reshape(-1, 1))[0][0]
    rain = Prec_scaler.inverse_transform(pred_scaled[:, 2].reshape(-1, 1))[0][0]
    wind = Wind_scaler.inverse_transform(pred_scaled[:, 3].reshape(-1, 1))[0][0]

    # ===============================
    # Display results
    # ===============================
    st.markdown("<h2>🌤️ Predicted Weather Outputs 🌤️</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Temperature (C)", f"{temp:.2f}")
    col2.metric("💧 Humidity (%)", f"{humidity:.2f}")
    col3.metric("☔ Precipitation (mm)", f"{rain:.2f}")
    col4.metric("💨 Wind Speed (km/h)", f"{wind:.2f}")

    st.markdown("---")
    st.markdown("<h2>🌱 Sustainable Recommendations 🌱</h2>", unsafe_allow_html=True)

    recs = energy_recommendations(temp, humidity, rain, wind)
    for title, desc in recs:
        st.markdown(f"""
        <div class="card">
        <h4>{title}</h4>
        <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

