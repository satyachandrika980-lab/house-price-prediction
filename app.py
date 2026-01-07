import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------
# Background & Animations
# ---------------------------
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Gradient overlay */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                135deg,
                #405DE6,
                #5851DB,
                #833AB4,
                #C13584,
                #E1306C,
                #FD1D1D,
                #F56040,
                #F77737,
                #FCAF45,
                #FFDC80
            );
            opacity: 0.85;
            z-index: -1;
        }}

        /* Floating Balloons */
        .balloon {{
            position: fixed;
            bottom: -150px;
            width: 60px;
            height: 80px;
            background: radial-gradient(circle at 30% 30%, #ffcccc, #ff6666);
            border-radius: 50% 50% 50% 50%;
            animation: floatUp 12s infinite ease-in-out;
            z-index: 0;
        }}

        .balloon::after {{
            content: "";
            position: absolute;
            bottom: -20px;
            left: 50%;
            width: 2px;
            height: 20px;
            background: #333;
        }}

        @keyframes floatUp {{
            0% {{ transform: translateY(0) rotate(0deg); opacity: 1; }}
            50% {{ transform: translateY(-400px) rotate(10deg); opacity: 0.8; }}
            100% {{ transform: translateY(-800px) rotate(-10deg); opacity: 0; }}
        }}

        /* Multiple balloons with delays */
        .balloon:nth-child(1) {{ left: 10%; animation-delay: 0s; }}
        .balloon:nth-child(2) {{ left: 30%; animation-delay: 3s; }}
        .balloon:nth-child(3) {{ left: 50%; animation-delay: 6s; }}
        .balloon:nth-child(4) {{ left: 70%; animation-delay: 9s; }}
        .balloon:nth-child(5) {{ left: 90%; animation-delay: 12s; }}

        /* Text colors */
        h1, h2, h3, p, label {{
            color: #000000;
        }}

        /* Buttons */
        .stButton>button {{
            background: linear-gradient(45deg, #405DE6, #833AB4);
            color: white;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            transform: scale(1.05);
            background: linear-gradient(45deg, #833AB4, #405DE6);
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #833AB4, #405DE6);
            color: white;
        }}
        </style>

        <!-- Balloon elements -->
        <div class="balloon"></div>
        <div class="balloon"></div>
        <div class="balloon"></div>
        <div class="balloon"></div>
        <div class="balloon"></div>
        """,
        unsafe_allow_html=True
    )

# ✅ ADD BACKGROUND IMAGE HERE
set_background(r"D:\html\bg.jpg")

# ---------------------------
# Header
# ---------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🏠 House Price Prediction System</h1>
    <p style='text-align: center;'>AI-powered real estate price estimation</p>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_csv("House Price Prediction Dataset.csv")

df = load_data()

# ---------------------------
# Sidebar Navigation
# ---------------------------
menu = st.sidebar.radio(
    "📌 Navigation",
    ["Home", "Predict Price", "Data Insights"]
)

# ---------------------------
# Encode Data
# ---------------------------
data = df.copy()
encoders = {}

for col in ["Location", "Condition", "Garage"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

X = data.drop(columns=["Id", "Price"])
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)

# ---------------------------
# HOME PAGE
# ---------------------------
if menu == "Home":
    col1, col2, col3 = st.columns(3)

    col1.metric("🏘 Total Houses", len(df))
    col2.metric("💰 Avg Price", f"${df['Price'].mean():,.0f}")
    col3.metric("📐 Avg Area", f"{df['Area'].mean():.0f} sq ft")

    st.markdown("---")
    st.subheader("📊 Price Distribution")

    fig, ax = plt.subplots()
    sns.histplot(df["Price"], kde=True, ax=ax, color="green")
    ax.set_xlabel("House Price")
    st.pyplot(fig)

# ---------------------------
# PREDICTION PAGE
# ---------------------------
elif menu == "Predict Price":
    st.subheader("🔮 Predict House Price")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area (sq ft)", 300, 10000, 1500)
        bedrooms = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1, 5, 2)
        floors = st.slider("Floors", 1, 4, 1)

    with col2:
        year = st.slider("Year Built", 1900, 2025, 2000)
        location = st.selectbox("Location", df["Location"].unique())
        condition = st.selectbox("Condition", df["Condition"].unique())
        garage = st.selectbox("Garage", df["Garage"].unique())

    if st.button("💰 Predict Price"):
        user_data = np.array([[ 
            area,
            bedrooms,
            bathrooms,
            floors,
            year,
            encoders["Location"].transform([location])[0],
            encoders["Condition"].transform([condition])[0],
            encoders["Garage"].transform([garage])[0]
        ]])

        prediction = model.predict(user_data)[0]
        st.success(f"🏷 Estimated House Price: **${prediction:,.2f}**")
        st.info(f"📈 Model Accuracy (R²): {score:.2f}")

# ---------------------------
# DATA INSIGHTS PAGE
# ---------------------------
elif menu == "Data Insights":
    st.subheader("📊 Dataset Overview")
    st.dataframe(df)

    st.markdown("---")
    st.subheader("🏗 Feature Importance")

    importance = model.feature_importances_
    features = X.columns

    fig, ax = plt.subplots()
    sns.barplot(x=importance, y=features, ax=ax, palette="viridis")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("🚀 Developed using Streamlit & Machine Learning 🎈")