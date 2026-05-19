import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import statsmodels.api as sm

st.set_page_config(
    page_title="Analiza activitatii organizatiei",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(
            135deg,
            #fff1f5 0%,
            #ffe4ec 40%,
            #ffeef3 100%
        );
    }

    h1, h2, h3 {
        color: #8b4b63;
        font-family: Arial, sans-serif;
    }

    .main-title {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(8px);
        padding: 25px;
        border-radius: 22px;
        box-shadow: 0 4px 18px rgba(255, 182, 193, 0.25);
        margin-bottom: 25px;
        border: 1px solid rgba(255,255,255,0.5);
    }

    div[data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.85);
        border-radius: 18px;
        padding: 10px;
        box-shadow: 0 2px 10px rgba(255, 182, 193, 0.15);
    }

    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(10px);
    }

    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.8);
        border-radius: 18px;
        padding: 15px;
        box-shadow: 0 2px 10px rgba(255, 182, 193, 0.15);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
    <h1>Analiza activitatii unei organizatii medicale</h1>
    <p>
        Proiect realizat in Python folosind Streamlit, Pandas,
        Scikit-learn si Statsmodels.
    </p>
</div>
""", unsafe_allow_html=True)

df = pd.read_csv("medical_students_dataset.csv")

st.header("1. Setul de date initial")
st.dataframe(df)

col1, col2 = st.columns(2)

with col1:
    st.metric("Numar randuri", df.shape[0])

with col2:
    st.metric("Numar coloane", df.shape[1])

st.subheader("Tipurile de date")
st.write(df.dtypes)

st.header("2. Tratarea valorilor lipsa")

st.subheader("Valori lipsa inainte de prelucrare")
st.write(df.isnull().sum())

df = df.replace([np.inf, -np.inf], np.nan)

numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

if "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = df["Age"].replace([np.inf, -np.inf], np.nan)
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Age"] = df["Age"].round(0).astype(int)

df.to_csv("medical_students_dataset_cleaned.csv", index=False)

st.success("Valorile lipsa au fost completate")
st.info("Fisier salvat: medical_students_dataset_cleaned.csv")

st.subheader("Valori lipsa dupa prelucrare")
st.write(df.isnull().sum())

st.header("3. Tratarea valorilor extreme")

fig1, ax1 = plt.subplots(figsize=(5, 2.5))
df.boxplot(column="BMI", ax=ax1)
ax1.set_title("Boxplot BMI")
st.pyplot(fig1)

Q1 = df["BMI"].quantile(0.25)
Q3 = df["BMI"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["BMI"] >= lower) & (df["BMI"] <= upper)]

df.to_csv("medical_students_dataset_without_outliers.csv", index=False)

st.write("Dimensiunea datasetului dupa eliminarea outlierilor:")
st.write(df.shape)

st.info("Fisier salvat: medical_students_dataset_without_outliers.csv")

st.header("4. Codificarea datelor")

encoder = LabelEncoder()

if "Gender" in df.columns:
    df["Gender"] = encoder.fit_transform(df["Gender"])

if "Diabetes" in df.columns:
    df["Diabetes"] = encoder.fit_transform(df["Diabetes"])

if "Smoking" in df.columns:
    df["Smoking"] = encoder.fit_transform(df["Smoking"])

if "Blood Type" in df.columns:
    blood_type_encoded = pd.get_dummies(
        df["Blood Type"],
        prefix="Blood"
    )

    df = pd.concat([df, blood_type_encoded], axis=1)
    df.drop("Blood Type", axis=1, inplace=True)

df.to_csv("medical_students_dataset_encoded.csv", index=False)

st.success("Datele categorice au fost codificate")
st.info("Fisier salvat: medical_students_dataset_encoded.csv")

st.dataframe(df.head())

st.header("5. Scalarea datelor")

numeric_columns_after_encoding = df.select_dtypes(include=np.number).columns

scaler_standard = StandardScaler()

scaled_standard = scaler_standard.fit_transform(
    df[numeric_columns_after_encoding]
)

df_standard = pd.DataFrame(
    scaled_standard,
    columns=numeric_columns_after_encoding
)

df_standard.to_csv("medical_students_dataset_standardized.csv", index=False)

st.subheader("StandardScaler")
st.dataframe(df_standard.head())
st.info("Fisier salvat: medical_students_dataset_standardized.csv")

scaler_minmax = MinMaxScaler()

scaled_minmax = scaler_minmax.fit_transform(
    df[numeric_columns_after_encoding]
)

df_minmax = pd.DataFrame(
    scaled_minmax,
    columns=numeric_columns_after_encoding
)

df_minmax.to_csv("medical_students_dataset_minmax.csv", index=False)

st.subheader("MinMaxScaler")
st.dataframe(df_minmax.head())
st.info("Fisier salvat: medical_students_dataset_minmax.csv")

st.header("6. Statistici descriptive")

st.subheader("Statistici generale")
st.write(df.describe())

st.subheader("Media")
st.write(df.mean(numeric_only=True))

st.subheader("Mediana")
st.write(df.median(numeric_only=True))

st.subheader("Suma")
st.write(df.sum(numeric_only=True))

statistics_df = pd.DataFrame({
    "Media": df.mean(numeric_only=True),
    "Mediana": df.median(numeric_only=True),
    "Suma": df.sum(numeric_only=True)
})

statistics_df.to_csv("medical_students_statistics.csv")

st.info("Fisier salvat: medical_students_statistics.csv")

st.header("7. Grupare si agregare")

group_gender = df.groupby("Gender")["BMI"].mean().reset_index()
group_gender.columns = ["Gender", "BMI_Mediu"]

st.subheader("Media BMI in functie de gen")
st.write(group_gender)

group_gender.to_csv("medical_students_group_gender.csv", index=False)

group_smoking = df.groupby("Smoking")["Cholesterol"].mean().reset_index()
group_smoking.columns = ["Smoking", "Colesterol_Mediu"]

st.subheader("Media colesterolului in functie de fumat")
st.write(group_smoking)

group_smoking.to_csv("medical_students_group_smoking.csv", index=False)

st.info("Fisiere salvate: medical_students_group_gender.csv si medical_students_group_smoking.csv")

st.header("8. Histograma")

fig2, ax2 = plt.subplots(figsize=(5, 2.5))

ax2.hist(df["Age"], bins=10)

ax2.set_title("Distributia varstei")
ax2.set_xlabel("Varsta")
ax2.set_ylabel("Frecventa")

st.pyplot(fig2)

st.header("9. Clusterizare KMeans")

features = df[["Age", "BMI", "Heart Rate"]]

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

df["Cluster"] = kmeans.fit_predict(features)

cluster_results = df[[
    "Age",
    "BMI",
    "Heart Rate",
    "Cluster"
]]

cluster_results.to_csv("medical_students_kmeans_clusters.csv", index=False)

st.write(cluster_results.head())

st.info("Fisier salvat: medical_students_kmeans_clusters.csv")

fig3, ax3 = plt.subplots(figsize=(5, 3))

ax3.scatter(
    df["Age"],
    df["BMI"],
    c=df["Cluster"]
)

ax3.set_xlabel("Age")
ax3.set_ylabel("BMI")
ax3.set_title("Clusterizare KMeans")

st.pyplot(fig3)

st.header("10. Regresie logistica")

X = df[["Age", "BMI", "Heart Rate"]]
y = df["Diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

st.subheader("Acuratete model")
st.metric("Accuracy", round(accuracy, 2))

st.subheader("Matrice de confuzie")
conf_matrix = confusion_matrix(y_test, predictions)
st.write(conf_matrix)

logistic_results = pd.DataFrame({
    "Valori reale": y_test.values,
    "Predictii": predictions
})

logistic_results.to_csv("medical_students_logistic_regression_results.csv", index=False)

confusion_df = pd.DataFrame(conf_matrix)
confusion_df.to_csv("medical_students_confusion_matrix.csv", index=False)

st.info("Fisiere salvate: medical_students_logistic_regression_results.csv si medical_students_confusion_matrix.csv")

st.header("11. Regresie multipla cu Statsmodels")

X_stats = df[[
    "Age",
    "BMI",
    "Heart Rate"
]]

X_stats = sm.add_constant(X_stats)

y_stats = df["Cholesterol"]

model_stats = sm.OLS(
    y_stats,
    X_stats
).fit()

st.text(model_stats.summary())

statsmodels_params = pd.DataFrame({
    "Coeficient": model_stats.params,
    "P_value": model_stats.pvalues
})

statsmodels_params.to_csv("medical_students_statsmodels_regression.csv")

st.info("Fisier salvat: medical_students_statsmodels_regression.csv")

st.header("12. Salvarea datasetului final")

df.to_csv(
    "medical_students_dataset_processed.csv",
    index=False
)

st.success("Fisierul final a fost salvat: medical_students_dataset_processed.csv")