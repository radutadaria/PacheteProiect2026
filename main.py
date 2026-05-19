import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import statsmodels.api as sm


st.set_page_config(
    page_title="Analiza activitatii unei organizatii medicale",
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
        Matplotlib, Seaborn, Scikit-learn si Statsmodels.
    </p>
</div>
""", unsafe_allow_html=True)


file_path = "medical_students_dataset.csv"

df_initial = pd.read_csv(file_path)
df = df_initial.copy()

df = df.replace([np.inf, -np.inf], np.nan)

numeric_columns_initial = df.select_dtypes(include=np.number).columns

for col in numeric_columns_initial:
    df[col] = df[col].fillna(df[col].mean())

categorical_columns_initial = df.select_dtypes(include="object").columns

for col in categorical_columns_initial:
    df[col] = df[col].fillna(df[col].mode()[0])

if "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = df["Age"].replace([np.inf, -np.inf], np.nan)
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Age"] = df["Age"].round(0).astype(int)

df_cleaned = df.copy()
df_cleaned.to_csv("medical_students_dataset_cleaned.csv", index=False)

if "BMI" in df_cleaned.columns:
    Q1 = df_cleaned["BMI"].quantile(0.25)
    Q3 = df_cleaned["BMI"].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_cleaned = df_cleaned[
        (df_cleaned["BMI"] >= lower) &
        (df_cleaned["BMI"] <= upper)
    ]

df_cleaned.to_csv("medical_students_dataset_without_outliers.csv", index=False)

df_encoded = df_cleaned.copy()

encoder = LabelEncoder()

for col in ["Gender", "Diabetes", "Smoking"]:
    if col in df_encoded.columns:
        df_encoded[col] = encoder.fit_transform(df_encoded[col])

if "Blood Type" in df_encoded.columns:
    blood_type_encoded = pd.get_dummies(
        df_encoded["Blood Type"],
        prefix="Blood",
        dtype=int
    )

    df_encoded = pd.concat(
        [df_encoded, blood_type_encoded],
        axis=1
    )

    df_encoded.drop(
        "Blood Type",
        axis=1,
        inplace=True
    )

df_encoded.to_csv("medical_students_dataset_encoded.csv", index=False)

numeric_columns = df_encoded.select_dtypes(include=np.number).columns

scaler_standard = StandardScaler()

df_standardized = df_encoded.copy()

df_standardized[numeric_columns] = scaler_standard.fit_transform(
    df_standardized[numeric_columns]
)

df_standardized[numeric_columns] = df_standardized[numeric_columns].round(4)

df_standardized.to_csv("medical_students_dataset_standardized_output.csv", index=False)

scaler_minmax = MinMaxScaler()

df_minmax = df_encoded.copy()

df_minmax[numeric_columns] = scaler_minmax.fit_transform(
    df_minmax[numeric_columns]
)

df_minmax[numeric_columns] = df_minmax[numeric_columns].round(4)

df_minmax.to_csv("medical_students_dataset_minmax_output.csv", index=False)


st.sidebar.title("Navigare")

option = st.sidebar.radio(
    "Alege sectiunea:",
    [
        "Set de date",
        "Valori lipsa si outliers",
        "Codificare si scalare",
        "Statistici descriptive",
        "Grupare si agregare",
        "Histograma",
        "Prelucrari avansate",
        "Interactiuni coloane",
        "Vizualizare outliers"
    ]
)


if option == "Set de date":

    st.header("1. Setul de date initial")

    st.dataframe(
        df_initial.head(100),
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Numar randuri", df_initial.shape[0])

    with col2:
        st.metric("Numar coloane", df_initial.shape[1])

    with col3:
        st.metric("Valori lipsa totale", int(df_initial.isnull().sum().sum()))

    st.subheader("Tipurile de date")

    st.write(df_initial.dtypes)

    st.subheader("Primele 10 randuri dupa curatare")

    st.dataframe(
        df_cleaned.head(10),
        use_container_width=True
    )


if option == "Valori lipsa si outliers":

    st.header("2. Tratarea valorilor lipsa")

    st.subheader("Valori lipsa inainte de prelucrare")

    st.dataframe(
        df_initial.isnull().sum().reset_index().rename(
            columns={
                "index": "Coloana",
                0: "Numar valori lipsa"
            }
        ),
        use_container_width=True
    )

    st.subheader("Valori lipsa dupa prelucrare")

    st.dataframe(
        df_cleaned.isnull().sum().reset_index().rename(
            columns={
                "index": "Coloana",
                0: "Numar valori lipsa"
            }
        ),
        use_container_width=True
    )

    st.header("3. Tratarea valorilor extreme")

    fig1, ax1 = plt.subplots(figsize=(4, 2))

    sns.boxplot(
        x=df_initial["BMI"],
        ax=ax1
    )

    ax1.set_title("Boxplot BMI inainte de tratarea outlierilor")

    st.pyplot(fig1)

    plt.close()

    fig2, ax2 = plt.subplots(figsize=(4, 2))

    sns.boxplot(
        x=df_cleaned["BMI"],
        ax=ax2
    )

    ax2.set_title("Boxplot BMI dupa tratarea outlierilor")

    st.pyplot(fig2)

    plt.close()

    st.success("Dataseturile curatate au fost salvate in fisiere CSV.")


if option == "Codificare si scalare":

    st.header("4. Codificarea datelor")

    st.subheader("Date dupa codificare")

    st.dataframe(
        df_encoded.head(10),
        use_container_width=True
    )

    st.info("Variabilele categorice au fost transformate in variabile numerice.")

    st.header("5. Scalarea datelor")

    st.subheader("StandardScaler")

    st.dataframe(
        df_standardized.head(10),
        use_container_width=True
    )

    st.subheader("MinMaxScaler")

    st.dataframe(
        df_minmax.head(10),
        use_container_width=True
    )


if option == "Statistici descriptive":

    st.header("6. Statistici descriptive")

    stats = df_encoded.describe().round(4)

    st.subheader("Statistici generale")

    st.dataframe(
        stats,
        use_container_width=True
    )

    st.subheader("Media valorilor numerice")

    mean_values = df_encoded.mean(numeric_only=True).round(4)

    st.dataframe(
        mean_values.reset_index().rename(
            columns={
                "index": "Coloana",
                0: "Media"
            }
        ),
        use_container_width=True
    )

    st.subheader("Mediana valorilor numerice")

    median_values = df_encoded.median(numeric_only=True).round(4)

    st.dataframe(
        median_values.reset_index().rename(
            columns={
                "index": "Coloana",
                0: "Mediana"
            }
        ),
        use_container_width=True
    )

    st.subheader("Suma valorilor numerice")

    sum_values = df_encoded.sum(numeric_only=True).round(4)

    st.dataframe(
        sum_values.reset_index().rename(
            columns={
                "index": "Coloana",
                0: "Suma"
            }
        ),
        use_container_width=True
    )

    statistics_df = pd.DataFrame({
        "Media": mean_values,
        "Mediana": median_values,
        "Suma": sum_values
    })

    statistics_df.to_csv("medical_students_statistics.csv")


if option == "Grupare si agregare":

    st.header("7. Grupare si agregare")

    st.subheader("Agregare pe gen si fumat")

    agg_gender_smoking = df_encoded.groupby(
        ["Gender", "Smoking"]
    )[[
        "BMI",
        "Heart Rate",
        "Blood Pressure",
        "Cholesterol"
    ]].agg([
        "mean",
        "sum",
        "min",
        "max",
        "std"
    ]).round(4)

    st.dataframe(
        agg_gender_smoking,
        use_container_width=True
    )

    agg_gender_smoking.to_csv("medical_students_group_gender_smoking.csv")

    st.subheader("Agregare pe diabet si gen")

    agg_diabetes_gender = df_encoded.groupby(
        ["Diabetes", "Gender"]
    )[[
        "BMI",
        "Blood Pressure",
        "Cholesterol"
    ]].agg([
        "mean",
        "sum",
        "min",
        "max",
        "std"
    ]).round(4)

    st.dataframe(
        agg_diabetes_gender,
        use_container_width=True
    )

    agg_diabetes_gender.to_csv("medical_students_group_diabetes_gender.csv")

    st.subheader("Agregare pe grupe de varsta")

    df_age_groups = df_encoded.copy()

    bins = [0, 20, 30, 40, 100]
    labels = ["sub 20", "20-30", "30-40", "peste 40"]

    df_age_groups["Age Group"] = pd.cut(
        df_age_groups["Age"],
        bins=bins,
        labels=labels,
        right=False
    )

    age_group_agg = df_age_groups.groupby(
        "Age Group",
        observed=False
    )[[
        "BMI",
        "Heart Rate",
        "Blood Pressure",
        "Cholesterol"
    ]].agg([
        "mean",
        "sum",
        "count",
        "std"
    ]).round(4)

    st.dataframe(
        age_group_agg,
        use_container_width=True
    )

    age_group_agg.to_csv("medical_students_group_age.csv")

    st.subheader("Numar de fumatori si persoane cu diabet pe grupe de varsta")

    count_smoke_diabetes = df_age_groups.groupby(
        "Age Group",
        observed=False
    )[[
        "Smoking",
        "Diabetes"
    ]].sum().round(4)

    st.dataframe(
        count_smoke_diabetes,
        use_container_width=True
    )

    count_smoke_diabetes.to_csv("medical_students_smoking_diabetes_age.csv")


if option == "Histograma":

    st.header("8. Histograma")

    numeric_columns_list = df_encoded.select_dtypes(
        include=np.number
    ).columns.tolist()

    selected_column = st.selectbox(
        "Alege o coloana numerica:",
        numeric_columns_list
    )

    fig3, ax3 = plt.subplots(figsize=(4, 2.2))

    ax3.hist(
        df_encoded[selected_column].dropna(),
        bins=15
    )

    ax3.set_title(f"Distributia variabilei {selected_column}")

    ax3.set_xlabel(selected_column)

    ax3.set_ylabel("Frecventa")

    st.pyplot(fig3)

    plt.close()


if option == "Prelucrari avansate":

    st.header("9. Prelucrari avansate")

    st.subheader("Clusterizare PCA + KMeans")

    numeric_cols_for_models = [
        "Height",
        "Weight",
        "BMI",
        "Temperature",
        "Heart Rate",
        "Blood Pressure",
        "Cholesterol"
    ]

    numeric_cols_for_models = [
        col for col in numeric_cols_for_models
        if col in df_standardized.columns
    ]

    k = st.slider(
        "Alege numarul de clustere:",
        2,
        10,
        3
    )

    pca = PCA(n_components=2)

    data_for_cluster = df_standardized[numeric_cols_for_models]

    reduced_data = pca.fit_transform(data_for_cluster)

    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    cluster_labels = kmeans.fit_predict(reduced_data)

    cluster_df = pd.DataFrame({
        "PCA1": reduced_data[:, 0],
        "PCA2": reduced_data[:, 1],
        "Cluster": cluster_labels
    })

    cluster_df.to_csv("medical_students_pca_kmeans.csv", index=False)

    fig4, ax4 = plt.subplots(figsize=(4.5, 3))

    scatter = ax4.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=cluster_labels,
        cmap="viridis",
        s=8
    )

    ax4.set_title("Rezultatul clusterizarii PCA + KMeans")

    ax4.set_xlabel("Componenta principala 1")

    ax4.set_ylabel("Componenta principala 2")

    st.pyplot(fig4)

    plt.close()

    st.write("Centrele clusterelor:")

    st.dataframe(
        pd.DataFrame(kmeans.cluster_centers_),
        use_container_width=True
    )

    st.subheader("Regresie logistica pentru predictia diabetului")

    X_logreg = df_standardized.drop(
        columns=["Diabetes"],
        errors="ignore"
    )

    y_logreg = df_encoded["Diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_logreg,
        y_logreg,
        test_size=0.2,
        random_state=42
    )

    logreg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    logreg.fit(
        X_train,
        y_train
    )

    y_pred = logreg.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    st.write("Acuratetea modelului:")

    st.metric(
        "Accuracy",
        round(accuracy, 4)
    )

    conf_matrix = confusion_matrix(
        y_test,
        y_pred
    )

    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        columns=["Predicted 0", "Predicted 1"],
        index=["Real 0", "Real 1"]
    )

    st.write("Matricea de confuzie:")

    st.dataframe(
        conf_matrix_df,
        use_container_width=True
    )

    conf_matrix_df.to_csv("medical_students_logistic_confusion_matrix.csv")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose().round(4)

    st.write("Classification report:")

    st.dataframe(
        report_df,
        use_container_width=True
    )

    report_df.to_csv("medical_students_logistic_report.csv")

    st.subheader("Random Forest pentru predictia diabetului")

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    rf_model.fit(
        X_train,
        y_train
    )

    y_pred_rf = rf_model.predict(X_test)

    accuracy_rf = accuracy_score(
        y_test,
        y_pred_rf
    )

    st.metric(
        "Accuracy Random Forest",
        round(accuracy_rf, 4)
    )

    conf_matrix_rf = confusion_matrix(
        y_test,
        y_pred_rf
    )

    conf_matrix_rf_df = pd.DataFrame(
        conf_matrix_rf,
        columns=["Predicted 0", "Predicted 1"],
        index=["Real 0", "Real 1"]
    )

    st.dataframe(
        conf_matrix_rf_df,
        use_container_width=True
    )

    conf_matrix_rf_df.to_csv("medical_students_random_forest_confusion_matrix.csv")

    report_rf = classification_report(
        y_test,
        y_pred_rf,
        output_dict=True
    )

    report_rf_df = pd.DataFrame(report_rf).transpose().round(4)

    st.dataframe(
        report_rf_df,
        use_container_width=True
    )

    report_rf_df.to_csv("medical_students_random_forest_report.csv")

    st.subheader("Regresie liniara multipla cu Statsmodels")

    X_multi = df_encoded[[
        "Age",
        "BMI",
        "Heart Rate"
    ]]

    y_multi = df_encoded["Cholesterol"]

    X_multi = sm.add_constant(X_multi)

    model_multi = sm.OLS(
        y_multi,
        X_multi
    ).fit()

    st.text(
        model_multi.summary()
    )

    statsmodels_params = pd.DataFrame({
        "Coeficient": model_multi.params,
        "P_value": model_multi.pvalues
    })

    statsmodels_params.to_csv("medical_students_statsmodels_regression.csv")


if option == "Interactiuni coloane":

    st.header("10. Interactiuni coloane DataFrame")

    if "df_interactive" not in st.session_state:
        st.session_state.df_interactive = df_standardized.copy()

    df_interactive = st.session_state.df_interactive

    st.subheader("DataFrame curent")

    st.dataframe(
        df_interactive.head(50),
        use_container_width=True
    )

    col_names = list(df_interactive.columns)

    selected_col = st.selectbox(
        "Selecteaza o coloana:",
        col_names
    )

    st.subheader(f"Operatii pentru coloana: {selected_col}")

    new_name = st.text_input(
        "Introdu noul nume pentru coloana:",
        value=selected_col
    )

    if st.button("Redenumeste coloana"):

        if new_name and new_name != selected_col:

            st.session_state.df_interactive = (
                st.session_state.df_interactive.rename(
                    columns={selected_col: new_name}
                )
            )

            st.success("Coloana a fost redenumita.")

            st.rerun()

        else:
            st.info("Noul nume trebuie sa fie diferit de cel curent.")

    if st.button("Sterge coloana"):

        st.session_state.df_interactive = (
            st.session_state.df_interactive.drop(
                columns=[selected_col]
            )
        )

        st.success("Coloana a fost stearsa.")

        st.rerun()

    if st.button("Afiseaza numarul de valori lipsa"):

        missing_count = (
            st.session_state.df_interactive[selected_col]
            .isna()
            .sum()
        )

        st.info(
            f"Coloana {selected_col} are {missing_count} valori lipsa."
        )


if option == "Vizualizare outliers":

    st.header("11. Vizualizare outliers")

    columns_for_boxplot = [
        "Height",
        "Weight",
        "BMI",
        "Temperature",
        "Heart Rate",
        "Blood Pressure",
        "Cholesterol"
    ]

    columns_for_boxplot = [
        col for col in columns_for_boxplot
        if col in df_encoded.columns
    ]

    selected_box_cols = st.multiselect(
        "Alege coloanele pentru boxplot:",
        columns_for_boxplot,
        default=columns_for_boxplot[:3]
    )

    if selected_box_cols:

        for col in selected_box_cols:

            fig, ax = plt.subplots(figsize=(4, 2))

            sns.boxplot(
                x=df_encoded[col],
                ax=ax
            )

            ax.set_title(f"Boxplot pentru {col}")

            st.pyplot(fig)

            plt.close()