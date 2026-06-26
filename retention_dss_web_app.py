# ==========================================================
# CUSTOMER RETENTION DECISION SUPPORT SYSTEM (WEB-BASED)
# Streamlit + Machine Learning + Closed-Loop DSS Simulation
# ==========================================================

import io
import time
import zipfile
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="Customer Retention DSS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown(
    """
    <style>
        .main {
            background-color: #f6f8fb;
        }
        .hero-box {
            padding: 28px;
            border-radius: 20px;
            background: linear-gradient(135deg, #102a43, #1f6f8b);
            color: white;
            margin-bottom: 25px;
        }
        .hero-title {
            font-size: 34px;
            font-weight: 800;
            margin-bottom: 6px;
        }
        .hero-subtitle {
            font-size: 17px;
            opacity: 0.92;
        }
        .metric-card {
            padding: 20px;
            border-radius: 16px;
            background-color: white;
            box-shadow: 0 4px 14px rgba(0,0,0,0.07);
            text-align: center;
        }
        .section-card {
            padding: 20px;
            border-radius: 16px;
            background-color: white;
            box-shadow: 0 4px 14px rgba(0,0,0,0.06);
            margin-bottom: 18px;
        }
        .small-note {
            font-size: 14px;
            color: #52616b;
        }
        .success-text {
            color: #15803d;
            font-weight: 700;
        }
        .warning-text {
            color: #b45309;
            font-weight: 700;
        }
        .danger-text {
            color: #b91c1c;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# GLOBAL FEATURE COLUMNS
# ==========================================================
FEATURE_COLUMNS = [
    "GENDER",
    "HIGH_SWITCHING_ENERGY_TIME",
    "HIGH_SWITCHING_COST",
    "VALUE_FOR_MONEY",
    "WIDE_COVERAGE",
    "CALL_DROPS",
    "STRONG_SIGNALS",
    "PROBLEM_SOLVING",
    "EXCEPTIONAL_SERVICE_EXPERIENCE",
    "GET_THROUGH",
    "DO_WHAT_THEY_SAY",
    "TIMELY_EFFECTIVE_COMPLAINTS",
    "FREE_COMPLAINTS",
    "TENURE_CLASS",
    "EARNINGS_CLASS",
    "LOAN_BOARD_CLASS",
]

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def initialize_session_state():
    defaults = {
        "df": None,
        "results_df": None,
        "best_model_name": None,
        "best_model": None,
        "scaler": None,
        "iteration_df": None,
        "explanation_df": None,
        "final_data": None,
        "classification_report_df": None,
        "confusion_matrix_df": None,
        "risk_summary_df": None,
        "seed_used": None,
        "pattern_type": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def risk_level(probability):
    if probability >= 0.75:
        return "HIGH RISK"
    elif probability >= 0.5:
        return "MEDIUM RISK"
    return "LOW RISK"


def generate_synthetic_data(n_rows=2000, seed=None, pattern_type="Random"):
    if seed is None:
        seed = int(time.time()) % 100000

    rng = np.random.default_rng(seed)

    data = np.zeros((n_rows, len(FEATURE_COLUMNS)), dtype=float)

    for i in range(len(FEATURE_COLUMNS)):
        p1 = rng.uniform(0.3, 0.7)
        data[:, i] = rng.choice([0, 1], size=n_rows, p=[1 - p1, p1])

    df = pd.DataFrame(data, columns=FEATURE_COLUMNS)

    for col in ["VALUE_FOR_MONEY", "CALL_DROPS", "PROBLEM_SOLVING"]:
        df[col] = df[col] + rng.normal(0, 0.5, n_rows)

    if pattern_type == "Random":
        pattern_type = rng.choice(["linear", "nonlinear", "tree", "noisy"])

    weights = rng.uniform(-0.8, 0.8, len(FEATURE_COLUMNS))

    if pattern_type == "linear":
        score = np.dot(df[FEATURE_COLUMNS], weights)

    elif pattern_type == "nonlinear":
        interaction1 = df["CALL_DROPS"] * df["PROBLEM_SOLVING"]
        interaction2 = df["VALUE_FOR_MONEY"] * df["WIDE_COVERAGE"]
        score = (
            np.dot(df[FEATURE_COLUMNS], weights)
            + 2.5 * interaction1
            - 2.0 * interaction2
            + np.sin(df["TENURE_CLASS"] * 3)
        )

    elif pattern_type == "tree":
        score = np.where(
            (df["CALL_DROPS"] > 0.5) & (df["PROBLEM_SOLVING"] < 0.5),
            4,
            -2,
        ).astype(float)
        score += np.where(
            (df["VALUE_FOR_MONEY"] > 0.5) & (df["STRONG_SIGNALS"] > 0.5),
            -3,
            2,
        )

    else:
        score = np.dot(df[FEATURE_COLUMNS], weights) + rng.normal(0, 2, n_rows)

    score += rng.normal(0, 1.5, n_rows)
    probability = 1 / (1 + np.exp(-score))
    df["CLASS"] = (probability > 0.55).astype(int)

    return df, seed, pattern_type


def prepare_uploaded_data(df):
    """
    Prepare uploaded CSV data for the DSS system.

    This version solves common upload problems:
    1. Cleans column names by removing spaces.
    2. Converts column names to uppercase.
    3. Accepts common target names such as CHURN, churn, TARGET, STATUS, and RETENTION_STATUS.
    4. Automatically creates missing feature columns with default value 0.
    """
    prepared_df = df.copy()

    # Clean and standardize column names
    prepared_df.columns = (
        prepared_df.columns
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    # Possible target column names in uploaded datasets
    possible_target_columns = [
        "CLASS",
        "CHURN",
        "CUSTOMER_CHURN",
        "CHURN_STATUS",
        "TARGET",
        "LABEL",
        "STATUS",
        "RETENTION_STATUS",
        "RISK",
        "RISK_STATUS",
    ]

    target_column = None
    for col in possible_target_columns:
        if col in prepared_df.columns:
            target_column = col
            break

    if target_column is None:
        return (
            None,
            False,
            "Dataset must contain a target column. Accepted target names include CLASS, CHURN, TARGET, LABEL, STATUS, or RISK.",
        )

    # Rename target column to CLASS for model training
    if target_column != "CLASS":
        prepared_df = prepared_df.rename(columns={target_column: "CLASS"})

    missing_features = []

    # Add missing feature columns automatically
    for col in FEATURE_COLUMNS:
        if col not in prepared_df.columns:
            prepared_df[col] = 0
            missing_features.append(col)

    # Keep required columns only, in correct order
    prepared_df = prepared_df[FEATURE_COLUMNS + ["CLASS"]]

    # Convert feature columns to numeric values
    for col in FEATURE_COLUMNS:
        prepared_df[col] = pd.to_numeric(prepared_df[col], errors="coerce").fillna(0)

    # Convert CLASS column to 0/1 even if it contains text values
    class_series = prepared_df["CLASS"]

    if class_series.dtype == "object":
        class_series = (
            class_series
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({
                "yes": 1,
                "y": 1,
                "true": 1,
                "churn": 1,
                "churned": 1,
                "high": 1,
                "high risk": 1,
                "risk": 1,
                "at risk": 1,
                "no": 0,
                "n": 0,
                "false": 0,
                "not churn": 0,
                "not_churn": 0,
                "retained": 0,
                "low": 0,
                "low risk": 0,
                "normal": 0,
            })
        )

    prepared_df["CLASS"] = pd.to_numeric(class_series, errors="coerce").fillna(0)
    prepared_df["CLASS"] = prepared_df["CLASS"].apply(lambda x: 1 if float(x) >= 1 else 0).astype(int)

    if prepared_df["CLASS"].nunique() < 2:
        return (
            None,
            False,
            "The target column must contain both classes 0 and 1. Please check your CLASS/CHURN/TARGET column values.",
        )

    if missing_features:
        message = (
            "Dataset uploaded successfully. These missing columns were automatically added with default value 0: "
            + ", ".join(missing_features)
        )
    else:
        message = "Dataset uploaded successfully. All required columns are available."

    return prepared_df, True, message


def build_models(seed):
    rng = np.random.default_rng(seed)

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=int(rng.integers(500, 1500))
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=int(rng.integers(3, 15)),
            min_samples_split=int(rng.integers(2, 10)),
            random_state=seed,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=int(rng.integers(30, 150)),
            learning_rate=float(rng.uniform(0.3, 1.5)),
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=int(rng.integers(20, 200)),
            max_depth=int(rng.integers(3, 20)),
            min_samples_split=int(rng.integers(2, 10)),
            random_state=seed,
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(
                int(rng.integers(20, 150)),
                int(rng.integers(10, 100)),
            ),
            learning_rate_init=float(rng.uniform(0.0005, 0.01)),
            max_iter=int(rng.integers(300, 1000)),
            random_state=seed,
        ),
    }


def train_and_evaluate_models(df, test_size=0.3, seed=42):
    X = df[FEATURE_COLUMNS].copy()
    y = df["CLASS"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y if y.nunique() == 2 else None,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = build_models(seed)
    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = y_pred

        try:
            auc_value = roc_auc_score(y_test, y_prob)
        except Exception:
            auc_value = 0

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                "AUC": auc_value,
            }
        )
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    best_pred = best_model.predict(X_test_scaled)
    report = classification_report(y_test, best_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "Class"})

    cm = confusion_matrix(y_test, best_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )

    return results_df, best_model_name, best_model, scaler, report_df, cm_df


def recalc_risk(df, model, scaler):
    working_df = df.copy()
    X_scaled = scaler.transform(working_df[FEATURE_COLUMNS])
    working_df["Risk_Prob"] = model.predict_proba(X_scaled)[:, 1]
    working_df["Prediction"] = model.predict(X_scaled)
    working_df["Risk_Level"] = working_df["Risk_Prob"].apply(risk_level)
    return working_df


def explain_iteration(iteration, before, after):
    high_change = (
        (after["Risk_Level"] == "HIGH RISK").sum()
        - (before["Risk_Level"] == "HIGH RISK").sum()
    )
    medium_change = (
        (after["Risk_Level"] == "MEDIUM RISK").sum()
        - (before["Risk_Level"] == "MEDIUM RISK").sum()
    )
    low_change = (
        (after["Risk_Level"] == "LOW RISK").sum()
        - (before["Risk_Level"] == "LOW RISK").sum()
    )

    if high_change < 0:
        interpretation = "Interventions reduced high-risk customers."
    elif high_change == 0 and medium_change < 0:
        interpretation = "High risk remained stable while medium risk decreased."
    elif high_change == 0:
        interpretation = "High risk remained stable; intervention impact was limited."
    else:
        interpretation = "Risk increased; intervention was less effective."

    return {
        "Iteration": iteration,
        "High Risk Change": high_change,
        "Medium Risk Change": medium_change,
        "Low Risk Change": low_change,
        "Interpretation": interpretation,
        "Framework Flow": "Inputs → Predictive Model → DSS Interpretation → Intervention → Updated Risk",
    }


def run_closed_loop_simulation(model, scaler, n_customers=500, iterations=10, seed=42):
    rng = np.random.default_rng(seed)

    new_data = pd.DataFrame(
        rng.integers(0, 2, size=(n_customers, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    ).astype(float)

    current_data = recalc_risk(new_data, model, scaler)
    iteration_results = []
    iteration_explanations = []

    for i in range(iterations):
        before = current_data.copy()
        editable_features = current_data[FEATURE_COLUMNS].copy().astype(float)

        high_mask = current_data["Risk_Level"] == "HIGH RISK"
        medium_mask = current_data["Risk_Level"] == "MEDIUM RISK"
        low_mask = current_data["Risk_Level"] == "LOW RISK"

        for col in FEATURE_COLUMNS:
            if high_mask.sum() > 0:
                editable_features.loc[high_mask, col] = rng.choice(
                    [0, 1], size=high_mask.sum(), p=[0.7, 0.3]
                )
            if medium_mask.sum() > 0:
                editable_features.loc[medium_mask, col] = rng.choice(
                    [0, 1], size=medium_mask.sum(), p=[0.5, 0.5]
                )
            if low_mask.sum() > 0:
                editable_features.loc[low_mask, col] = rng.choice(
                    [0, 1], size=low_mask.sum(), p=[0.3, 0.7]
                )

        current_data = recalc_risk(editable_features, model, scaler)
        after = current_data.copy()

        iteration_results.append(
            {
                "Iteration": i + 1,
                "High Risk Customers": int((after["Risk_Level"] == "HIGH RISK").sum()),
                "Medium Risk Customers": int((after["Risk_Level"] == "MEDIUM RISK").sum()),
                "Low Risk Customers": int((after["Risk_Level"] == "LOW RISK").sum()),
            }
        )
        iteration_explanations.append(explain_iteration(i + 1, before, after))

    iteration_df = pd.DataFrame(iteration_results)
    explanation_df = pd.DataFrame(iteration_explanations)
    risk_summary_df = current_data["Risk_Level"].value_counts().reset_index()
    risk_summary_df.columns = ["Risk Level", "Number of Customers"]

    return iteration_df, explanation_df, current_data, risk_summary_df


def make_high_risk_chart(iteration_df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        iteration_df["Iteration"],
        iteration_df["High Risk Customers"],
        marker="o",
        linewidth=2,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("High Risk Customers")
    ax.set_title("High Risk Reduction Over Iterations")
    ax.grid(True, alpha=0.3)
    return fig


def make_risk_distribution_chart(risk_summary_df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(risk_summary_df["Risk Level"], risk_summary_df["Number of Customers"])
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Final Customer Risk Distribution")
    ax.grid(axis="y", alpha=0.3)
    return fig


def make_model_performance_chart(results_df):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(results_df["Model"], results_df["AUC"])
    ax.set_xlabel("Model")
    ax.set_ylabel("AUC Score")
    ax.set_title("Model Comparison by AUC")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    return fig


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def create_excel_report():
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        if st.session_state.results_df is not None:
            st.session_state.results_df.to_excel(writer, sheet_name="Model Performance", index=False)
        if st.session_state.iteration_df is not None:
            st.session_state.iteration_df.to_excel(writer, sheet_name="Iterations", index=False)
        if st.session_state.explanation_df is not None:
            st.session_state.explanation_df.to_excel(writer, sheet_name="Explanations", index=False)
        if st.session_state.final_data is not None:
            st.session_state.final_data.to_excel(writer, sheet_name="Final Data", index=False)
        if st.session_state.classification_report_df is not None:
            st.session_state.classification_report_df.to_excel(writer, sheet_name="Classification Report", index=False)
        if st.session_state.confusion_matrix_df is not None:
            st.session_state.confusion_matrix_df.to_excel(writer, sheet_name="Confusion Matrix", index=True)
    output.seek(0)
    return output


def create_zip_outputs():
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        if st.session_state.results_df is not None:
            zf.writestr("model_results.csv", st.session_state.results_df.to_csv(index=False))
        if st.session_state.iteration_df is not None:
            zf.writestr("iteration_results.csv", st.session_state.iteration_df.to_csv(index=False))
        if st.session_state.explanation_df is not None:
            zf.writestr("iteration_explanations.csv", st.session_state.explanation_df.to_csv(index=False))
        if st.session_state.final_data is not None:
            zf.writestr("final_customer_risk_data.csv", st.session_state.final_data.to_csv(index=False))

        excel_file = create_excel_report()
        zf.writestr("DSS_Explainable_Report.xlsx", excel_file.getvalue())

        if st.session_state.iteration_df is not None:
            fig = make_high_risk_chart(st.session_state.iteration_df)
            chart_buffer = io.BytesIO()
            fig.savefig(chart_buffer, format="png", bbox_inches="tight")
            plt.close(fig)
            chart_buffer.seek(0)
            zf.writestr("high_risk_trend.png", chart_buffer.getvalue())

    zip_buffer.seek(0)
    return zip_buffer

# ==========================================================
# INITIALIZE SESSION
# ==========================================================
initialize_session_state()

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.title("📊 Retention DSS")
st.sidebar.caption("Machine Learning Decision Support System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Data Source",
        "Train Models",
        "Risk Prediction & Simulation",
        "Reports & Downloads",
        "About System",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This system trains ML models, classifies customer risk, runs closed-loop interventions, and exports DSS reports."
)

# ==========================================================
# HERO HEADER
# ==========================================================
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">Customer Retention Decision Support System</div>
        <div class="hero-subtitle">
            Web-based Python system for predictive analysis, risk classification, closed-loop DSS simulation, explainable outputs, charts, and downloadable reports.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# DASHBOARD PAGE
# ==========================================================
if page == "Dashboard":
    st.subheader("System Overview")

    c1, c2, c3, c4 = st.columns(4)

    total_records = 0 if st.session_state.df is None else len(st.session_state.df)
    best_model = st.session_state.best_model_name or "Not trained"

    if st.session_state.final_data is not None:
        high_risk = int((st.session_state.final_data["Risk_Level"] == "HIGH RISK").sum())
        medium_risk = int((st.session_state.final_data["Risk_Level"] == "MEDIUM RISK").sum())
        low_risk = int((st.session_state.final_data["Risk_Level"] == "LOW RISK").sum())
    else:
        high_risk = medium_risk = low_risk = 0

    c1.metric("Training Records", total_records)
    c2.metric("Best Model", best_model)
    c3.metric("High Risk", high_risk)
    c4.metric("Low Risk", low_risk)

    st.markdown("---")

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Workflow")
        st.markdown(
            """
            1. Generate synthetic customer data or upload your own customer dataset.  
            2. Train multiple machine learning models.  
            3. Select the best model using AUC score.  
            4. Classify customers into LOW, MEDIUM, and HIGH risk.  
            5. Simulate interventions across iterations.  
            6. Export CSV, Excel, chart, and ZIP reports.
            """
        )

    with right:
        st.markdown("### Current Status")
        if st.session_state.df is None:
            st.warning("No dataset loaded yet. Go to Data Source.")
        elif st.session_state.best_model is None:
            st.info("Dataset is ready. Go to Train Models.")
        elif st.session_state.final_data is None:
            st.info("Model is trained. Run prediction and simulation.")
        else:
            st.success("System outputs are ready for download.")

    if st.session_state.results_df is not None:
        st.markdown("### Latest Model Performance")
        st.dataframe(st.session_state.results_df, use_container_width=True)

    if st.session_state.iteration_df is not None:
        st.markdown("### High Risk Trend")
        fig = make_high_risk_chart(st.session_state.iteration_df)
        st.pyplot(fig)
        plt.close(fig)

# ==========================================================
# DATA SOURCE PAGE
# ==========================================================
elif page == "Data Source":
    st.subheader("Data Source")

    st.markdown(
        """
        This page allows you to either generate synthetic customer data or upload your own CSV file.
        Your uploaded dataset must contain all required feature columns and a target column named `CLASS`.
        """
    )

    data_option = st.radio(
        "Choose data source",
        ["Generate synthetic data", "Upload CSV dataset"],
        horizontal=True,
    )

    if data_option == "Generate synthetic data":
        col1, col2, col3 = st.columns(3)
        with col1:
            n_rows = st.number_input("Number of rows", min_value=100, max_value=100000, value=2000, step=100)
        with col2:
            seed_input = st.number_input("Seed", min_value=1, max_value=999999, value=int(time.time()) % 100000)
        with col3:
            pattern_choice = st.selectbox("Data pattern", ["Random", "linear", "nonlinear", "tree", "noisy"])

        if st.button("Generate Dataset", type="primary"):
            df, seed_used, pattern_type = generate_synthetic_data(
                n_rows=int(n_rows),
                seed=int(seed_input),
                pattern_type=pattern_choice,
            )
            st.session_state.df = df
            st.session_state.seed_used = seed_used
            st.session_state.pattern_type = pattern_type
            st.success(f"Dataset generated successfully. Seed: {seed_used}. Pattern: {pattern_type}.")

    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                prepared_df, valid, message = prepare_uploaded_data(raw_df)

                if valid:
                    st.session_state.df = prepared_df

                    # Clear previous training and simulation outputs after loading new data
                    st.session_state.results_df = None
                    st.session_state.best_model_name = None
                    st.session_state.best_model = None
                    st.session_state.scaler = None
                    st.session_state.iteration_df = None
                    st.session_state.explanation_df = None
                    st.session_state.final_data = None
                    st.session_state.classification_report_df = None
                    st.session_state.confusion_matrix_df = None
                    st.session_state.risk_summary_df = None

                    st.success(message)
                    st.info("Your uploaded file has been prepared for training. Open the Train Models page and click Train Models.")
                else:
                    st.error(message)
                    st.markdown("### Columns found in your uploaded file")
                    st.write(list(raw_df.columns))

            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")

    if st.session_state.df is not None:
        st.markdown("### Dataset Preview")
        st.dataframe(st.session_state.df.head(20), use_container_width=True)

        st.markdown("### Class Distribution")
        class_dist = st.session_state.df["CLASS"].value_counts(normalize=True).reset_index()
        class_dist.columns = ["Class", "Proportion"]
        st.dataframe(class_dist, use_container_width=True)

        st.download_button(
            "Download Current Dataset CSV",
            data=dataframe_to_csv_bytes(st.session_state.df),
            file_name="customer_training_dataset.csv",
            mime="text/csv",
        )

# ==========================================================
# TRAIN MODELS PAGE
# ==========================================================
elif page == "Train Models":
    st.subheader("Train and Evaluate Machine Learning Models")

    if st.session_state.df is None:
        st.warning("Please load or generate data first from the Data Source page.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
        with col2:
            train_seed = st.number_input("Training seed", min_value=1, max_value=999999, value=42)

        if st.button("Train Models", type="primary"):
            with st.spinner("Training models and evaluating performance..."):
                (
                    results_df,
                    best_model_name,
                    best_model,
                    scaler,
                    report_df,
                    cm_df,
                ) = train_and_evaluate_models(
                    st.session_state.df,
                    test_size=float(test_size),
                    seed=int(train_seed),
                )

                st.session_state.results_df = results_df
                st.session_state.best_model_name = best_model_name
                st.session_state.best_model = best_model
                st.session_state.scaler = scaler
                st.session_state.classification_report_df = report_df
                st.session_state.confusion_matrix_df = cm_df

            st.success(f"Training completed. Best model: {best_model_name}")

        if st.session_state.results_df is not None:
            st.markdown("### Model Performance")
            st.dataframe(st.session_state.results_df, use_container_width=True)

            fig = make_model_performance_chart(st.session_state.results_df)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### Best Model")
            st.success(st.session_state.best_model_name)

            st.markdown("### Classification Report")
            st.dataframe(st.session_state.classification_report_df, use_container_width=True)

            st.markdown("### Confusion Matrix")
            st.dataframe(st.session_state.confusion_matrix_df, use_container_width=True)

# ==========================================================
# RISK PREDICTION AND SIMULATION PAGE
# ==========================================================
elif page == "Risk Prediction & Simulation":
    st.subheader("Risk Prediction and Closed-Loop DSS Simulation")

    if st.session_state.best_model is None or st.session_state.scaler is None:
        st.warning("Please train the models first from the Train Models page.")
    else:
        st.markdown(
            """
            This section creates new customer records, predicts customer risk, applies simulated interventions,
            and recalculates risk across several iterations.
            """
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            n_customers = st.number_input("New customers", min_value=10, max_value=100000, value=500, step=10)
        with col2:
            iterations = st.number_input("Iterations", min_value=1, max_value=100, value=10, step=1)
        with col3:
            simulation_seed = st.number_input("Simulation seed", min_value=1, max_value=999999, value=123)

        if st.button("Run DSS Simulation", type="primary"):
            with st.spinner("Running risk prediction and closed-loop DSS simulation..."):
                iteration_df, explanation_df, final_data, risk_summary_df = run_closed_loop_simulation(
                    model=st.session_state.best_model,
                    scaler=st.session_state.scaler,
                    n_customers=int(n_customers),
                    iterations=int(iterations),
                    seed=int(simulation_seed),
                )

                st.session_state.iteration_df = iteration_df
                st.session_state.explanation_df = explanation_df
                st.session_state.final_data = final_data
                st.session_state.risk_summary_df = risk_summary_df

            st.success("Simulation completed successfully.")

        if st.session_state.iteration_df is not None:
            st.markdown("### Iteration Results")
            st.dataframe(st.session_state.iteration_df, use_container_width=True)

            fig = make_high_risk_chart(st.session_state.iteration_df)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### Iteration Explanations")
            st.dataframe(st.session_state.explanation_df, use_container_width=True)

            st.markdown("### Final Risk Summary")
            st.dataframe(st.session_state.risk_summary_df, use_container_width=True)

            fig = make_risk_distribution_chart(st.session_state.risk_summary_df)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### Final Customer Risk Data")
            st.dataframe(st.session_state.final_data.head(100), use_container_width=True)

# ==========================================================
# REPORTS AND DOWNLOADS PAGE
# ==========================================================
elif page == "Reports & Downloads":
    st.subheader("Reports and Downloadable Outputs")

    if st.session_state.results_df is None:
        st.warning("No outputs available yet. Train models and run simulation first.")
    else:
        st.markdown("### Available Outputs")

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "Download Model Results CSV",
                data=dataframe_to_csv_bytes(st.session_state.results_df),
                file_name="model_results.csv",
                mime="text/csv",
            )

            if st.session_state.iteration_df is not None:
                st.download_button(
                    "Download Iteration Results CSV",
                    data=dataframe_to_csv_bytes(st.session_state.iteration_df),
                    file_name="iteration_results.csv",
                    mime="text/csv",
                )

            if st.session_state.explanation_df is not None:
                st.download_button(
                    "Download Iteration Explanations CSV",
                    data=dataframe_to_csv_bytes(st.session_state.explanation_df),
                    file_name="iteration_explanations.csv",
                    mime="text/csv",
                )

        with col2:
            if st.session_state.final_data is not None:
                st.download_button(
                    "Download Final Customer Risk Data CSV",
                    data=dataframe_to_csv_bytes(st.session_state.final_data),
                    file_name="final_customer_risk_data.csv",
                    mime="text/csv",
                )

            excel_output = create_excel_report()
            st.download_button(
                "Download Excel Report",
                data=excel_output,
                file_name="DSS_Explainable_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            zip_output = create_zip_outputs()
            st.download_button(
                "Download All Outputs ZIP",
                data=zip_output,
                file_name="DSS_All_Outputs.zip",
                mime="application/zip",
            )

        if st.session_state.iteration_df is not None:
            st.markdown("### Chart Preview")
            fig = make_high_risk_chart(st.session_state.iteration_df)
            st.pyplot(fig)
            plt.close(fig)

# ==========================================================
# ABOUT SYSTEM PAGE
# ==========================================================
elif page == "About System":
    st.subheader("About the System")

    st.markdown(
        """
        ### Purpose
        This web-based Python system supports customer retention decision-making using machine learning and a closed-loop Decision Support System workflow.

        ### Main Functionalities
        - Generate synthetic telecom customer data.
        - Upload external CSV customer data.
        - Train multiple machine learning models.
        - Compare models using Accuracy, Precision, Recall, F1 Score, and AUC.
        - Select the best model automatically.
        - Predict customer risk probability.
        - Classify customers as LOW RISK, MEDIUM RISK, or HIGH RISK.
        - Simulate retention interventions across several iterations.
        - Explain how customer risk changes after each intervention.
        - Display tables and charts.
        - Export CSV, Excel, PNG chart, and ZIP reports.

        ### Required Dataset Columns
        Your CSV file must include the following columns:
        """
    )

    st.code(", ".join(FEATURE_COLUMNS) + ", CLASS")

    st.markdown(
        """
        ### Interpretation
        The system follows this DSS flow:

        **Inputs → Predictive Model → DSS Interpretation → Intervention → Updated Risk**

        This means that customer data is first processed by a predictive model. The system then identifies customer risk levels,
        applies simulated intervention decisions, recalculates risk, and generates reports that can support managerial decisions.
        """
    )
