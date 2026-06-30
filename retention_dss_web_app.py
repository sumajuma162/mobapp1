 
# ==========================================================
# CUSTOMER RETENTION DECISION SUPPORT SYSTEM (WEB-BASED)
# Streamlit + Machine Learning + Closed-Loop DSS Simulation
# ==========================================================

import io
import time
import zipfile
import warnings
from datetime import datetime
from itertools import combinations
import importlib.util

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
    page_icon="",
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

CUSTOMER_ID_CANDIDATES = [
    "CUSTOMER_ID",
    "CUSTOMER_NAME",
    "CUSTOMER",
    "CUSTOMER_NO",
    "CLIENT_ID",
    "ACCOUNT_NUMBER",
    "PHONE_NUMBER",
    "MSISDN",
]

# Risk factor configuration used for explainable high-risk reporting.
# risk_when="high" means high values increase risk. risk_when="low" means low values increase risk.
RISK_FACTOR_CONFIG = {
    "HIGH_SWITCHING_ENERGY_TIME": {
        "label": "High switching energy time",
        "risk_when": "high",
        "threshold": 0.5,
        "recommendation": (
            "Retain this customer by reducing the effort and time required to stay satisfied: "
            "assign fast support, simplify service processes, provide guided assistance, and show clear benefits of remaining."
        ),
    },
    "HIGH_SWITCHING_COST": {
        "label": "High switching cost",
        "risk_when": "high",
        "threshold": 0.5,
        "recommendation": (
            "Use a loyalty-based retention offer such as renewal rewards, contract benefits, bonus services, "
            "or personalized incentives that make staying more valuable than leaving."
        ),
    },
    "VALUE_FOR_MONEY": {
        "label": "Value for money concern",
        "risk_when": "low",
        "threshold": 0.5,
        "recommendation": (
            "Improve perceived value for money using a targeted discount, better bundle, bonus data/minutes, "
            "or a package that matches the customer's usage pattern."
        ),
    },
    "WIDE_COVERAGE": {
        "label": "Coverage concern",
        "risk_when": "low",
        "threshold": 0.5,
        "recommendation": (
            "Retain the customer by addressing coverage concerns: verify network availability in their area, "
            "escalate weak coverage zones, and offer a suitable coverage-support solution."
        ),
    },
    "CALL_DROPS": {
        "label": "Call drops",
        "risk_when": "high",
        "threshold": 0.5,
        "recommendation": (
            "Prioritize technical network support, investigate dropped-call locations, escalate quality issues, "
            "and follow up after resolution."
        ),
    },
    "STRONG_SIGNALS": {
        "label": "Weak signal concern",
        "risk_when": "low",
        "threshold": 0.5,
        "recommendation": (
            "Retain the customer by improving signal experience: troubleshoot device/network settings, "
            "recommend stronger coverage options, and escalate persistent weak-signal areas."
        ),
    },
    "PROBLEM_SOLVING": {
        "label": "Poor problem solving",
        "risk_when": "low",
        "threshold": 0.5,
        "recommendation": (
            "Assign a dedicated support agent, resolve the customer's main problem quickly, and confirm satisfaction "
            "after the issue is closed."
        ),
    },
    "EXCEPTIONAL_SERVICE_EXPERIENCE": {
        "label": "Poor service experience",
        "risk_when": "low",
        "threshold": 0.5,
        "recommendation": (
            "Improve customer experience through priority handling, apology/recovery action, and personalized follow-up "
            "to rebuild trust."
        ),
    },
    "GET_THROUGH": {
        "label": "Difficulty getting through",
        "risk_when": "low",
        "threshold": 0.5,
        "recommendation": (
            "Provide easier access to support channels, call-back service, WhatsApp/chat support, or a priority contact route."
        ),
    },
    "DO_WHAT_THEY_SAY": {
        "label": "Promise fulfilment concern",
        "risk_when": "low",
        "threshold": 0.5,
        "recommendation": (
            "Retain the customer by keeping service promises: document commitments, give realistic resolution times, "
            "and follow up until the promise is fulfilled."
        ),
    },
    "TIMELY_EFFECTIVE_COMPLAINTS": {
        "label": "Slow or ineffective complaints handling",
        "risk_when": "low",
        "threshold": 0.5,
        "recommendation": (
            "Escalate complaint resolution, shorten turnaround time, provide status updates, and close the loop with the customer."
        ),
    },
    "FREE_COMPLAINTS": {
        "label": "Free complaints concern",
        "risk_when": "low",
        "threshold": 0.5,
        "recommendation": (
            "Retain the customer by making complaint channels easy and free to use, removing complaint barriers, "
            "and giving a clear complaint-tracking path."
        ),
    },
    "TENURE_CLASS": {
        "label": "Low tenure concern",
        "risk_when": "low",
        "threshold": 0.5,
        "recommendation": (
            "Use onboarding retention actions: welcome support, early-life check-ins, education on benefits, "
            "and first-month loyalty incentives."
        ),
    },
    "EARNINGS_CLASS": {
        "label": "Price sensitivity concern",
        "risk_when": "low",
        "threshold": 0.5,
        "recommendation": (
            "Offer an affordable plan, flexible package, or usage-based bundle that matches the customer's financial capacity."
        ),
    },
    "LOAN_BOARD_CLASS": {
        "label": "Loan board class concern",
        "risk_when": "high",
        "threshold": 0.5,
        "recommendation": (
            "Use targeted student/customer-segment support, affordable packages, and payment-friendly retention options."
        ),
    },
}

HIGH_RISK_REPORT_COLUMNS = [
    "Rank",
    "Customer_ID",
    "Risk_Probability",
    "Risk_Category",
    "High_Risk_Causes",
    "Most_Important_Factor",
    "Retention_Recommendation",
    "ARM_Rule",
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
        "high_risk_customer_report_df": None,
        "arm_rules_df": None,
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
    df.insert(0, "CUSTOMER_ID", [f"CUST-{i + 1:05d}" for i in range(n_rows)])

    return df, seed, pattern_type


def prepare_uploaded_data(df):
    """
    Prepare uploaded CSV data for the DSS system.

    This version solves common upload problems:
    1. Cleans column names by removing spaces.
    2. Converts column names to uppercase.
    3. Accepts common target names such as CHURN, TARGET, STATUS, and RETENTION_STATUS.
    4. Automatically creates missing feature columns with default value 0.
    5. Preserves or generates CUSTOMER_ID so high-risk customers can appear clearly in reports.
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

    # Preserve or generate a clear customer identifier for high-risk reporting
    if "CUSTOMER_ID" not in prepared_df.columns:
        source_identifier = None
        for col in CUSTOMER_ID_CANDIDATES:
            if col in prepared_df.columns and col != "CUSTOMER_ID":
                source_identifier = col
                break

        if source_identifier is not None:
            prepared_df["CUSTOMER_ID"] = prepared_df[source_identifier].astype(str)
        else:
            prepared_df["CUSTOMER_ID"] = [f"CUST-{i + 1:05d}" for i in range(len(prepared_df))]

    missing_features = []

    # Add missing feature columns automatically
    for col in FEATURE_COLUMNS:
        if col not in prepared_df.columns:
            prepared_df[col] = 0
            missing_features.append(col)

    # Keep customer identifier, required features, and target in correct order
    prepared_df = prepared_df[["CUSTOMER_ID"] + FEATURE_COLUMNS + ["CLASS"]]

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


def ensure_customer_identifier(df):
    working_df = df.copy()

    if "CUSTOMER_ID" not in working_df.columns:
        working_df.insert(0, "CUSTOMER_ID", [f"CUST-{i + 1:05d}" for i in range(len(working_df))])
    else:
        working_df["CUSTOMER_ID"] = working_df["CUSTOMER_ID"].astype(str)

    return working_df


def recalc_risk(df, model, scaler):
    working_df = ensure_customer_identifier(df)
    X_scaled = scaler.transform(working_df[FEATURE_COLUMNS])
    working_df["Risk_Prob"] = model.predict_proba(X_scaled)[:, 1]
    working_df["Prediction"] = model.predict(X_scaled)
    working_df["Risk_Level"] = working_df["Risk_Prob"].apply(risk_level)
    return working_df


def get_model_feature_importance(model):
    if hasattr(model, "feature_importances_"):
        importance = np.array(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.array(model.coef_).reshape(-1))
    else:
        importance = np.ones(len(FEATURE_COLUMNS), dtype=float)

    if importance.size != len(FEATURE_COLUMNS):
        importance = np.ones(len(FEATURE_COLUMNS), dtype=float)

    if np.nanmax(importance) > 0:
        importance = importance / np.nanmax(importance)

    return pd.Series(importance, index=FEATURE_COLUMNS).fillna(0)


def is_feature_risk_active(feature, value):
    config = RISK_FACTOR_CONFIG.get(feature)
    if config is None:
        return False

    threshold = float(config.get("threshold", 0.5))
    risk_when = config.get("risk_when", "high")

    try:
        numeric_value = float(value)
    except Exception:
        numeric_value = 0.0

    if risk_when == "high":
        return numeric_value >= threshold

    return numeric_value <= threshold


def get_customer_risk_causes(row, feature_importance):
    causes = []

    for feature, config in RISK_FACTOR_CONFIG.items():
        if feature in row.index and is_feature_risk_active(feature, row[feature]):
            causes.append(
                {
                    "feature": feature,
                    "label": config["label"],
                    "importance": float(feature_importance.get(feature, 0)),
                }
            )

    causes = sorted(causes, key=lambda item: item["importance"], reverse=True)
    return causes


def build_risk_transaction_matrix(df):
    transaction_matrix = pd.DataFrame(index=df.index)

    for feature, config in RISK_FACTOR_CONFIG.items():
        label = config["label"]

        if feature not in df.columns:
            transaction_matrix[label] = False
            continue

        threshold = float(config.get("threshold", 0.5))
        values = pd.to_numeric(df[feature], errors="coerce").fillna(0)

        if config.get("risk_when", "high") == "high":
            transaction_matrix[label] = values >= threshold
        else:
            transaction_matrix[label] = values <= threshold

    return transaction_matrix


def generate_arm_rules(df, min_support=0.02, min_confidence=0.50, max_antecedents=3):
    """
    Generate simple Association Rule Mining outputs without requiring extra libraries.
    Rule format: IF risk factor(s) THEN HIGH RISK.
    """
    if df is None or df.empty or "Risk_Level" not in df.columns:
        return pd.DataFrame(columns=["Antecedents", "Consequent", "Support", "Confidence", "Lift", "ARM_Rule"])

    working_df = df.copy()
    total_records = len(working_df)

    if total_records == 0:
        return pd.DataFrame(columns=["Antecedents", "Consequent", "Support", "Confidence", "Lift", "ARM_Rule"])

    transaction_matrix = build_risk_transaction_matrix(working_df)
    labels = list(transaction_matrix.columns)
    high_risk_mask = working_df["Risk_Level"].eq("HIGH RISK")
    high_risk_rate = high_risk_mask.mean()

    if high_risk_rate == 0:
        return pd.DataFrame(columns=["Antecedents", "Consequent", "Support", "Confidence", "Lift", "ARM_Rule"])

    min_count = max(2, int(np.ceil(total_records * min_support)))
    rules = []

    for antecedent_size in range(1, max_antecedents + 1):
        for antecedents in combinations(labels, antecedent_size):
            antecedent_mask = transaction_matrix[list(antecedents)].all(axis=1)
            antecedent_count = int(antecedent_mask.sum())

            if antecedent_count < min_count:
                continue

            rule_count = int((antecedent_mask & high_risk_mask).sum())
            if rule_count == 0:
                continue

            support = rule_count / total_records
            confidence = rule_count / antecedent_count
            lift = confidence / high_risk_rate if high_risk_rate > 0 else 0

            if support < min_support or confidence < min_confidence:
                continue

            antecedent_text = " + ".join(antecedents)
            arm_rule = (
                f"IF {antecedent_text} THEN HIGH RISK "
                f"(support={support:.2f}, confidence={confidence:.2f}, lift={lift:.2f})"
            )

            rules.append(
                {
                    "Antecedents_List": list(antecedents),
                    "Antecedents": antecedent_text,
                    "Consequent": "HIGH RISK",
                    "Support": round(support, 4),
                    "Confidence": round(confidence, 4),
                    "Lift": round(lift, 4),
                    "ARM_Rule": arm_rule,
                }
            )

    if not rules:
        return pd.DataFrame(columns=["Antecedents", "Consequent", "Support", "Confidence", "Lift", "ARM_Rule"])

    rules_df = pd.DataFrame(rules)
    rules_df = rules_df.sort_values(
        by=["Confidence", "Lift", "Support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return rules_df.head(100)


def get_matching_arm_rule(cause_labels, main_factor_label, arm_rules_df):
    if arm_rules_df is None or arm_rules_df.empty:
        return f"IF {main_factor_label} THEN HIGH RISK"

    cause_set = set(cause_labels)

    # Prefer rules that include the most important factor and are fully supported by this customer's causes.
    for _, rule in arm_rules_df.iterrows():
        antecedents = rule.get("Antecedents_List", [])
        if isinstance(antecedents, str):
            antecedents = [item.strip() for item in antecedents.split("+")]

        antecedent_set = set(antecedents)
        if antecedent_set.issubset(cause_set) and main_factor_label in antecedent_set:
            return rule["ARM_Rule"]

    # Otherwise use the strongest matching rule supported by this customer's causes.
    for _, rule in arm_rules_df.iterrows():
        antecedents = rule.get("Antecedents_List", [])
        if isinstance(antecedents, str):
            antecedents = [item.strip() for item in antecedents.split("+")]

        if set(antecedents).issubset(cause_set):
            return rule["ARM_Rule"]

    return f"IF {main_factor_label} THEN HIGH RISK"


def build_high_risk_customer_report(final_data, model):
    """
    Build the ordered high-risk customer report requested by the user:
    customer list, causes, most important factor, retention recommendation, and ARM rule in the last column.
    """
    if final_data is None or final_data.empty:
        empty_report = pd.DataFrame(columns=HIGH_RISK_REPORT_COLUMNS)
        empty_rules = pd.DataFrame(columns=["Antecedents", "Consequent", "Support", "Confidence", "Lift", "ARM_Rule"])
        return empty_report, empty_rules

    working_df = ensure_customer_identifier(final_data)
    arm_rules_df = generate_arm_rules(working_df)
    feature_importance = get_model_feature_importance(model)

    high_risk_df = (
        working_df[working_df["Risk_Level"].eq("HIGH RISK")]
        .copy()
        .sort_values(by="Risk_Prob", ascending=False)
        .reset_index(drop=True)
    )

    rows = []

    for index, row in high_risk_df.iterrows():
        causes = get_customer_risk_causes(row, feature_importance)

        if causes:
            cause_labels = [cause["label"] for cause in causes]
            most_important = causes[0]
            most_important_label = most_important["label"]
            recommendation = RISK_FACTOR_CONFIG[most_important["feature"]]["recommendation"]
        else:
            cause_labels = ["Combined model signals"]
            most_important_label = "Combined model signals"
            recommendation = (
                "Use a personalized retention call, review the customer profile manually, "
                "identify the strongest dissatisfaction area, and offer a targeted retention package."
            )

        arm_rule = get_matching_arm_rule(cause_labels, most_important_label, arm_rules_df)

        rows.append(
            {
                "Rank": index + 1,
                "Customer_ID": row["CUSTOMER_ID"],
                "Risk_Probability": round(float(row["Risk_Prob"]), 4),
                "Risk_Category": row["Risk_Level"],
                "High_Risk_Causes": "; ".join(cause_labels),
                "Most_Important_Factor": most_important_label,
                "Retention_Recommendation": recommendation,
                "ARM_Rule": arm_rule,
            }
        )

    report_df = pd.DataFrame(rows, columns=HIGH_RISK_REPORT_COLUMNS)

    if not arm_rules_df.empty and "Antecedents_List" in arm_rules_df.columns:
        export_rules_df = arm_rules_df.drop(columns=["Antecedents_List"])
    else:
        export_rules_df = arm_rules_df.copy()

    return report_df, export_rules_df


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
    new_data.insert(0, "CUSTOMER_ID", [f"CUST-{i + 1:05d}" for i in range(n_customers)])

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

        editable_features.insert(0, "CUSTOMER_ID", current_data["CUSTOMER_ID"].values)
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


def get_available_excel_engine():
    """
    Return an installed Excel writer engine.
    Streamlit Cloud will crash if the selected engine is not listed in requirements.txt,
    so this function checks availability before Pandas tries to import the engine.
    """
    if importlib.util.find_spec("xlsxwriter") is not None:
        return "xlsxwriter"
    if importlib.util.find_spec("openpyxl") is not None:
        return "openpyxl"
    return None


def create_excel_report():
    excel_engine = get_available_excel_engine()

    # If no Excel writer package is installed, return None instead of crashing the whole app.
    if excel_engine is None:
        return None

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine=excel_engine) as writer:
        if st.session_state.results_df is not None:
            st.session_state.results_df.to_excel(writer, sheet_name="Model Performance", index=False)
        if st.session_state.iteration_df is not None:
            st.session_state.iteration_df.to_excel(writer, sheet_name="Iterations", index=False)
        if st.session_state.explanation_df is not None:
            st.session_state.explanation_df.to_excel(writer, sheet_name="Explanations", index=False)
        if st.session_state.final_data is not None:
            st.session_state.final_data.to_excel(writer, sheet_name="Final Data", index=False)
        if st.session_state.high_risk_customer_report_df is not None:
            st.session_state.high_risk_customer_report_df.to_excel(writer, sheet_name="High Risk Customers", index=False)
        if st.session_state.arm_rules_df is not None:
            st.session_state.arm_rules_df.to_excel(writer, sheet_name="ARM Rules", index=False)
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
        if st.session_state.high_risk_customer_report_df is not None:
            zf.writestr("high_risk_customer_retention_report.csv", st.session_state.high_risk_customer_report_df.to_csv(index=False))
        if st.session_state.arm_rules_df is not None:
            zf.writestr("association_rule_mining_rules.csv", st.session_state.arm_rules_df.to_csv(index=False))

        excel_file = create_excel_report()
        if excel_file is not None:
            zf.writestr("DSS_Explainable_Report.xlsx", excel_file.getvalue())
        else:
            zf.writestr(
                "EXCEL_EXPORT_NOT_CREATED.txt",
                "Excel export was not created because neither xlsxwriter nor openpyxl is installed. "
                "Add xlsxwriter or openpyxl to requirements.txt, then redeploy the app."
            )

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
                    st.session_state.high_risk_customer_report_df = None
                    st.session_state.arm_rules_df = None

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

                high_risk_report_df, arm_rules_df = build_high_risk_customer_report(
                    final_data,
                    st.session_state.best_model,
                )

                st.session_state.iteration_df = iteration_df
                st.session_state.explanation_df = explanation_df
                st.session_state.final_data = final_data
                st.session_state.risk_summary_df = risk_summary_df
                st.session_state.high_risk_customer_report_df = high_risk_report_df
                st.session_state.arm_rules_df = arm_rules_df

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

            st.markdown("### Ordered High-Risk Customer Retention Report")
            st.caption(
                "Customers are ordered by highest risk probability. Recommendation is based on the most important factor. "
                "The ARM rule is placed in the last column."
            )

            if st.session_state.high_risk_customer_report_df is not None and not st.session_state.high_risk_customer_report_df.empty:
                st.dataframe(st.session_state.high_risk_customer_report_df, use_container_width=True)
            else:
                st.info("No HIGH RISK customers were found in the final simulation output.")

            if st.session_state.arm_rules_df is not None and not st.session_state.arm_rules_df.empty:
                st.markdown("### Association Rule Mining Rules")
                st.dataframe(st.session_state.arm_rules_df, use_container_width=True)

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

            if st.session_state.high_risk_customer_report_df is not None:
                st.download_button(
                    "Download High-Risk Customer Retention Report CSV",
                    data=dataframe_to_csv_bytes(st.session_state.high_risk_customer_report_df),
                    file_name="high_risk_customer_retention_report.csv",
                    mime="text/csv",
                )

            if st.session_state.arm_rules_df is not None:
                st.download_button(
                    "Download ARM Rules CSV",
                    data=dataframe_to_csv_bytes(st.session_state.arm_rules_df),
                    file_name="association_rule_mining_rules.csv",
                    mime="text/csv",
                )

            excel_output = create_excel_report()
            if excel_output is not None:
                st.download_button(
                    "Download Excel Report",
                    data=excel_output,
                    file_name="DSS_Explainable_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.warning(
                    "Excel export is unavailable because neither xlsxwriter nor openpyxl is installed. "
                    "Add xlsxwriter or openpyxl to requirements.txt, then redeploy the app. "
                    "CSV and ZIP outputs are still available."
                )

            zip_output = create_zip_outputs()
            st.download_button(
                "Download All Outputs ZIP",
                data=zip_output,
                file_name="DSS_All_Outputs.zip",
                mime="application/zip",
            )

        if st.session_state.high_risk_customer_report_df is not None:
            st.markdown("### High-Risk Customer Retention Report Preview")
            if st.session_state.high_risk_customer_report_df.empty:
                st.info("No HIGH RISK customers were found in the final simulation output.")
            else:
                st.dataframe(st.session_state.high_risk_customer_report_df, use_container_width=True)

        if st.session_state.arm_rules_df is not None and not st.session_state.arm_rules_df.empty:
            st.markdown("### ARM Rules Preview")
            st.dataframe(st.session_state.arm_rules_df, use_container_width=True)

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
        - Produce an ordered high-risk customer retention report.
        - Show risk causes, most important factor, factor-based retention recommendation, and ARM rule.
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
