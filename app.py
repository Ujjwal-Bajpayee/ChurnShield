import streamlit as st
import pandas as pd

from src.prediction import ChurnPredictor
from src.explainability import ChurnExplainer

st.set_page_config(
    page_title="ChurnShield",
    layout="wide"
)

@st.cache_resource
def load_predictor():
    return ChurnPredictor()

@st.cache_resource
def load_explainer():
    return ChurnExplainer()

FEATURE_LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior Citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "tenure": "Tenure",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines",
    "InternetService": "Internet Service",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "Contract": "Contract Type",
    "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Method",
    "MonthlyCharges": "Monthly Charges",
    "TotalCharges": "Total Charges",
    "AvgMonthlySpend": "Average Monthly Spend",
    "ChargeRatio": "Monthly to Avg Spend Ratio",
    "ServiceCount": "Subscribed Services Count",
    "HasStreaming": "Streaming Services Active",
    "IsShortTenureMonthToMonth": "New Month-to-Month Customer"
}

def format_feature_value(feature, value):
    if feature == "tenure":
        return f"{value} months"
    if feature in ["MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]:
        return f"${float(value):,.2f}"
    if feature == "ChargeRatio":
        return f"{float(value):.2f}x"
    if feature == "ServiceCount":
        return f"{int(value)} services"
    if feature in ["SeniorCitizen", "HasStreaming", "IsShortTenureMonthToMonth"]:
        return "Yes" if int(value) == 1 else "No"
    return str(value)

def format_feature_name(feature, value=None):
    label = FEATURE_LABELS.get(feature, feature)
    if value is not None:
        val_str = format_feature_value(feature, value)
        return f"{label}: {val_str}"
    return label

def get_explanation_text(feature, value, direction):
    label = FEATURE_LABELS.get(feature, feature)

    if feature == "Contract":
        base_text = f"The customer has a {value.lower()} contract."
    elif feature == "InternetService":
        if value == "No":
            base_text = "The customer does not have internet service."
        else:
            base_text = f"The customer uses {value} internet service."
    elif feature == "tenure":
        base_text = f"The customer has been with the company for {value} months."
    elif feature == "MonthlyCharges":
        base_text = f"The customer's monthly charge is ${float(value):.2f}."
    elif feature == "TotalCharges":
        base_text = f"The customer's total accumulated charges are ${float(value):.2f}."
    elif feature == "AvgMonthlySpend":
        base_text = f"The customer averages ${float(value):.2f} per month across their tenure."
    elif feature == "ChargeRatio":
        base_text = f"The ratio of current monthly bill to average tenure spend is {float(value):.2f}x."
    elif feature == "ServiceCount":
        base_text = f"The customer currently subscribes to {int(value)} add-on services."
    elif feature == "IsShortTenureMonthToMonth":
        status = "Yes" if int(value) == 1 else "No"
        base_text = f"The customer is a new subscriber on a month-to-month plan ({status})."
    elif feature == "HasStreaming":
        status = "Yes" if int(value) == 1 else "No"
        base_text = f"The customer has active streaming add-ons ({status})."
    elif feature == "PaymentMethod":
        base_text = f"The customer pays via {value.lower()}."
    elif feature == "SeniorCitizen":
        status = "Yes" if int(value) == 1 else "No"
        base_text = f"The customer is a senior citizen: {status}."
    elif feature == "gender":
        base_text = f"The customer is {value.lower()}."
    elif feature in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        if value == "Yes":
            base_text = f"The customer has {label.lower()}."
        else:
            base_text = f"The customer does not have {label.lower()}."
    elif feature in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]:
        if value == "Yes":
            base_text = f"The customer is subscribed to {label.lower()}."
        elif value == "No":
            base_text = f"The customer is not subscribed to {label.lower()}."
        else:
            base_text = f"The customer has no internet service for {label.lower()}."
    elif feature == "MultipleLines":
        if value == "Yes":
            base_text = "The customer has multiple phone lines."
        elif value == "No":
            base_text = "The customer has a single phone line."
        else:
            base_text = "The customer does not have phone service."
    else:
        base_text = f"The customer's {label} is {value}."

    if direction == "increase":
        return f"{base_text} This characteristic increased the model's estimated churn risk."
    return f"{base_text} This characteristic reduced the model's estimated churn risk."

predictor = load_predictor()
explainer = load_explainer()

st.title("🛡️ ChurnShield")
st.subheader("Customer Churn Risk Prediction & Explainability Platform")

st.write(
    """
    ChurnShield predicts customer churn risk with calibrated ML pipelines
    and provides exact, non-redundant factor attributions using SHAP.
    """
)

st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

with col2:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

with col3:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=1000.0
    )

customer = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

st.divider()

if st.button(
    "Predict Churn",
    type="primary"
):
    result = predictor.predict(customer)

    probability = result["churn_probability"]
    prediction = result["prediction"]
    risk = result["risk"]
    threshold = result["threshold"]
    model_name = result["model_name"]

    st.header("Prediction Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.1f}%"
        )

    with col2:
        st.metric(
            "Risk Category",
            risk
        )

    with col3:
        prediction_text = (
            "Likely to Churn"
            if prediction == 1
            else "Likely to Stay"
        )
        st.metric(
            "Decision",
            prediction_text
        )

    with col4:
        st.metric(
            "Optimal Threshold",
            f"{threshold * 100:.1f}%"
        )

    st.caption(f"Active Classifier: **{model_name}** | Optimized for F1 & Business Value")

    if probability >= 0.7:
        st.warning(
            f"""
            The model estimates a **{probability * 100:.1f}% probability**
            that this customer will churn. High churn risk detected.
            """
        )
    elif probability >= 0.4:
        st.info(
            f"""
            The model estimates a **{probability * 100:.1f}% probability**
            that this customer will churn. Moderate churn risk detected.
            """
        )
    else:
        st.success(
            f"""
            The model estimates a **{probability * 100:.1f}% probability**
            that this customer will churn. Customer exhibits strong retention signals.
            """
        )

    st.divider()

    st.header("Why did the model make this prediction?")

    st.write(
        """
        **SHAP (SHapley Additive exPlanations)** computes each feature's exact contribution
        relative to baseline customer profiles.
        """
    )

    explanation = explainer.explain(customer, aggregate=True)

    increasing_risk = explanation[
        explanation["shap_value"] > 0
    ].head(5)

    decreasing_risk = explanation[
        explanation["shap_value"] < 0
    ].sort_values(
        "shap_value"
    ).head(5)

    factor_col1, factor_col2 = st.columns(2)

    with factor_col1:
        st.subheader("🔺 Factors increasing churn risk")

        if increasing_risk.empty:
            st.write(
                "None of the primary factors significantly increased estimated churn risk."
            )
        else:
            for _, row in increasing_risk.iterrows():
                feat = row["feature"]
                val = row["feature_value"]
                shap_val = row["shap_value"]

                readable_feature = format_feature_name(feat, val)
                explanation_text = get_explanation_text(feat, val, "increase")

                st.markdown(f"**{readable_feature}**")
                st.caption(
                    f"{explanation_text} (SHAP contribution: +{shap_val:.4f})"
                )

    with factor_col2:
        st.subheader("🔻 Factors reducing churn risk")

        if decreasing_risk.empty:
            st.write(
                "None of the primary factors significantly reduced estimated churn risk."
            )
        else:
            for _, row in decreasing_risk.iterrows():
                feat = row["feature"]
                val = row["feature_value"]
                shap_val = row["shap_value"]

                readable_feature = format_feature_name(feat, val)
                explanation_text = get_explanation_text(feat, val, "decrease")

                st.markdown(f"**{readable_feature}**")
                st.caption(
                    f"{explanation_text} (SHAP contribution: {shap_val:.4f})"
                )

    with st.expander("📊 View detailed feature contributions (All Features)"):
        display_df = explanation.copy()

        display_df["Feature"] = display_df["feature"].apply(
            lambda f: FEATURE_LABELS.get(f, f)
        )
        display_df["Customer Value"] = display_df.apply(
            lambda r: format_feature_value(r["feature"], r["feature_value"]),
            axis=1
        )
        display_df["Impact"] = display_df["shap_value"].apply(
            lambda x: "Increases churn risk" if x > 0 else (
                "Reduces churn risk" if x < 0 else "Neutral"
            )
        )

        display_df = display_df.rename(
            columns={
                "shap_value": "SHAP Value",
                "importance": "Importance"
            }
        )

        display_df = display_df[
            [
                "Feature",
                "Customer Value",
                "SHAP Value",
                "Importance",
                "Impact"
            ]
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "SHAP Value": st.column_config.NumberColumn(
                    format="%.4f"
                ),
                "Importance": st.column_config.NumberColumn(
                    format="%.4f"
                )
            }
        )

    st.caption(
        """
        ℹ️ **SHAP Attribution Note**: Positive contributions increase predicted churn risk,
        while negative contributions push the score toward customer retention.
        """
    )