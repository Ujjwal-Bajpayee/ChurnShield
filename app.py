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


def format_feature_name(feature, customer):
    feature = feature.replace(
        "categorical__",
        ""
    ).replace(
        "numerical__",
        ""
    )

    categorical_features = {
        "gender": "Gender",
        "Partner": "Partner",
        "Dependents": "Dependents",
        "PhoneService": "Phone Service",
        "MultipleLines": "Multiple Lines",
        "InternetService": "Internet Service",
        "OnlineSecurity": "Online Security",
        "OnlineBackup": "Online Backup",
        "DeviceProtection": "Device Protection",
        "TechSupport": "Technical Support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies",
        "Contract": "Contract Type",
        "PaperlessBilling": "Paperless Billing",
        "PaymentMethod": "Payment Method"
    }

    numerical_features = {
        "tenure": "Tenure",
        "MonthlyCharges": "Monthly Charges",
        "TotalCharges": "Total Charges",
        "SeniorCitizen": "Senior Citizen"
    }

    for feature_name, readable_name in categorical_features.items():

        prefix = feature_name + "_"

        if feature.startswith(prefix):

            value = feature[len(prefix):]

            return f"{readable_name}: {value}"

    if feature in numerical_features:

        readable_name = numerical_features[feature]
        value = customer[feature]

        if feature == "tenure":
            return f"{readable_name}: {value} months"

        if feature == "MonthlyCharges":
            return f"{readable_name}: ${value:.2f}"

        if feature == "TotalCharges":
            return f"{readable_name}: ${value:.2f}"

        if feature == "SeniorCitizen":
            value = "Yes" if value == 1 else "No"
            return f"{readable_name}: {value}"

        return f"{readable_name}: {value}"

    return feature


def get_explanation_text(feature, direction, customer):
    feature_clean = (
        feature.replace("categorical__", "")
        .replace("numerical__", "")
    )

    descriptions = {
        "Contract_Month-to-month":
            "The customer is on a month-to-month contract.",
        "Contract_One year":
            "The customer is on a one-year contract.",
        "Contract_Two year":
            "The customer is on a two-year contract.",
        "InternetService_DSL":
            "The customer uses DSL internet service.",
        "InternetService_Fiber optic":
            "The customer uses fiber optic internet service.",
        "InternetService_No":
            "The customer does not have internet service.",
        "StreamingTV_Yes":
            "The customer has streaming TV service.",
        "StreamingTV_No":
            "The customer does not have streaming TV service.",
        "StreamingMovies_Yes":
            "The customer has streaming movie service.",
        "StreamingMovies_No":
            "The customer does not have streaming movie service.",
        "MultipleLines_Yes":
            "The customer has multiple phone lines.",
        "MultipleLines_No":
            "The customer does not have multiple phone lines.",
        "Partner_Yes":
            "The customer has a partner.",
        "Partner_No":
            "The customer does not have a partner.",
        "Dependents_Yes":
            "The customer has dependents.",
        "Dependents_No":
            "The customer does not have dependents.",
        "OnlineSecurity_Yes":
            "The customer has online security service.",
        "OnlineSecurity_No":
            "The customer does not have online security service.",
        "OnlineBackup_Yes":
            "The customer has online backup service.",
        "OnlineBackup_No":
            "The customer does not have online backup service.",
        "DeviceProtection_Yes":
            "The customer has device protection service.",
        "DeviceProtection_No":
            "The customer does not have device protection service.",
        "TechSupport_Yes":
            "The customer has technical support service.",
        "TechSupport_No":
            "The customer does not have technical support service.",
        "PaperlessBilling_Yes":
            "The customer uses paperless billing.",
        "PaperlessBilling_No":
            "The customer does not use paperless billing.",
        "PhoneService_Yes":
            "The customer has phone service.",
        "PhoneService_No":
            "The customer does not have phone service.",
        "gender_Male":
            "The customer is male.",
        "gender_Female":
            "The customer is female.",
        "PaymentMethod_Electronic check":
            "The customer uses electronic check for payments.",
        "PaymentMethod_Mailed check":
            "The customer uses mailed check for payments.",
        "PaymentMethod_Bank transfer (automatic)":
            "The customer uses automatic bank transfer for payments.",
        "PaymentMethod_Credit card (automatic)":
            "The customer uses automatic credit card payment."
    }

    if feature_clean in descriptions:
        base_text = descriptions[feature_clean]
    elif feature_clean == "tenure":
        base_text = (
            f"The customer has been with the company for "
            f"{customer['tenure']} months."
        )
    elif feature_clean == "MonthlyCharges":
        base_text = (
            f"The customer's monthly charge is "
            f"${customer['MonthlyCharges']:.2f}."
        )
    elif feature_clean == "TotalCharges":
        base_text = (
            f"The customer's total charges are "
            f"${customer['TotalCharges']:.2f}."
        )
    elif feature_clean == "SeniorCitizen":
        status = "Yes" if customer["SeniorCitizen"] == 1 else "No"
        base_text = f"The customer is a senior citizen: {status}."
    else:
        base_text = (
            f"The customer has the characteristic "
            f"{format_feature_name(feature, customer)}."
        )

    if direction == "increase":
        return (
            f"{base_text} This characteristic increased "
            f"the model's estimated churn risk."
        )

    return (
        f"{base_text} This characteristic reduced "
        f"the model's estimated churn risk."
    )


predictor = load_predictor()
explainer = load_explainer()


st.title("ChurnShield")
st.subheader("Customer Churn Risk Prediction")

st.write(
    """
    ChurnShield predicts whether a customer is likely to leave
    the company and explains the main factors behind the prediction.
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
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges",
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

    st.header("Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.1f}%"
        )

    with col2:
        st.metric(
            "Risk Level",
            risk
        )

    with col3:

        prediction_text = (
            "Likely to Churn"
            if prediction == 1
            else "Likely to Stay"
        )

        st.metric(
            "Prediction",
            prediction_text
        )


    st.subheader("What does this mean?")

    if probability >= 0.7:

        st.warning(
            f"""
            The model estimates a {probability * 100:.1f}% probability
            that this customer may leave the company.

            This customer shows a relatively high predicted churn risk.
            """
        )

    elif probability >= 0.4:

        st.info(
            f"""
            The model estimates a {probability * 100:.1f}% probability
            that this customer may leave the company.

            The customer shows some signs of churn risk.
            """
        )

    else:

        st.success(
            f"""
            The model estimates a {probability * 100:.1f}% probability
            that this customer may leave the company.

            Based on the information provided, the customer currently
            shows relatively low predicted churn risk.
            """
        )


    st.divider()

    st.header("Why did the model make this prediction?")

    st.write(
        """
        The model considers many characteristics of the customer.
        SHAP helps explain which characteristics had the largest
        influence on this particular prediction.

        These explanations describe how the model arrived at its
        prediction. They do not mean that a particular characteristic
        directly causes a customer to churn.
        """
    )


    explanation = explainer.explain(customer)

    top_features = explanation.head(8).copy()

    increasing_risk = top_features[
        top_features["shap_value"] > 0
    ].head(5)

    decreasing_risk = top_features[
        top_features["shap_value"] < 0
    ].sort_values(
        "shap_value"
    ).head(5)


    st.subheader("Factors increasing churn risk")

    if increasing_risk.empty:

        st.write(
            "None of the strongest contributing factors increased "
            "the model's estimated churn risk."
        )

    else:

        for _, row in increasing_risk.iterrows():

            feature = row["feature"]
            shap_value = row["shap_value"]

            readable_feature = format_feature_name(
                feature,
                customer
            )

            explanation_text = get_explanation_text(
                feature,
                "increase",
                customer
            )

            st.write(
                f"**{readable_feature}**"
            )

            st.caption(
                f"{explanation_text} "
                f"(Model contribution: {shap_value:.4f})"
            )


    st.subheader("Factors reducing churn risk")

    if decreasing_risk.empty:

        st.write(
            "None of the strongest contributing factors reduced "
            "the model's estimated churn risk."
        )

    else:

        for _, row in decreasing_risk.iterrows():

            feature = row["feature"]
            shap_value = row["shap_value"]

            readable_feature = format_feature_name(
                feature,
                customer
            )

            explanation_text = get_explanation_text(
                feature,
                "decrease",
                customer
            )

            st.write(
                f"**{readable_feature}**"
            )

            st.caption(
                f"{explanation_text} "
                f"(Model contribution: {shap_value:.4f})"
            )


    with st.expander(
        "View detailed model explanation"
    ):

        display_df = explanation[
            [
                "feature",
                "shap_value",
                "importance"
            ]
        ].copy()

        display_df["Feature"] = display_df[
            "feature"
        ].apply(
            lambda x: format_feature_name(
                x,
                customer
            )
        )

        display_df["Impact"] = display_df[
            "shap_value"
        ].apply(
            lambda x:
                "Increases churn risk"
                if x > 0
                else "Reduces churn risk"
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
        SHAP values indicate how individual features influenced the
        model's prediction. Positive values push the prediction toward
        churn, while negative values push it toward staying. SHAP
        describes model behavior and should not be interpreted as
        proof of causation.
        """
    )