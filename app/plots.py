import plotly.express as px

def churn_by_contract(df):
    return px.histogram(
        df,
        x="Contract",
        color="Churn",
        barmode="group",
        title="Churn by Contract Type"
    )

def monthly_charges_vs_churn(df):
    return px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges vs Churn"
    )

def tenure_distribution(df):
    return px.histogram(
        df,
        x="tenure",
        color="Churn",
        nbins=30,
        barmode="overlay",
        title="Tenure Distribution by Churn"
    )

def feature_importance_chart(feat_imp):
    return px.bar(
        feat_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Factors Driving Churn"
    )
