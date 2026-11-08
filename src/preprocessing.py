# Libraries
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Learn thresholds from training data only
        self.credit_q75_ = X["credit_amount"].quantile(0.75)
        self.duration_q75_ = X["duration_months"].quantile(0.75)

        return self

    def transform(self, X):
        X = X.copy()

        # 1. Monthly principal repayment proxy
        X["monthly_principal_proxy"] = (
            X["credit_amount"] / X["duration_months"]
        )

        # 2. Large loan and short employment
        X["large_loan_short_employment"] = (
            (X["credit_amount"] > self.credit_q75_) &
            (X["employment_since"].isin([
                "Unemployed", "<1 year"
            ]))
        ).astype(int)

        # 3. Large loan and low savings
        X["large_loan_low_savings"] = (
            (X["credit_amount"] > self.credit_q75_) &
            (X["savings_account"] == "<100 DM")
        ).astype(int)

        # 4. Low savings and negative checking balance
        X["low_savings_negative_checking"] = (
            (X["savings_account"] == "<100 DM") &
            (X["checking_account"] == "<0 DM")
        ).astype(int)

        # 5. Loan amount per existing credit
        X["loan_per_credit"] = (
            X["credit_amount"] / X["existing_credits"]
        )

        # 6. Has other installment plan
        X["has_other_installment_plan"] = (
            X["other_installment_plans"] != "No / none"
        ).astype(int)

        # 7. Poor history and multiple credits
        poor_history = [
            "Past repayment delay",
            "Critical account / other credits elsewhere"
        ]

        X["poor_history_multiple_credits"] = (
            X["credit_history"].isin(poor_history) &
            (X["existing_credits"] > 1)
        ).astype(int)

        # 8. Long duration and large loan
        X["long_duration_large_loan"] = (
            (X["duration_months"] > self.duration_q75_) &
            (X["credit_amount"] > self.credit_q75_)
        ).astype(int)

        # Log transformations
        X["log_monthly_principal_proxy"] = np.log1p(
            X["monthly_principal_proxy"]
        )

        X["log_loan_per_credit"] = np.log1p(
            X["loan_per_credit"]
        )

        X["log1p_credit_amount"] = np.log1p(
            X["credit_amount"]
        )

        X["log1p_duration_months"] = np.log1p(
            X["duration_months"]
        )

        return X

# Numerical features
NUMERIC_FEATURES = [
    "existing_credits",
    "people_liable",
    "log1p_credit_amount",
    "log1p_duration_months",
    "log_monthly_principal_proxy",
    "log_loan_per_credit"
]

# Categorical features
CATEGORICAL_FEATURES = [
    "checking_account",
    "credit_history",
    "purpose",
    "savings_account",
    "employment_since",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
    "installment_rate",
    "residence_since"
]

# Binary engineered features
BINARY_FEATURES = [
    "large_loan_short_employment",
    "large_loan_low_savings",
    "low_savings_negative_checking",
    "has_other_installment_plan",
    "poor_history_multiple_credits",
    "long_duration_large_loan"
]

def make_preprocessor(scale_numeric=True):

    # Numerical preprocessing
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_transformer = Pipeline(numeric_steps)

    # Categorical preprocessing
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ("binary", "passthrough", BINARY_FEATURES)
    ])

    return preprocessor

def make_pipeline(classifier, scale_numeric=True):
    return Pipeline([
        ("feature_engineering", FeatureEngineer()),
        ("preprocessing", make_preprocessor(scale_numeric)),
        ("classifier", classifier)
    ])