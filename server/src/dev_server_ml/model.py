from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def get_xgb_model():
    return XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.15,
        random_state=42,
        use_label_encoder=False,
        eval_metric="auc",
    )

def get_logistic_model():
    return LogisticRegression(
        C=1.0,
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )