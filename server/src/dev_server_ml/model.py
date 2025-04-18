from xgboost import XGBClassifier

def get_xgb_model():
    return XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.15,
        random_state=42,
        use_label_encoder=False,
        eval_metric="auc",
    )