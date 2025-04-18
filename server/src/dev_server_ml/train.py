import warnings
from sklearn.metrics import f1_score, roc_auc_score
import joblib

from .__models_registry import Models, models_registry
from .preprocess import prepare_dataset
from .model import get_xgb_model

from mlops_config import RAW_DATA_DIR, MODEL_OUTPUT_DIR

warnings.filterwarnings("ignore")

def train():
    X_train, X_val, y_train, y_val, encoder = prepare_dataset(
        path= RAW_DATA_DIR / "train.csv",
        drop_columns=["UID"],
        test_size=0.2,
        random_state=42,
    )
    model = get_xgb_model()

    # train XGBoost model 
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # save the model
    model.save_model(MODEL_OUTPUT_DIR / models_registry[Models.XGBOOST].filename)
    joblib.dump(encoder, MODEL_OUTPUT_DIR / models_registry[Models.ENCODER].filename)
    
    # evaluate the model
    y_pred = model.predict(X_val)
    print("F1 Score:", f1_score(y_val, y_pred))
    print("AUC Score:", roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))

if __name__ == "__main__":
    train()