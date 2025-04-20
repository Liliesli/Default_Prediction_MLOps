import warnings
from sklearn.metrics import f1_score, roc_auc_score
import joblib
import json
from pathlib import Path

from .__models_registry import Models, models_registry
from .preprocess import prepare_dataset
from .model import get_xgb_model, get_logistic_model

from mlops_config import RAW_DATA_DIR, MODEL_OUTPUT_DIR

warnings.filterwarnings("ignore")

def train_xgboost(X_train, X_val, y_train, y_val):
    model = get_xgb_model()
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    model.save_model(MODEL_OUTPUT_DIR / models_registry[Models.XGBOOST].filename)
    return model

def train_logistic(X_train, X_val, y_train, y_val):
    model = get_logistic_model()
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_OUTPUT_DIR / models_registry[Models.LOGISTIC].filename)
    return model

def save_metrics(metrics, model_type):
    metrics_file = MODEL_OUTPUT_DIR / "model_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
    
    all_metrics[model_type] = metrics
    
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    
    print("F1 Score:", f1)
    print("AUC Score:", auc)
    
    return {
        "f1_score": f1,
        "auc_score": auc
    }

def get_best_model_type():
    metrics_file = MODEL_OUTPUT_DIR / "model_metrics.json"
    if not metrics_file.exists():
        return "xgboost"  # 기본값
        
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # AUC 점수를 기준으로 최적의 모델 선택
    best_model = max(metrics.items(), key=lambda x: x[1]['auc_score'])
    return best_model[0]

def train(model_type="xgboost"):
    X_train, X_val, y_train, y_val, encoder = prepare_dataset(
        path=RAW_DATA_DIR / "train.csv",
        drop_columns=["UID"],
        test_size=0.2,
        random_state=42,
    )
    
    # Save encoder
    joblib.dump(encoder, MODEL_OUTPUT_DIR / models_registry[Models.ENCODER].filename)
    
    # Train selected model
    if model_type == "xgboost":
        model = train_xgboost(X_train, X_val, y_train, y_val)
    elif model_type == "logistic":
        model = train_logistic(X_train, X_val, y_train, y_val)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Evaluate the model and save metrics
    print(f"\nEvaluating {model_type} model:")
    metrics = evaluate_model(model, X_val, y_val)
    save_metrics(metrics, model_type)
    
    # Save best model type
    best_model_type = get_best_model_type()
    with open(MODEL_OUTPUT_DIR / "best_model.txt", 'w') as f:
        f.write(best_model_type)

if __name__ == "__main__":
    train("xgboost")
    train("logistic") 