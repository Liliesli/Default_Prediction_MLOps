import pandas as pd
import joblib
from mlops_config import RAW_DATA_DIR, MODEL_OUTPUT_DIR

from dev_server_ml.__models_registry import (
    Models, 
    models_registry, 
    evaluation_config,
    data_config
)
from dev_server_ml.constants import CATEGORICAL_COLS, TARGET_COL
from dev_server_ml.model import get_xgb_model, get_logistic_model

def load_best_model():
    # 최적의 모델 타입 읽기
    try:
        with open(MODEL_OUTPUT_DIR / evaluation_config["best_model_filename"], 'r') as f:
            model_type = f.read().strip()
    except FileNotFoundError:
        print(f"Warning: {evaluation_config['best_model_filename']} not found, using default XGBoost model")
        model_type = "xgboost"
    
    # 모델 타입에 따라 적절한 모델 로드
    if model_type == "xgboost":
        model = get_xgb_model()
        model.load_model(MODEL_OUTPUT_DIR / models_registry[Models.XGBOOST].filename)
    elif model_type == "logistic":
        model = joblib.load(MODEL_OUTPUT_DIR / models_registry[Models.LOGISTIC].filename)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def inference():
    # load models    
    encoder = joblib.load(MODEL_OUTPUT_DIR / models_registry[Models.ENCODER].filename)
    model = load_best_model()

    # load test data
    test_df = pd.read_csv(
        RAW_DATA_DIR / data_config["test_filename"]
    ).drop(columns=data_config["drop_columns"])
    
    test_encoded = encoder.transform(test_df[CATEGORICAL_COLS])
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(CATEGORICAL_COLS))
    test_df = pd.concat([test_df.drop(columns=CATEGORICAL_COLS).reset_index(drop=True), test_encoded_df], axis=1)

    # predict
    preds = model.predict_proba(test_df)[:,1]
    submit = pd.read_csv(RAW_DATA_DIR / data_config["submission"]["sample_filename"])

    # 결과 저장
    submit[TARGET_COL] = preds
    submit.to_csv(
        RAW_DATA_DIR / data_config["submission"]["output_filename"], 
        encoding=data_config["submission"]["encoding"], 
        index=False
    )

if __name__ == "__main__":
    inference()