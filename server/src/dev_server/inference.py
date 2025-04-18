
import pandas as pd
import joblib
from mlops_config import RAW_DATA_DIR, MODEL_OUTPUT_DIR

from dev_server_ml.__models_registry import Models, models_registry
from dev_server_ml.constants import CATEGORICAL_COLS, TARGET_COL
from dev_server_ml.model import get_xgb_model

def inference():
    # load models    
    encoder = joblib.load(MODEL_OUTPUT_DIR / models_registry[Models.ENCODER].filename)
    model = get_xgb_model()
    model.load_model(MODEL_OUTPUT_DIR / models_registry[Models.XGBOOST].filename)

    # load test data
    test_df = pd.read_csv(RAW_DATA_DIR/"test.csv").drop(columns=["UID"])
    test_encoded = encoder.transform(test_df[CATEGORICAL_COLS])
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(CATEGORICAL_COLS))
    test_df = pd.concat([test_df.drop(columns=CATEGORICAL_COLS).reset_index(drop=True), test_encoded_df], axis=1)

    # predict
    preds = model.predict_proba(test_df)[:,1]
    submit = pd.read_csv(RAW_DATA_DIR / 'sample_submission.csv')

    # 결과 저장
    submit[TARGET_COL] = preds
    submit.to_csv(RAW_DATA_DIR / 'submission.csv', encoding='UTF-8-sig', index=False)

if __name__ == "__main__":
    inference()