import enum
from collections import namedtuple
from pathlib import Path


class Models(enum.Enum):
    XGBOOST = enum.auto()
    ENCODER = enum.auto()
    LOGISTIC = enum.auto()

ModelInfo = namedtuple("ModelInfo", ["filename", "params"])

# 모델 레지스트리 및 설정
models_registry = {
    Models.XGBOOST: ModelInfo(
        filename="xgb_model.json",
        params={
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.15,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "auc"
        }
    ),
    Models.ENCODER: ModelInfo(
        filename="encoder.joblib",
        params={}
    ),
    Models.LOGISTIC: ModelInfo(
        filename="logistic_model.joblib",
        params={
            "C": 1.0,
            "random_state": 42,
            "max_iter": 1000,
            "class_weight": "balanced"
        }
    )
}

# 평가 설정
evaluation_config = {
    "test_size": 0.2,
    "random_state": 42,
    "metric_priority": "auc_score",
    "metrics_filename": "model_metrics.json",
    "best_model_filename": "best_model.txt"
}

# 데이터 처리 설정
data_config = {
    "drop_columns": ["UID"],
    "train_filename": "train.csv",
    "test_filename": "test.csv",
    "submission": {
        "output_filename": "submission.csv",
        "sample_filename": "sample_submission.csv",
        "encoding": "UTF-8-sig"
    }
}