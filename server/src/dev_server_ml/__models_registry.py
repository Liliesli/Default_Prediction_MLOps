import enum
from collections import namedtuple


class Models(enum.Enum):
    XGBOOST = enum.auto()
    ENCODER = enum.auto()
    LOGISTIC = enum.auto()

ModelInfo = namedtuple("ModelInfo", "filename")

models_registry = {
    Models.XGBOOST: ModelInfo(
        "xgb_model.json",
    ),
    Models.ENCODER: ModelInfo(
        "encoder.joblib",
    ),
    Models.LOGISTIC: ModelInfo(
        "logistic_model.joblib",
    ),
}