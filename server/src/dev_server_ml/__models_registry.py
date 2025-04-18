import enum
from collections import namedtuple


class Models(enum.Enum):
    XGBOOST = enum.auto()
    ENCODER = enum.auto()

ModelInfo = namedtuple("ModelInfo", "filename")

models_registry = {
    Models.XGBOOST: ModelInfo(
        "xgb_model.json",
    ),
    Models.ENCODER: ModelInfo(
        "encoder.joblib",
    ),
}