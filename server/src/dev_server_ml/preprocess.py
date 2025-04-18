import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

from .constants import CATEGORICAL_COLS, TARGET_COL

def prepare_dataset(
    path: Path,
    drop_columns: list,
    test_size: float = 0.2,
    random_state: int = 42
):
    dataset = pd.read_csv(path).drop(columns=drop_columns)

    # encoder 학습
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(dataset[CATEGORICAL_COLS])
    
    encoded = encoder.transform(dataset[CATEGORICAL_COLS])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(CATEGORICAL_COLS))

    df_encoded = pd.concat(
        [dataset.drop(columns=CATEGORICAL_COLS).reset_index(drop=True), encoded_df],
        axis=1
    )

    # separate features and target variable
    X = df_encoded.drop(columns=[TARGET_COL])
    y = df_encoded[TARGET_COL]

    # separate train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_val, y_train, y_val, encoder