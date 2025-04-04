import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

class CreditDefaultModel:
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path)
        self.best_params = None
        self.best_scaler = None
        self.uid_col = 'UID'
        
        if self.uid_col in self.df.columns:
            self.df.drop(columns=[self.uid_col], inplace=True)

    def preprocess_data(self, target_col: str, scaler):
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', scaler, numeric_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
            ]
        )

        X_processed = preprocessor.fit_transform(X)
        return train_test_split(X_processed, y, test_size=0.2, random_state=42), preprocessor

    def tune_hyperparameters(self, target_col: str, param_grid: Dict):
        scalers = [StandardScaler(), MinMaxScaler(), RobustScaler()]
        best_score = 0
        best_model = None
        
        for scaler in scalers:
            (X_train, X_test, y_train, y_test), preprocessor = self.preprocess_data(target_col, scaler)
            
            xgb_clf = xgb.XGBClassifier(eval_metric='logloss')
            grid_search = GridSearchCV(xgb_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            y_pred = grid_search.best_estimator_.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            
            if score > best_score:
                best_score = score
                self.best_params = grid_search.best_params_
                self.best_scaler = scaler
                best_model = grid_search.best_estimator_

        print(f"Best Scaler: {self.best_scaler}")
        print(f"Best Parameters: {self.best_params}")
        return best_model
    
    def evaluate_model(self, target_col: str, model):        
        (X_train, X_test, y_train, y_test), _ = self.preprocess_data(target_col, self.best_scaler)
        
        y_pred = model.predict(X_test) 
        print(classification_report(y_test, y_pred))
        
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        return results

    def predict(self, model, X_new: pd.DataFrame):
        categorical_cols = X_new.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = X_new.select_dtypes(include=[np.number]).columns.tolist()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.best_scaler, numeric_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        X_processed = preprocessor.fit_transform(X_new)
        predictions = model.predict(X_processed)
        return predictions

if __name__ == "__main__":
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1]
    }
    
    model = CreditDefaultModel("../data/train.csv")
    best_model = model.tune_hyperparameters("채무 불이행 여부", param_grid)
    model.evaluate_model("채무 불이행 여부", best_model)
    
    # test_df = pd.read_csv("../data/test.csv")
    # uids = test_df[model.uid_col] if model.uid_col in test_df.columns else None
    # if model.uid_col in test_df.columns:
    #     test_df.drop(columns=[model.uid_col], inplace=True)
    
    # predictions = model.predict(best_model, test_df)
    
    # result_df = pd.DataFrame({
    #     "UID": uids,
    #     "Prediction": predictions
    # })
    # result_df.to_csv("data/predictions.csv", index=False)
