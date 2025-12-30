import os
import joblib
from pandas import DataFrame
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split

from lib import settings

class Classifier:
    def __init__(self, clf_type: type, default_hyper_params = None):
        assert(clf_type is not None), "clf_type must be provided"
        
        self.clf_type = clf_type
        self.hyper_params = default_hyper_params or {}
        self.model = None
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        
    def __grid_search_hyper_params(self, param_grid = None):
        assert(param_grid is not None and isinstance(param_grid, dict)), "param_grid must be a defined dictionary"
        assert(self.train_x is not None and self.train_y is not None), "Training data must be set before hyperparameter search"

        rf = self.clf_type(random_state=settings.RANDOM_STATE)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=settings.CV_FOLD, n_jobs=settings.N_JOBS, scoring='accuracy')
        grid_search.fit(self.train_x, self.train_y)
        self.hyper_params = grid_search.best_params_
        
    def __load_data(self, df: DataFrame, y_col: str):
        # default to last column if specified column not found
        if y_col not in df.columns:
            y_col = df.columns[-1]
        
        self.train_x = df.drop(columns=[y_col])
        self.train_y = df[y_col]
        
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.train_x, self.train_y,
            test_size=settings.TEST_SIZE,
            random_state=settings.RANDOM_STATE
        )
    
    def fit(self, df: DataFrame, y_col: str, param_grid = None):
        self.__load_data(df, y_col)
        
        if self.model is None:
            if param_grid is not None and isinstance(param_grid, dict):
                print("Performing hyperparameter grid search...")
                self.__grid_search_hyper_params(param_grid)
                print(f"Best Hyperparameters: {self.hyper_params}")
        
        if self.model is None:
            self.model = self.clf_type(**self.hyper_params)
            
        self.model.fit(self.train_x, self.train_y)
    
    def predict(self, x: DataFrame):
        assert(self.model is not None), "Model must be trained before prediction"
        return self.model.predict(x)
    
    def evaluate(self, df):
        assert(self.model is not None), "Model must be trained before evaluation"
        assert(self.test_x is not None and self.test_y is not None), "Test data must be set before evaluation"
        
        yhat = self.model.predict(self.test_x)
        return {"accuracy": accuracy_score(self.test_y, yhat), "f1": f1_score(self.test_y, yhat, average='weighted')}
    
    def load(self, path):
        try:
            self.model = joblib.load(path)
            self.clf_type = type(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {str(e)}") from e
    
    def save(self, path):
        assert(self.model is not None), "Model must be trained before saving"
        
        if not os.path.exists(settings.MODEL_ROOT):
            os.makedirs(settings.MODEL_ROOT)
        try:
            joblib.dump(self.model, f'{settings.MODEL_ROOT}/{settings.MODEL_NAME}.joblib')
        except Exception as e:
            raise RuntimeError(f"Failed to save model to {path}: {str(e)}") from e