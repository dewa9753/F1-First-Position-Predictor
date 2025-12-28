from pandas import DataFrame
from sklearn.model_selection import train_test_split

from lib import settings

class Classifier:
    def __init__(self, model_type: type, default_hyper_params = None):
        assert(model_type is not None), "model_type must be provided"
        
        self.model_type = model_type
        self.hyper_params = default_hyper_params or {}
        self.model = None
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        
    def grid_search_hyper_params(self, param_grid = None):
        pass
    
    def fit(self, df: DataFrame, y_col: str, param_grid = None):
        if param_grid is not None and isinstance(param_grid, dict):
            self.grid_search_hyper_params(param_grid)
        
        self.model = self.model_type(**self.hyper_params)
        
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
        
        self.model.fit(self.train_x, self.train_y)
    
    def predict(self, x: DataFrame):
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() before predict().")
        
        return self.model.predict(x)
    
    def evaluate(self, df):
        pass
    
    def load(self, path):
        pass
    
    def save(self, path):
        pass