from pandas import DataFrame
from lib import settings

class Classifier:
    def __init__(self, model_type: type, hyper_params=None):
        assert(model_type is not None), "model_type must be provided"
        
        self.model_type = model_type
        self.model = model_type()
        self.hyper_params = hyper_params
        self.train_x = None
        self.train_y = None
        
    def tune_hyper_params(self, df: DataFrame):
        pass
    
    def fit(self, df: DataFrame, y_col: str):
        if self.hyper_params is None:
            self.tune_hyper_params(df)
        
        self.model = self.model_type(random_state=settings.RANDOM_STATE, **self.hyper_params)
        
        assert(y_col in df.columns), f"{y_col} must be a column in given the dataframe"
        
        self.train_x = df.drop(columns=[y_col])
        self.train_y = df[y_col]
        
        self.model.fit(self.train_x, self.train_y)
    
    def predict(self, X):
        pass
    
    def evaluate(self, df):
        pass
    
    def load(self, path):
        pass
    
    def save(self, path):
        pass