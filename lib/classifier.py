"""
Classifier class module that encapsulates model training, hyperparameter tuning, and evaluation.

This module defines a Classifier class that can be used to train machine learning models using scikit-learn.
It supports hyperparameter tuning via grid search, model evaluation using accuracy and F1 score.
It also provides methods to save and load trained models using joblib.

Requires:
- pandas
- scikit-learn
- joblib
"""

import os
import joblib
from pandas import DataFrame
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split

from lib import settings

class Classifier:
    """
    Classification wrapper class for training, hyperparameter tuning, evaluation, saving, and loading of sklearn classification models.
    
    Attributes:
        clf_type (type): The sklearn classifier type to be used (e.g., GradientBoostingClassifier).
        hyper_params (dict): Hyperparameters for the classifier.
        model: The trained sklearn model instance.
        train_x (DataFrame): Training feature data.
        test_x (DataFrame): Testing feature data.
        train_y (Series): Training target data.
        test_y (Series): Testing target data.
        
    Methods:
        fit(df: DataFrame, y_col: str, param_grid = None): Train the model with optional hyperparameter grid search.
        predict(x: DataFrame): Make predictions using the trained model.
        evaluate(): Evaluate the model's performance on the test set.
        load(path: str, y_col: str, df: DataFrame): Load a trained model from disk.
        save(path: str): Save the trained model to disk. 
        
    Raises:
        AssertionError: If required attributes are not set before method calls.
        RuntimeError: If loading or saving the model fails.
    """
    
    def __init__(self, clf_type: type, default_hyper_params = None):
        assert(clf_type is not None), "clf_type must be provided"
        
        self.clf_type = clf_type
        self.hyper_params = default_hyper_params or {}
        self.model = None
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        
    def _grid_search_hyper_params(self, param_grid = None):
        assert(param_grid is not None and isinstance(param_grid, dict)), "param_grid must be a defined dictionary"
        assert(self.train_x is not None and self.train_y is not None), "Training data must be set before hyperparameter search"

        # perform grid search
        rf = self.clf_type(random_state=settings.RANDOM_STATE)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=settings.CV_FOLD, n_jobs=settings.N_JOBS, scoring='accuracy')
        grid_search.fit(self.train_x, self.train_y)
        
        # update hyperparameters with best found parameters
        self.hyper_params = grid_search.best_params_
        
    def _load_data(self, df: DataFrame, y_col: str):
        # default to last column if specified column not found
        if y_col not in df.columns:
            y_col = df.columns[-1]
        
        # split data into features and target
        self.train_x = df.drop(columns=[y_col])
        self.train_y = df[y_col]
        
        # split into training and testing sets
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.train_x, self.train_y,
            test_size=settings.TEST_SIZE,
            random_state=settings.RANDOM_STATE
        )
    
    def fit(self, df: DataFrame, y_col: str, param_grid = None):
        """
        Train the classifier model with optional hyperparameter grid search.
        Set param_grid to None to bypass grid search and use default hyperparameters.
        
        :param df: DataFrame containing feature and target data
        :type df: DataFrame
        :param y_col: Name of the target column in df
        :type y_col: str
        :param param_grid: Dictionary defining hyperparameter grid for search (or None to skip)
        :type param_grid: dict or None
        
        :return: None
        :rtype: None
        
        :raises AssertionError: if training data is not properly set
        """
        self._load_data(df, y_col)
        
        if self.model is None:
            if param_grid is not None and isinstance(param_grid, dict):
                print("Performing hyperparameter grid search...")
                self._grid_search_hyper_params(param_grid)
                print(f"Best Hyperparameters: {self.hyper_params}")
        
        if self.model is None:
            self.model = self.clf_type(**self.hyper_params)
            
        self.model.fit(self.train_x, self.train_y)
    
    def predict(self, x: DataFrame):
        """
        Make predictions using the trained model and the given feature data.
        
        :param x: DataFrame containing feature data for prediction (must match training features)
        :type x: DataFrame
        
        :return: Predicted class labels
        :rtype: ndarray
        """
        assert(self.model is not None), "Model must be trained before prediction"
        return self.model.predict(x)
    
    def evaluate(self):
        """
        Evaluate the trained model's performance on the test set using accuracy and F1 score.
        :return: Dictionary containing accuracy and F1 score
        :rtype: dict
        """
        assert(self.model is not None), "Model must be trained before evaluation"
        assert(self.test_x is not None and self.test_y is not None), "Test data must be set before evaluation"
        
        yhat = self.model.predict(self.test_x)
        return {"accuracy": accuracy_score(self.test_y, yhat), "f1": f1_score(self.test_y, yhat, average='weighted')}
    
    def load(self, path, y_col, df: DataFrame):
        """
        Load the trained model from path and prepare data for evaluation.
        
        :param path: Description
        :param y_col: Description
        :param df: Description
        :type df: DataFrame
        
        :return: None
        :rtype: None
        
        :raises RuntimeError: if loading the model fails
        """
        try:
            self.model = joblib.load(path)
            self.clf_type = type(self.model)
            self._load_data(df, y_col=y_col)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {str(e)}") from e
    
    def save(self, path):
        """
        Save the trained model in memory to the specified path.
        
        :param path: Description
        :type path: str
        
        :return: None
        :rtype: None
        """
        assert(self.model is not None), "Model must be trained before saving"
        
        if not os.path.exists(settings.MODEL_ROOT):
            os.makedirs(settings.MODEL_ROOT)
        try:
            joblib.dump(self.model, f'{settings.MODEL_ROOT}/{settings.MODEL_NAME}.joblib')
        except Exception as e:
            raise RuntimeError(f"Failed to save model to {path}: {str(e)}") from e