import sys
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

from lib import settings
from lib.classifier import Classifier

def cv_hyperparameters(x_, y_):
    # kind of randomly chose the grid to search over
    grid = {
        'n_estimators': list(np.arange(50, 500, 50)),
        'max_depth': [None] + list(np.arange(1, 50, 5)), # unlimited depth may be the best, who knows
        'min_samples_split': list(np.arange(5, 20, 2)),
    }

    rf = RandomForestClassifier(random_state=settings.RANDOM_STATE)
    print("Starting grid search...")
    grid_search = GridSearchCV(estimator=rf, param_grid=grid, cv=settings.CV_FOLD, n_jobs=settings.N_JOBS, scoring='accuracy')
    grid_search.fit(x_, y_)
    print("Grid search complete.")

    return grid_search.best_params_

def train_model(x_, y_, params: dict = None):
    if params is None:
        params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        }
    model_ = RandomForestClassifier(random_state=settings.RANDOM_STATE, **params) # ** --> unpack dictionary into keyword arguments
    model_.fit(x_, y_)

    return model_

def test_model(model_, x_test, _y_test):
    yhat = model_.predict(x_test)
    return accuracy_score(_y_test, yhat)

def load_model(model_name):
    return joblib.load(f'{settings.MODEL_ROOT}/{model_name}.joblib')

def save_model(model_):
    if not os.path.exists(settings.MODEL_ROOT):
        os.makedirs(settings.MODEL_ROOT)
    joblib.dump(model_, f'{settings.MODEL_ROOT}/{settings.MODEL_NAME}.joblib')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--force-train':
            import shutil
            if os.path.exists(settings.MODEL_ROOT):
                print("Forcing re-training of models...")
                shutil.rmtree(settings.MODEL_ROOT)
                print(f"Removed existing model directory '{settings.MODEL_ROOT}'.")
    
    # load data
    df = pd.read_csv(f'{settings.DATA_ROOT}/final_data.csv')
    X = df.drop(columns=['finalPosition'])
    y = df['finalPosition']

    # split data, cross-validation, and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE)
    model = Classifier()
    
    if not os.path.exists(f'{settings.MODEL_ROOT}/{settings.MODEL_NAME}.joblib'):
        print("Tuning hyperparameters...")
        best_params = cv_hyperparameters(X_train, y_train)

        print(f"Found the best parameters: {best_params}")

        print("Training model...")
        model = train_model(X_train, y_train, best_params)

        # ask if the user wants to save the model
        print("Would you like to save this model? (y/n): ")
        user_input = input().strip().lower()
        if user_input == 'y':
            save_model(model)
            print(f'Model saved in {settings.MODEL_ROOT}/{settings.MODEL_NAME}.joblib')
    else:
        print('Loading existing model...')
        model = load_model(settings.MODEL_NAME)
    
    # test model
    acc = test_model(model, X_test, y_test)
    print(f'Model test accuracy: {acc}')
    f1 = f1_score(y_test, model.predict(X_test), average='weighted')
    print(f'Model test F1 score: {f1}')