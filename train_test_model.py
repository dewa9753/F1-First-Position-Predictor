"""
The main script of the project.
It handles loading the data, training the model with hyperparameter tuning, saving/loading the model, and evaluating its performance.
This script assumes: the data has been preprocessed with feature engineering and is stored in data/final_data.csv.

Before running this script, run the following commands:
    py preprocess_data.py --force-final
    py EDA.py
    
Optional arguments:
    --force-train : If provided, forces retraining of the model even if a saved model already exists.

Requires:
- sys
- os
- numpy
- pandas
- scikit-learn
- lib.{settings, classifier}
"""
import sys
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from lib import settings
from lib.classifier import Classifier

if __name__ == '__main__':
    force_train_flag: bool = False
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--force-train':
            import shutil
            if os.path.exists(settings.MODEL_ROOT):
                print("Forcing re-training of models...")
                shutil.rmtree(settings.MODEL_ROOT)
                print(f"Removed existing model directory '{settings.MODEL_ROOT}'.")
            force_train_flag = True
    # load data
    df = pd.read_csv(f'{settings.DATA_ROOT}/final_data.csv')

    # initialize classifier
    # previously found hyperparameters through grid search
    clf = Classifier(clf_type=GradientBoostingClassifier, default_hyper_params={
        'loss': 'log_loss',
        'n_estimators': 100,
        'min_samples_split': 4,
        'min_samples_leaf': 1,
        'min_impurity_decrease': 0.2
        }
    )
    
    if not force_train_flag and os.path.exists(f'{settings.MODEL_ROOT}/{settings.MODEL_NAME}.joblib'):
        print("Loading existing model...")
        clf.load(f'{settings.MODEL_ROOT}/{settings.MODEL_NAME}.joblib', 'podium', df)
    else:
        print("Training new model...")
        
        param_grid = {
            'n_estimators': np.arange(100, 300, 50),
            'min_samples_split': np.arange(2, 5, 1),
            'min_samples_leaf': np.arange(1, 11, 2),
            'min_impurity_decrease': np.arange(0.0, 0.5, 0.1)
        }
        
        # to enable grid search, uncomment the following line and comment out the line after it
        clf.fit(df, y_col='podium', param_grid=None)
        # clf.fit(df, y_col='podium', param_grid=param_grid)
        clf.save(f'{settings.MODEL_ROOT}/{settings.MODEL_NAME}.joblib')
    
    # evaluate model
    results = clf.evaluate()
    print(f"Model Performance: Accuracy = {results['accuracy']:.4f}, F1 Score = {results['f1']:.4f}")