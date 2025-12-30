import sys
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from lib import settings
from lib.classifier import Classifier

if __name__ == '__main__':
    FORCE_TRAIN: bool = False
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--force-train':
            import shutil
            if os.path.exists(settings.MODEL_ROOT):
                print("Forcing re-training of models...")
                shutil.rmtree(settings.MODEL_ROOT)
                print(f"Removed existing model directory '{settings.MODEL_ROOT}'.")
            FORCE_TRAIN = True
    # load data
    df = pd.read_csv(f'{settings.DATA_ROOT}/final_data.csv')

    # initialize classifier
    # previously found hyperparameters through grid search
    clf = Classifier(clf_type=GradientBoostingClassifier, default_hyper_params={
        'loss': 'log_loss',
        'n_estimators': 100,
        'min_samples_split': 2,
        'min_samples_leaf': 9,
        'min_impurity_decrease': 0.4
        }
    )
    
    if not FORCE_TRAIN and os.path.exists(f'{settings.MODEL_ROOT}/{settings.MODEL_NAME}.joblib'):
        print("Loading existing model...")
        clf.load(f'{settings.MODEL_ROOT}/{settings.MODEL_NAME}.joblib', df)
    else:
        print("Training new model...")
        param_grid = {
            'loss': ['log_loss', 'exponential'],
            'n_estimators': np.arange(50, 300, 50),
            'min_samples_split': np.arange(2, 5, 1),
            'min_samples_leaf': np.arange(1, 11, 2),
            'min_impurity_decrease': np.arange(0.0, 0.5, 0.1)
        }
        #clf.fit(df, y_col='finalPosition', param_grid=param_grid)
        clf.fit(df, y_col='finalPosition', param_grid=None)
        clf.save(f'{settings.MODEL_ROOT}/{settings.MODEL_NAME}.joblib')
    
    # evaluate model
    results = clf.evaluate(df)
    print(f"Model Performance: Accuracy = {results['accuracy']:.4f}, F1 Score = {results['f1']:.4f}")