import sys
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

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
    clf = Classifier(clf_type=RandomForestClassifier)
    
    if not FORCE_TRAIN and os.path.exists(f'{settings.MODEL_ROOT}/{settings.MODEL_NAME}.joblib'):
        print("Loading existing model...")
        clf.load(f'{settings.MODEL_ROOT}/{settings.MODEL_NAME}.joblib')
        clf.fit(df, y_col='finalPosition')
    else:
        print("Training new model...")
        param_grid = {
            'n_estimators': np.arange(100, 500, 50),
            'max_depth': np.arange(10, 100, 10),
            'min_samples_split': np.arange(2, 5, 1),
            'min_samples_leaf': np.arange(1, 11, 2)
        }
        clf.fit(df, y_col='finalPosition', param_grid=param_grid)
        clf.save(f'{settings.MODEL_ROOT}/{settings.MODEL_NAME}.joblib')
    
    # evaluate model
    results = clf.evaluate(df)
    print(f"Model Performance: Accuracy = {results['accuracy']:.4f}, F1 Score = {results['f1']:.4f}")