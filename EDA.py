"""
EDA is used to perform exploratory data analysis on the final dataset.
To be used after preprocess_data.py has been run to create final_data.csv.

Optional arguments:
    --show-plots : If provided, will display plots for EDA.
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from lib.settings import DATA_ROOT
import sys

if __name__ == '__main__':
    show_plots = False
    if len(sys.argv) > 1 and '--show-plots' in sys.argv:
        show_plots = True

    df = pd.read_csv(f'{DATA_ROOT}/final_data.csv')
    print("All final data columns: ", df.columns.tolist())
    
    if show_plots:
        print("Description of numerical features:")
        print(df[['q1','q2','q3', 'prevFinalTime', 'prevFastestLapTime', 'prevPoints', 'circuitId']].describe())

        print("Showing pairplot of all numerical variables with hue of podium.")
        sns.pairplot(df, vars=['q1','q2','q3', 'prevFinalTime', 'prevFastestLapTime', 'prevPoints', 'circuitId'], hue='podium', diag_kind='kde')
        plt.show()

        print("Showing correlation matrix of all features.")
        sns.heatmap(df.corr(), annot=True, cmap='cool', fmt='.2f')
        plt.show()
        
        print("Showing contingency table between driverId and podium")
        sns.heatmap(pd.crosstab(df['driverId'], df['podium']))
        plt.show()
        print("Showing contingency table between constructorPosition and podium")
        sns.heatmap(pd.crosstab(df['constructorPosition'], df['podium']))
        plt.show()

        print("Showing contingency table between gridPosition and podium")
        sns.heatmap(pd.crosstab(df['gridPosition'], df['podium']))
        plt.show()

        print("Showing contingency table between constructorId and podium")
        sns.heatmap(pd.crosstab(df['constructorId'], df['podium']))
        plt.show()

    # Based on EDA, select features that are most relevant to predicting podium finish
    selected_features = ['driverId', 'constructorId', 'circuitId', 'prevPoints', 'constructorPosition', 'gridPosition', 'q3', 'podium']
    print("Based on the EDA, selected features are: ", selected_features)
    df = df[selected_features]
    df.to_csv(f'{DATA_ROOT}/final_data.csv', index=False)
    print("Updated final_data.csv with only selected features.")
