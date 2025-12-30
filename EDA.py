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
    
    print("Removing rows with finalPosition > 12 to keep more observations per finalPosition value.")
    df = df[df['finalPosition'] <= 12]
    
    if show_plots:
        print("Description of numerical features:")
        print(df[['q1','q2','q3', 'prevFinalTime', 'prevFastestLapTime', 'prevPoints']].describe())

        print("Showing pairplot of all numerical variables with hue of finalPosition")
        sns.pairplot(df, vars=['q1','q2','q3', 'prevFinalTime', 'prevFastestLapTime', 'prevPoints'], hue='finalPosition')
        plt.show()

        print("Plotting prevFinalTime vs finalPosition")
        sns.scatterplot(data=df, x='finalPosition', y='prevFinalTime')
        plt.show()

        print("Showing histogram plot of finalPosition vs constructorPosition")
        sns.histplot(df, x='constructorPosition', y='finalPosition')
        plt.show()

        print("Showing correlation matrix of all features.")
        sns.heatmap(df.corr(), annot=True, cmap='cool', fmt='.2f')
        plt.show()
        
        print("Showing contingency table between driverId and finalPosition")
        sns.heatmap(pd.crosstab(df['driverId'], df['finalPosition']))
        plt.show()
        print("Showing contingency table between constructorPosition and finalPosition")
        sns.heatmap(pd.crosstab(df['constructorPosition'], df['finalPosition']))
        plt.show()

        print("Showing contingency table between gridPosition and finalPosition")
        sns.heatmap(pd.crosstab(df['gridPosition'], df['finalPosition']))
        plt.show()

        print("Showing contingency table between constructorId and finalPosition")
        sns.heatmap(pd.crosstab(df['constructorId'], df['finalPosition']))
        plt.show()

        print("Seeing how much of the data I can keep. I need the value_counts to be reasonably high for each category in finalPosition.")
        val_counts = df['finalPosition'].value_counts().sort_index()
        print(val_counts)
        sns.barplot(x=val_counts.index, y=val_counts.values)
        plt.show()

    # EDA showed that q3 is the best choice out of the q features, and that all the q features are highly correlated to each other
    # constructorPosition and finalPosition are also correlated enough to be useful
    # prevFinalTime is not useful because it is uncorrelated with finalPosition
    # prevFastestLapTime is highly correlated with q3
    # driverId is weakly correlated with finalPosition and everything else, but I think I'll keep it since we don't have many features
    # circuitId is fairly correlated to other features than finalPosition, so this is unneeded
    # gridPosition is somewhat correlated to finalPosition
    # prevPoints is correlated to finalPosition but it is even more correlated to constructorPosition. So I will drop it.
    selected_features = ['driverId', 'constructorId', 'constructorPosition', 'gridPosition', 'q3', 'finalPosition', 'circuitId']
    print("Based on the EDA, selected features are: ", selected_features)
    df = df[selected_features]
    df.to_csv(f'{DATA_ROOT}/final_data.csv', index=False)
    print("Updated final_data.csv with only selected features.")
