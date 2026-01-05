"""
lib.create_final_data is used by preprocess_data.py to create the final dataset for training.

Optional arguments:
    --force-final : If provided, will re-create the final_data.csv even if it already exists.
"""
import os
import sys
import pandas as pd
from settings import DATA_ROOT

def form_data_path(filename):
    return f'{DATA_ROOT}/{filename}.csv'

if __name__ == '__main__':
    if len(sys.argv) > 1 and '--force-final' in sys.argv:
        if os.path.exists(form_data_path('final_data')):
            os.remove(form_data_path('final_data'))
            print(f"Force re-creating: removed '{form_data_path('final_data')}' to re-create the final data.")
    
    if not os.path.exists(form_data_path('final_data')):
        # load cleaned data
        dfs = {}
        for file in os.listdir(DATA_ROOT):
            key = file.replace('_clean.csv', '')
            dfs[key] = pd.read_csv(form_data_path(key + '_clean'))

        # start with results_clean data
        final_df = dfs['results'].copy()
        final_df.rename(columns={'finalMilliseconds': 'finalTime'}, inplace=True)
        
        # remove rows with finalPosition <= 12 to keep more observations per class
        final_df = final_df[final_df['finalPosition'] <= 12]

        # insert circuit id
        races_df = dfs['races']
        final_df = final_df.merge(races_df[['raceId', 'circuitId']], on='raceId', how='left')
        #final_df.drop(columns=['circuitId_y'], inplace=True)
        #final_df.rename(columns={'circuitId_x': 'circuitId'}, inplace=True)

        # insert q1,q2,q3 columns from qualifying data
        qualifying_times_df = dfs['qualifying']
        qualifying_times_df.drop(columns=['qualifyingPosition'], inplace=True)
        qualifying_times_df['q1'] = qualifying_times_df['q1'].apply(lambda x: '00:' + x if pd.notna(x) else x)
        qualifying_times_df['q2'] = qualifying_times_df['q2'].apply(lambda x: '00:' + x if pd.notna(x) else x)
        qualifying_times_df['q3'] = qualifying_times_df['q3'].apply(lambda x: '00:' + x if pd.notna(x) else x)
        qualifying_times_df['q1'] = (pd.to_timedelta(qualifying_times_df['q1']).dt.total_seconds()*1000).astype('int64')
        qualifying_times_df['q2'] = (pd.to_timedelta(qualifying_times_df['q2']).dt.total_seconds()*1000).astype('int64')
        qualifying_times_df['q3'] = (pd.to_timedelta(qualifying_times_df['q3']).dt.total_seconds()*1000).astype('int64')
        final_df = final_df.merge(qualifying_times_df, on=['raceId', 'driverId', 'constructorId'], how='left')


        # add previous race numerical statistics
        final_df['prevFinalTime'] = final_df.groupby('driverId')['finalTime'].shift(1)
        final_df.fillna({'prevFinalTime': final_df['prevFinalTime'].mean()}, inplace=True)
        final_df['prevFinalTime'] = final_df['prevFinalTime'].astype('int64')

        results_df = dfs['results']
        results_df['fastestLapTime'] = results_df['fastestLapTime'].apply(lambda x: '00:' + x if pd.notna(x) else x)
        results_df['fastestLapTime'] = pd.to_timedelta(results_df['fastestLapTime']).dt.total_seconds()*1000
        results_df.rename(columns={'fastestLapTime': 'prevFastestLapTime'}, inplace=True)
        final_df = final_df.merge(results_df[['raceId', 'driverId', 'prevFastestLapTime']], on=['raceId', 'driverId'], how='left')

        final_df['prevPoints'] = final_df.groupby('driverId')['points'].shift(1)
        final_df.fillna({'prevPoints': final_df['prevPoints'].mean()}, inplace=True)
        final_df['prevPoints'] = final_df['prevPoints'].astype('int64')

        # insert constructor standing
        constructor_standings_df = dfs['constructor_standings']
        constructor_standings_df.rename(columns={'position': 'constructorPosition'}, inplace=True)
        final_df = final_df.merge(constructor_standings_df[['raceId', 'constructorId', 'constructorPosition']], on=['raceId', 'constructorId'], how='left')
        
        # change final position to binary podium or not
        final_df['finalPosition'] = final_df['finalPosition'].apply(lambda x: 1 if x <= 3 else 0)
        final_df['finalPosition'] = final_df['finalPosition'].astype('int64')
        final_df.rename(columns={'finalPosition': 'podium'}, inplace=True)

        # clean final data
        # drop all columns that are known only after the races / are not useful for prediction
        final_df.drop(columns=['totalLaps', 'raceId', 'points', 'finalTime', 'fastestLap', 'fastestLapTime', 'fastestLapSpeed', 'statusId'], inplace=True)
        final_df.sort_values(by=['driverId'], inplace=True)
        final_df.drop_duplicates(inplace=True)
        final_df.dropna(inplace=True)

        # save final data
        final_df.to_csv(form_data_path('final_data'), index=False)
        print(f"Created final data at '{form_data_path('final_data')}'")

