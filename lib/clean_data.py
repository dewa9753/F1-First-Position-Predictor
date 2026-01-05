"""
lib.clean_data is used by preprocess_data.py to clean the original dataset.

Optional arguments:
    --force-clean : If provided, will re-clean the data even if cleaned files already exist.
"""
import pandas as pd
import numpy as np
import os
import sys
import shutil
import settings

## functions
def get_original_data(file_name):
    df = pd.read_csv(f'{settings.ORIGINAL_DATA_ROOT}/{file_name}')
    df.replace(to_replace=r'\N', value=np.nan, inplace=True)
    return df

def create_clean_data(df, columns_to_drop, output_file_name):
    df_cleaned = df.drop(columns_to_drop, axis=1)
    df_cleaned = df_cleaned.dropna()
    df_cleaned.to_csv(f'{settings.DATA_ROOT}/{output_file_name}', index=False)
    print(f'Cleaned {output_file_name} and created {settings.DATA_ROOT}/{output_file_name}')

## clean data if not already cleaned
if __name__ == '__main__':
    if len(sys.argv) > 1 and '--force-clean' in sys.argv:
        if os.path.exists(settings.DATA_ROOT+'/'):
            shutil.rmtree(settings.DATA_ROOT+'/')
            print(f'Force cleaning: removed \'{settings.DATA_ROOT}\' directory to re-clean the data.')

    if not os.path.exists(settings.DATA_ROOT):
        os.makedirs(settings.DATA_ROOT)

    if not os.path.exists(settings.DATA_ROOT + '/circuits_clean.csv'):
        df = get_original_data('circuits.csv')
        new_col_names = {'name': 'circuitName'}
        df.rename(columns=new_col_names, inplace=True)
        create_clean_data(
            df,
            ['url', 'circuitRef', 'lat', 'lng', 'alt', 'location', 'country'],
            'circuits_clean.csv'
        )
    
    if not os.path.exists(settings.DATA_ROOT + '/constructor_results_clean.csv'):
        df = get_original_data('constructor_results.csv')
        create_clean_data(    
            df,
            ['status'],
            'constructor_results_clean.csv'
        )
    
    if not os.path.exists(settings.DATA_ROOT + '/constructor_standings_clean.csv'):
        df = get_original_data('constructor_standings.csv')
        create_clean_data(
            df,
            ['positionText', 'constructorStandingsId'],
            'constructor_standings_clean.csv'
        )
    
    if not os.path.exists(settings.DATA_ROOT + '/constructors_clean.csv'):
        df = get_original_data('constructors.csv')
        new_col_names = {'name': 'constructorName'}
        df.rename(columns=new_col_names, inplace=True)
        create_clean_data(
            df,
            ['url', 'nationality', 'constructorRef'],
            'constructors_clean.csv'
        )
    
    if not os.path.exists(settings.DATA_ROOT + '/drivers_standings_clean.csv'):
        df = get_original_data('driver_standings.csv')
        create_clean_data(
            df,
            ['positionText'],
            'drivers_standings_clean.csv'
        )

    if not os.path.exists(settings.DATA_ROOT + '/drivers_clean.csv'):
        df = get_original_data('drivers.csv')
        new_col_names = {
            'forename': 'driverForename',
            'surname': 'driverSurname'
        }
        df.rename(columns=new_col_names, inplace=True)
        create_clean_data(
            df,
            ['driverRef', 'number', 'code', 'url', 'dob', 'nationality'],
            'drivers_clean.csv'
        )

    if not os.path.exists(settings.DATA_ROOT + '/lap_times_clean.csv'):
        df = get_original_data('lap_times.csv')
        new_col_names = {
            'position': 'positionInLap',
            'lap': 'lapNumber',
            'milliseconds': 'lapMilliseconds'
        }
        df.rename(columns=new_col_names, inplace=True)
        create_clean_data(
            df,
            ['time'],
            'lap_times_clean.csv'
        )
    
    if not os.path.exists(settings.DATA_ROOT + '/pit_stops_clean.csv'):
        df = get_original_data('pit_stops.csv')
        new_col_names = {
            'stop': 'stopNumber',
            'lap': 'stopLap',
            'milliseconds': 'stopMilliseconds'
        }
        df.rename(columns=new_col_names, inplace=True)
        create_clean_data(
            df,
            ['time', 'duration'],
            'pit_stops_clean.csv'
        )

    if not os.path.exists(settings.DATA_ROOT + '/qualifying_clean.csv'):
        df = get_original_data('qualifying.csv')
        new_col_names = {
            'position': 'qualifyingPosition'
        }
        df.rename(columns=new_col_names, inplace=True)
        create_clean_data(
            df,
            ['qualifyId', 'number'],
            'qualifying_clean.csv'
    )
        
    if not os.path.exists(settings.DATA_ROOT + '/races_clean.csv'):
        df = get_original_data('races.csv')
        create_clean_data(
            df,
            ['url', 'time', 'name', 
             'date', 'round', 'fp1_date', 
             'fp1_time', 'fp2_date', 'fp2_time', 
             'fp3_date', 'fp3_time', 'quali_date', 
             'quali_time', 'sprint_date', 'sprint_time'],
            'races_clean.csv'
        )

    if not os.path.exists(settings.DATA_ROOT + '/results_clean.csv'):
        df = get_original_data('results.csv')
        new_col_names = {
            'time': 'finalTime',
            'milliseconds': 'finalMilliseconds',
            'position': 'finalPosition',
            'grid': 'gridPosition',
            'laps': 'totalLaps'
        }
        df.rename(columns=new_col_names, inplace=True)
        create_clean_data(
            df,
            ['resultId', 'number', 'positionText', 'positionOrder', 'rank', 'finalTime'],
            'results_clean.csv'
        )

    if not os.path.exists(settings.DATA_ROOT + '/sprint_results_clean.csv'):
        df = get_original_data('sprint_results.csv')
        new_col_names = {
            'time': 'finalSprintTime',
            'milliseconds': 'finalSprintMilliseconds',
            'position': 'finalSprintPosition'
        }
        df.rename(columns=new_col_names, inplace=True)
        create_clean_data(
            df,
            ['number', 'positionText', 'positionOrder', 'grid', 'laps', 'rank'],
            'sprint_results_clean.csv'
        )

    if not os.path.exists(settings.DATA_ROOT + '/status_clean.csv'):
        df = get_original_data('status.csv')
        new_col_names = {
            'status': 'statusText'
        }
        df.rename(columns=new_col_names, inplace=True)
        create_clean_data(
            df,
            [],
            'status_clean.csv'
        )