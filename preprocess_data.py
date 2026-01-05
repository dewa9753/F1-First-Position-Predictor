"""
preprocess_data runs the full preprocessing pipeline to create the final dataset for training.

Optional arguments:
    --force-final : If provided, will re-create the final_data.csv even if it already exists. (lib.create_final_data)
    --force-clean: If provided, will re-clean the data even if cleaned files already exist. (lib.clean_data)
"""
import subprocess
import sys

if __name__ == "__main__":
    command_and_args = ['py', '']
    if len(sys.argv) > 1:
        command_and_args.extend(sys.argv[1:])
    try:
        command_and_args[1] = 'lib/import_data.py'
        result = subprocess.run(command_and_args, capture_output=True, text=True, check=True)
        print(result.stdout)

        command_and_args[1] = 'lib/clean_data.py'
        result = subprocess.run(command_and_args, capture_output=True, text=True, check=True)
        print(result.stdout)

        command_and_args[1] = 'lib/create_final_data.py'
        result = subprocess.run(command_and_args, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running subprocesses: {e.stderr}")