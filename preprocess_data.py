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