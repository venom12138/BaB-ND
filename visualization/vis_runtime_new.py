import os
import numpy as np
import json
import matplotlib.pyplot as plt
import re

file_dir = 'scripts_iclr'

def process_runtime():
    num_test = 1
    num_iter = 12
    num_record = num_test * num_iter
    file_path_dict = {file.split('_')[0]: os.path.join(file_dir, file) for file in os.listdir(file_dir) if file.endswith('.txt')}
    file_path_dict = {"test": os.path.join(file_dir, "time_test.txt")}
    pattern = r"---------- (\w+): (\d+\.\d{8})----------"
    time_dict = {}
    for model_id, file_path in file_path_dict.items():
        end_times = []
        cumulative_times = []
        runtimes = {}

        with open(file_path, 'r') as file:
            contents = file.readlines()  # Read the file line by line

            # Loop over each line and find matches
            for line in contents:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1)
                    time = match.group(2)
                    if name in runtimes:
                        runtimes[name] += float(time)
                    else:
                        runtimes[name] = float(time)
                elif 'Result: unknown in' in line:
                    value = float(line.split('in')[1].split('seconds')[0].strip())
                    end_times.append(value)
                elif 'Cumulative time:' in line:
                    value = float(line.split('Cumulative time:')[1].strip())
                    cumulative_times.append(value)

        assert len(end_times) == num_test
        assert len(cumulative_times) == num_record
        cumulative_times = np.array(cumulative_times).reshape(num_test, num_iter).tolist()
        # for key, value in runtimes.items():
        #     assert len(value) == num_record
        #     value = np.array(value).reshape(num_test, num_iter).tolist()
        time_dict[model_id] = {
            'end_times': end_times,
            'cumulative_times': cumulative_times,
            'runtimes': runtimes
        }
    
    # save to json
    json_file = os.path.join(file_dir, 'time_dict.json')
    with open(json_file, 'w') as file:
        json.dump(time_dict, file, indent=4)

    print('Saved to', json_file)
    

if __name__ == '__main__':
    process_runtime()