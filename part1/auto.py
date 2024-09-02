import subprocess
import re
import os
import pandas as pd

# Function to run the program with specified parameters
def run_program_with_parameters(params):
    try:
        # Construct the command to run the program
        # command = ["sudo perf stat -e L1-dcache-load-misses,L1-dcache-loads -r 20 ./build/naive"] + params
        command = [
            "sudo", "perf", "stat", "-e", 
            "L1-dcache-load-misses,L1-dcache-loads,instructions", "-r", "20", 
            "./build/prefetch"
        ]
        
        # Run the program and capture stderr output
        result = subprocess.run(command+params, stderr=subprocess.PIPE, text=True)

        # Return the stderr output
        return result.stderr
    except Exception as e:
        print(f"An error occurred while running the program: {e}")
        return ""

# Function to parse the stderr output
def parse_performance_data(input_string):
    # Regex to find the numbers before the labels for L1-dcache-loads and L1-dcache-load-misses
    loads_pattern = r'([\d,]+)\s+L1-dcache-loads'
    misses_pattern = r'([\d,]+)\s+L1-dcache-load-misses'
    instrc_pattern = r'([\d,]+)\s+instructions'
    # Search for patterns in the input string
    loads_match = re.search(loads_pattern, input_string)
    misses_match = re.search(misses_pattern, input_string)
    ins_match = re.search(instrc_pattern, input_string)
    # Extract numbers, preserving commas
    if loads_match and misses_match and ins_match:
        loads = loads_match.group(1)
        misses = misses_match.group(1)
        ins = ins_match.group(1)
        return loads, misses, ins
    else:
        return None, None, None

# Function to append parsed output to a file
def append_to_file(file_path, content):
    try:
        with open(file_path, 'a') as file:
            file.write(content + "\n")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

def main():
    # List of parameters for each run (customize as needed)
    prs = [1024, 2048, 4096, 8192]
    prs_ = [8]

    # Output file path
    output_file = "prefetch.csv"
    #create a new dataframe
    df = pd.DataFrame(columns=['sz','tsz','loads', 'misses','ins','mpki'])

    for x in prs:
        for y in prs_:
            params = [str(x), str(y)]   
            stderr_output = run_program_with_parameters(params)
            # print(f"params: {params}")
            # print(f"stderr_output: {stderr_output}")
            if stderr_output:
                parsed_output = parse_performance_data(stderr_output)
                if parsed_output:
                    #append a new row to the dataframe using concat
                    # print(type(parsed_output[0]))
                    t1 = int(parsed_output[0].replace(',',''))
                    t2 = int(parsed_output[1].replace(',',''))
                    t3 = int(parsed_output[2].replace(',',''))
                    mpki = (t2/t3)*1000
                    df = pd.concat([df, pd.DataFrame([[x, y, t1, t2, t3, mpki]], columns=['sz','tsz','loads', 'misses','ins','mpki'])], ignore_index=True)
                else:
                    print("No parsed output found.")
            else:
                print("No stderr output captured.")
    #save the dataframe to a csv file
    df.to_csv(output_file, index=False)
if __name__ == "__main__":
    main()
