#!/bin/bash


# Define the program to run
PROGRAM="sudo perf stat -e L1-dcache-load-misses,L1-dcache-loads -r 20 ./build/naive"  # Replace with your program's path

# Define the output file
OUTPUT_FILE="results.txt"

# Clear the output file
echo "" > "$OUTPUT_FILE"

# Define a function to parse program output
parse_output() {
    local output="$1"
    
    # Extract L1-dcache-load-misses
    load_misses=$(echo "$output" | grep "L1-dcache-load-misses" | awk '{print $1}' | tr -d ',')
    
    # Extract L1-dcache-loads
    loads=$(echo "$output" | grep "L1-dcache-loads" | awk '{print $1}' | tr -d ',')
    
    # Format the result as load-misses|loads
    echo "${load_misses}|${loads}"
}




# Define different parameters to run the program with
declare -a params=(1024 2048 4096 8192)  # Replace with actual parameters
declare -a params2=(8 16 32 64)  # Replace with actual parameters
# Loop through each parameter
for param in "${params[@]}"; do
    for param2 in "${params2[@]}"; do
        # Run the program with the current parameter
        program_output=$($PROGRAM "$param" "$param2")
        
        # Parse the program output
        parsed_result=$(parse_output "$program_output" 2>&1)
        echo "$program_output"
        # Append the parsed result to the output file
        echo "Parameter: $param $param2, Result: $parsed_result" >> "$OUTPUT_FILE"
    done
done

echo "Results have been appended to $OUTPUT_FILE"
