#!/bin/bash

# Output file name
output_file="merged_python_files.txt"

# Clear the content of the output file if it exists, or create it if it doesn't
> "$output_file"

echo "Output file cleared/created: $output_file"

# Function to process a Python file
process_file() {
    local file="$1"
    local rel_path="${file#./}"  # Remove leading './' if present

    # Add the file path as a header
    echo "$rel_path" >> "$output_file"
    echo "------" >> "$output_file"

    # Add the file contents
    cat "$file" >> "$output_file"

    # Add a newline for separation
    echo -e "\n" >> "$output_file"
}

# Process Python files in the current directory (non-recursively)
echo "Processing files in the current directory..."
for file in *.py; do
    if [ -f "$file" ]; then
        process_file "$file"
    fi
done

# Process each provided directory recursively
if [ $# -gt 0 ]; then
    echo "Processing specified directories..."
    for dir in "$@"; do
        if [ ! -d "$dir" ]; then
            echo "Warning: $dir is not a valid directory. Skipping."
            continue
        fi

        # Find all Python files recursively in the specified directory and process them
        find "$dir" -type f -name "*.py" | sort | while read -r file; do
            process_file "$file"
        done
    done
else
    echo "No additional directories specified. Only processing current directory."
fi

echo "All Python files have been merged into $output_file"