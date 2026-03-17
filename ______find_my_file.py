import os
import fnmatch

# Set the directory to search in
directory = os.path.dirname(__file__)
# Set the string to search for
search_string = r'_simulate_mps'  # example string, replace with your desired string

# Loop over all the .py files in the directory
for root, dirnames, filenames in os.walk(directory):
    for filename in fnmatch.filter(filenames, '*.py'):
        # Open the file
        with open(os.path.join(root, filename), 'r', encoding='utf-8') as f:
            # Check if the string is in the file
            if search_string in f.read():
                print(f'{search_string} found in {os.path.join(root, filename)}')
