import os
import re
import glob

def update_arguments_name(filename):
    # Extract the <x> part from main_<x>.py
    match = re.match(r'main_(.+)\.py', os.path.basename(filename))
    if not match:
        print(f"Skipping {filename}: doesn't match the expected pattern")
        return False
    
    x_value = match.group(1)
    new_name_value = f'{x_value}'

    if new_name_value.startswith('trajectory_'):
        new_name_value = new_name_value.replace('trajectory_', '3d_diffuser_actor')
    
    # Read the file
    with open(filename, 'r') as f:
        content = f.read()
    
    # Replace the name parameter using regex
    updated_content = re.sub(
        r'(class\s+Arguments\s*\(\s*tap\.Tap\s*\)\s*:.*?\n\s*)name\s*:\s*str\s*=\s*[\'"].*?[\'"]',
        f'\\1name: str = \'{new_name_value}\'',
        content,
        flags=re.DOTALL
    )
    
    # Check if any replacement was made
    if content == updated_content:
        print(f"No replacement made in {filename}")
        return False
    
    # Write the updated content back to the file
    with open(filename, 'w') as f:
        f.write(updated_content)
    
    print(f"Updated {filename}: set name = '{new_name_value}'")
    return True

def main():
    # Find all main_*.py files in the current directory
    files = glob.glob('main_*.py')
    
    if not files:
        print("No main_*.py files found in the current directory")
        return
    
    print(f"Found {len(files)} files to process")
    
    # Process each file
    success_count = 0
    for file in files:
        if update_arguments_name(file):
            success_count += 1
    
    print(f"Processed {len(files)} files, updated {success_count} files")

if __name__ == "__main__":
    main()