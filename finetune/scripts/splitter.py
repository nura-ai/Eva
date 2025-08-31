import random
import os

# --- Configuration ---
input_file_path = 'openai_finetune_data.jsonl'
output_dir = 'split_datasets' # A new folder will be created
num_partitions = 1000
split_ratio = 0.8 # 80% for training, 20% for validation
# -------------------

try:
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Read {len(lines)} total lines from {input_file_path}.")

    # Shuffle the lines once to ensure randomness across all partitions
    random.shuffle(lines)
    print("Shuffled all lines randomly.")

    # Calculate the size of each large partition
    partition_size = len(lines) // num_partitions
    
    # Loop to create 20 sets of files
    for i in range(num_partitions):
        print(f"Processing partition {i + 1}/{num_partitions}...")
        
        # Define the start and end points for this partition in the shuffled list
        start_index = i * partition_size
        # For the last partition, take all remaining lines
        end_index = (i + 1) * partition_size if i < num_partitions - 1 else len(lines)
        
        partition_lines = lines[start_index:end_index]
        
        # Calculate the split point for this specific partition
        split_point = int(len(partition_lines) * split_ratio)
        
        train_lines = partition_lines[:split_point]
        validation_lines = partition_lines[split_point:]
        
        # Define file names for this partition
        train_file_path = os.path.join(output_dir, f'train_{i + 1}.jsonl')
        validation_file_path = os.path.join(output_dir, f'validation_{i + 1}.jsonl')
        
        # Write the training file
        with open(train_file_path, 'w', encoding='utf-8') as f:
            f.writelines(train_lines)
        
        # Write the validation file
        with open(validation_file_path, 'w', encoding='utf-8') as f:
            f.writelines(validation_lines)

    print("\n✅ Splitting complete!")
    print(f"{num_partitions} pairs of training and validation files have been created in the '{output_dir}' folder.")

except FileNotFoundError:
    print(f"❌ ERROR: The file '{input_file_path}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")