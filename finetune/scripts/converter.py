import csv
import json
import re
import ast

# --- Configuration ---
# Make sure this filename matches your CSV file exactly.
csv_file_path = 'train.csv' 
jsonl_file_path = 'openai_finetune_data.jsonl'
# -------------------

def parse_and_format_for_openai(s: str) -> list:
    """
    Cleans the malformed string, parses it, and formats it for OpenAI fine-tuning.
    """
    # Step 1: Replace various whitespace characters with a single space.
    s = re.sub(r'\s+', ' ', s).strip()
    
    # Step 2: Insert commas between adjacent dictionaries.
    s = re.sub(r'\}\s*\{', '}, {', s)
    
    # Step 3: Safely parse the string into a Python list of dictionaries.
    parsed_list = ast.literal_eval(s)
    
    # Step 4: Convert to OpenAI's required format.
    formatted_messages = []
    for turn in parsed_list:
        role = ""
        if turn.get('from') == 'human':
            role = 'user'
        elif turn.get('from') == 'gpt':
            role = 'assistant'
        
        if role and 'value' in turn:
            formatted_messages.append({
                "role": role,
                "content": turn['value']
            })
            
    return formatted_messages

try:
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader) # Skip header
        
        with open(jsonl_file_path, mode='w', encoding='utf-8') as jsonl_file:
            print("⚙️ Starting final conversion for OpenAI... This may take several minutes.")
            count = 0
            skipped = 0
            
            for i, row_list in enumerate(csv_reader, 1):
                if not row_list or len(row_list) < 1:
                    skipped += 1
                    continue
                
                conversation_text = row_list[0]

                try:
                    if not conversation_text:
                        skipped += 1
                        continue

                    # Use our function to clean, parse, and format the data.
                    final_messages = parse_and_format_for_openai(conversation_text)
                    
                    if not final_messages:
                        skipped += 1
                        continue

                    # Create the final JSON object with the "messages" key.
                    output_obj = {"messages": final_messages}
                    
                    # Write the perfectly formatted line to the output file.
                    jsonl_file.write(json.dumps(output_obj) + '\n')
                    count += 1
                except (ValueError, SyntaxError) as e:
                    print(f"--- ⚠️ ERROR on source row {i+1}: Could not parse. ---")
                    print(f"Error Details: {e}\n")
                    skipped += 1

            print(f"\n✅ Conversion complete!")
            print(f"Successfully converted {count} rows.")
            if skipped > 0:
                print(f"Skipped {skipped} rows due to parsing errors.")
            print(f"Output saved to: {jsonl_file_path}")

except FileNotFoundError:
    print(f"❌ ERROR: The file '{csv_file_path}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")