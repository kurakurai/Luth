from datasets import load_dataset
import json

hf_repo = "legmlai/openhermes-fr"
output_file = "datasets/openhermes-fr.jsonl"

# Load only the 'train' split of the OpenHermes-fr dataset
dataset = load_dataset(hf_repo, split='train')

# Function to transform and filter the dataset
def preprocess_example(example):
    if example['bad_entry'] is False:
        messages = [
            {"role": "user", "content": example['prompt']},
            {"role": "assistant", "content": example['accepted_completion']}
        ]
        return {"messages": messages}
    else:
        return None

# Apply the preprocessing
processed_dataset = dataset.map(preprocess_example, remove_columns=['prompt', 'accepted_completion', 'bad_entry'])

# Remove None entries
processed_dataset = processed_dataset.filter(lambda x: x is not None)

# Save the dataset to JSONL
with open(output_file, 'w', encoding='utf-8') as f:
    for example in processed_dataset:
        json.dump(example, f, ensure_ascii=False)
        f.write('\n')
