import os
import json
import math
import shutil

def split_dataset(base_dataset_name='fb15k-237', num_splits=4):
    """
    Splits the KICGPT dataset into a specified number of parts for distributed processing.
    """
    print(f"Starting to split dataset '{base_dataset_name}' into {num_splits} parts...")

    base_path = os.path.join('dataset', base_dataset_name)
    if not os.path.exists(base_path):
        print(f"Error: Base dataset path not found at '{base_path}'")
        return

    # 1. Load the main test file and split it
    test_answer_path = os.path.join(base_path, 'test_answer.txt')
    with open(test_answer_path, 'r', encoding='utf-8') as f:
        test_samples = json.load(f)
    
    chunk_size = math.ceil(len(test_samples) / num_splits)
    split_samples = [test_samples[i:i + chunk_size] for i in range(0, len(test_samples), chunk_size)]
    print(f"Test samples split into {len(split_samples)} chunks.")

    # Load mapping files needed for key generation
    with open(os.path.join(base_path, 'get_neighbor', 'entity2id.txt'), 'r', encoding='utf-8') as f:
        ent2id = {line.strip().split('\t')[0]: line.strip().split('\t')[1] for line in f}
    with open(os.path.join(base_path, 'get_neighbor', 'relation2id.txt'), 'r', encoding='utf-8') as f:
        rel2id = {line.strip().split('\t')[0]: line.strip().split('\t')[1] for line in f}

    # 2. Process each split
    for i in range(num_splits):
        part_num = i + 1
        part_dataset_name = f"{base_dataset_name}-part{part_num}"
        part_path = os.path.join('dataset', part_dataset_name)
        
        print(f"\n--- Processing Part {part_num}/{num_splits} ---")
        print(f"Creating directory: {part_path}")
        os.makedirs(os.path.join(part_path, 'demonstration'), exist_ok=True)
        os.makedirs(os.path.join(part_path, 'alignment'), exist_ok=True)

        # Copy small, essential files that don't need splitting
        shutil.copytree(os.path.join(base_path, 'get_neighbor'), os.path.join(part_path, 'get_neighbor'))
        shutil.copy(os.path.join(base_path, 'entity2text.txt'), part_path)
        shutil.copy(os.path.join(base_path, 'relation2text.txt'), part_path)

        # Save the split test answer file
        current_samples = split_samples[i]
        with open(os.path.join(part_path, 'test_answer.txt'), 'w', encoding='utf-8') as f:
            json.dump(current_samples, f, indent=1)
        print(f"Saved split 'test_answer.txt' with {len(current_samples)} samples.")

        # Determine the necessary keys for this split
        required_keys_head = set()
        required_keys_tail = set()
        for sample in current_samples:
            head_key = '\t'.join([sample['Answer'], sample['Question']])
            tail_key = '\t'.join([sample['HeadEntity'], sample['Question']])
            required_keys_head.add(head_key)
            required_keys_tail.add(tail_key)

            head_id_key = '\t'.join([ent2id.get(sample['Answer'], ''), rel2id.get(sample['Question'], '')])
            tail_id_key = '\t'.join([ent2id.get(sample['HeadEntity'], ''), rel2id.get(sample['Question'], '')])
            required_keys_head.add(head_id_key)
            required_keys_tail.add(tail_id_key)

        # 3. Split the large data files based on required keys
        files_to_split = {
            'demonstration/head_analogy.txt': required_keys_head,
            'demonstration/tail_analogy.txt': required_keys_tail,
            'demonstration/head_supplement.txt': required_keys_head,
            'demonstration/tail_supplement.txt': required_keys_tail,
            'demonstration/T_link_base_head.txt': required_keys_head,
            'demonstration/T_link_base_tail.txt': required_keys_tail,
            'retriever_candidate_head.txt': required_keys_head,
            'retriever_candidate_tail.txt': required_keys_tail,
        }

        for file_path, required_keys in files_to_split.items():
            full_path = os.path.join(base_path, file_path)
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    full_data = json.load(f)
                
                # Filter the dictionary
                split_data = {k: v for k, v in full_data.items() if k in required_keys}
                
                # Save the filtered data
                dest_path = os.path.join(part_path, file_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(dest_path, 'w', encoding='utf-8') as f:
                    json.dump(split_data, f, indent=1)
                print(f"Split and saved '{file_path}'. Kept {len(split_data)}/{len(full_data)} items.")

    print("\nDataset splitting complete!")

if __name__ == '__main__':
    # You can change the dataset name and number of splits here
    split_dataset(base_dataset_name='fb15k-237', num_splits=4)
    # split_dataset(base_dataset_name='wn18rr', num_splits=4) 