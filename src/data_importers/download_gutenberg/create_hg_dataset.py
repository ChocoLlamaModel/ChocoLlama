from datasets import Dataset
import re
import os
from datetime import datetime
import argparse
from global_vars import DATA_DIR, MAX_SHARD_SIZE

# Function to extract metadata and text from a file
def extract_metadata_and_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    metadata = {}

    for i, line in enumerate(lines):
        if '*** START' in line or '***START' in line:
            start_book = i + 1
            break
        if ':' not in line:
            continue
        key, value = line.strip().split(":", 1)
        metadata[key.strip().lower()] = value.strip()

    # find the end of the book as well
    for i, line in enumerate(lines):
        if '*** END' in line or '***END' in line:
            end_book = i

    text = "".join(lines[start_book:end_book])
    return metadata, text


def prep_publication(x):
    if isinstance(x, str):
        year_pattern = r'\b\d{4}\b'
        match = re.search(year_pattern, x)
        if match:
            return match.group()
        else:
            return None
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_directory', type=str, required=True)
    args = parser.parse_args()

    # Directory containing your txt files
    files_directory = args.source_directory

    # List to store dataset entries
    dataset_entries = []

    # Loop through each file in the directory
    for filename in os.listdir(files_directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(files_directory, filename)
            try:
                metadata, text = extract_metadata_and_text(file_path)
                print(metadata)
                release_date_str = metadata.get("release date", "")
                # make sure there is no EBOOK number
                release_date_str = release_date_str.split('[')[0].strip()
                release_date = datetime.strptime(release_date_str, "%B %d, %Y")

                # get original publication date
                og_publication_str = metadata.get("original publication", "")
                og_publication = prep_publication(og_publication_str)

                entry = {
                    "title": metadata.get("title", ""),
                    "release_date": release_date,
                    "original_publication": og_publication,
                    "text": text
                }
                dataset_entries.append(entry)
            except Exception as e:
                print(e)

    # Create the dataset
    dataset = Dataset.from_dict({"title": [entry["title"] for entry in dataset_entries],
                                 "release_date": [entry["release_date"] for entry in dataset_entries],
                                 "original_publication": [entry["original_publication"] for entry in dataset_entries],
                                 "text": [entry["text"] for entry in dataset_entries]})

    print(f'Dataset created for {len(dataset)} books.')

    # Save the dataset
    dataset.save_to_disk(
        DATA_DIR / "raw" / f"Gutenberg_{len(dataset)}",
        max_shard_size=MAX_SHARD_SIZE,
    )

if __name__ == "__main__":
    main()
