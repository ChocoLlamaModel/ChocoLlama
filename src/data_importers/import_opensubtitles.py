from datasets import load_dataset, concatenate_datasets, Dataset
import argparse
from tqdm import tqdm
from global_vars import DATA_DIR, MAX_SHARD_SIZE
from typing import Tuple, Optional

LANGUAGE_MAP = {
    'ar': 'Arabisch',
    'bn': 'Bengaals',
    'ca': 'Catalaans',
    'en': 'Engels',
    'es': 'Spaans',
    'eu': 'Baskisch',
    'fr': 'Frans',
    'hi': 'Hindi',
    'id': 'Indonesisch',
    'ml': 'Malayalam',
    'pt': 'Portugees',
    'ta': 'Tamil',
    'te': 'Telugu',
    'ur': 'Urdu',
    'vi': 'Vietnamees',
    'zhs': 'Chinees (vereenvoudigd)',
    'zht': 'Chinees (traditioneel)',
    'nl': 'Nederlands'
}

def group_by_films(dataset, language='nl'):
    dataset_entries = []

    previous_filmid = None

    for i in tqdm(range(len(dataset))):

        film_id = dataset[i]['meta']['imdbId']

        if previous_filmid is None:
            film_samples = dataset[i]['translation'][language]
            previous_filmid = dataset[i]['meta']['imdbId']

        elif film_id != previous_filmid:
            ### switching to a new film

            # so first saving old film
            film_entry = {
                "imdbId": previous_filmid,
                "text": film_samples
            }
            dataset_entries.append(film_entry)

            # then continue to other film
            film_samples = dataset[i]['translation'][language]
            previous_filmid = film_id

        else:
            film_samples += ' ' + dataset[i]['translation'][language]
            previous_filmid = dataset[i]['meta']['imdbId']

    dataset = Dataset.from_dict({"imdbId": [entry["imdbId"] for entry in dataset_entries],
                                 "text": [entry["text"] for entry in dataset_entries]})

    return dataset

def process_pair(from_language: str, to_language: str, num_proc: int) -> Optional[Tuple[Dataset,Dataset]]:
    print(f'Processing {from_language}-{to_language}')
    try:
        dataset = load_dataset("open_subtitles", lang1=from_language, lang2=to_language, num_proc=num_proc)
    except FileNotFoundError:
        return
    dataset = concatenate_datasets(list(dataset.values()))
    nl_ds = group_by_films(dataset)
    print(f'Processing {from_language}-{to_language} done with {len(nl_ds)} unique movies.')
    return nl_ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_workers", type=int, default=1, required=False)
    args = parser.parse_args()

    nl_dataset = []
    for language in LANGUAGE_MAP:
        if language == 'nl':
            continue

        for from_language, to_language in ((language,'nl'),('nl',language)):
            result = process_pair(from_language,to_language,args.nb_workers)
            if result is not None:
                nl_dataset.append(result)
    
    nl_dataset = concatenate_datasets(nl_dataset)

    nl_dataset.save_to_disk(
        DATA_DIR / "raw" / "OpenSubtitles_by_movie_NL",
        max_shard_size=MAX_SHARD_SIZE,
    )

if __name__ == "__main__":
    main()
