import pandas as pd
import requests
import time
import csv
import os
from tqdm import tqdm

INPUT_CSV = '../Data/filtered_titles_to_fetch.csv'  
MAX_CHARACTERS = 5
SLEEP_TIME = 1.2
SAVE_INTERVAL = 50
OUTPUT_CHARACTERS = 'anime_characters.csv'
OUTPUT_DESCRIPTIONS = 'character_descriptions.csv'
PROCESSED_TITLES_FILE = 'processed_titles.txt'
FAILED_TITLES_FILE = 'failed_titles.txt'

# Loads titles
df = pd.read_csv(INPUT_CSV)
anime_titles = df['japanese_names'].dropna().unique()

# Resume support
if os.path.exists(PROCESSED_TITLES_FILE):
    with open(PROCESSED_TITLES_FILE, 'r', encoding='utf-8') as f:
        processed_titles = set(line.strip() for line in f)
else:
    processed_titles = set()

# output containers
anime_characters = []
character_descriptions = []

def safe_api_call(url):
    for _ in range(3):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(SLEEP_TIME)
    return None

# Main Loop
for i, title in enumerate(tqdm(anime_titles, desc="Fetching characters")):
    if title in processed_titles:
        continue

    search_url = f'https://api.jikan.moe/v4/anime?q={title}&limit=1'
    search_data = safe_api_call(search_url)
    if not search_data or 'data' not in search_data or not search_data['data']:
        with open(FAILED_TITLES_FILE, 'a', encoding='utf-8') as f:
            f.write(title + '\n')
        continue

    anime = search_data['data'][0]
    mal_id = anime['mal_id']

    char_url = f'https://api.jikan.moe/v4/anime/{mal_id}/characters'
    char_data = safe_api_call(char_url)
    if not char_data or 'data' not in char_data:
        with open(FAILED_TITLES_FILE, 'a', encoding='utf-8') as f:
            f.write(title + '\n')
        continue

    for entry in char_data['data'][:MAX_CHARACTERS]:
        char_id = entry['character']['mal_id']
        char_name = entry['character']['name']
        role = entry['role']

        anime_characters.append({
            'anime_title': title,
            'anime_mal_id': mal_id,
            'character_id': char_id,
            'character_name': char_name,
            'role': role
        })

        desc_url = f'https://api.jikan.moe/v4/characters/{char_id}'
        desc_data = safe_api_call(desc_url)
        desc = desc_data['data']['about'] if desc_data and 'data' in desc_data and 'about' in desc_data['data'] else None

        character_descriptions.append({
            'character_id': char_id,
            'character_name': char_name,
            'anime_mal_id': mal_id,
            'description': desc
        })

        time.sleep(SLEEP_TIME)

    with open(PROCESSED_TITLES_FILE, 'a', encoding='utf-8') as f:
        f.write(title + '\n')

    if i % SAVE_INTERVAL == 0 and i > 0:
        pd.DataFrame(anime_characters).to_csv(OUTPUT_CHARACTERS, index=False, quoting=csv.QUOTE_ALL)
        pd.DataFrame(character_descriptions).to_csv(OUTPUT_DESCRIPTIONS, index=False, quoting=csv.QUOTE_ALL)

pd.DataFrame(anime_characters).to_csv(OUTPUT_CHARACTERS, index=False, quoting=csv.QUOTE_ALL)
pd.DataFrame(character_descriptions).to_csv(OUTPUT_DESCRIPTIONS, index=False, quoting=csv.QUOTE_ALL)

print("All files saved.")
