from pathlib import Path
import pickle


import pandas as pd


TRANSLATION_DIR = Path(__file__).parent
TRANSLATION_PICKLE = TRANSLATION_DIR / "sbi_translations.pickle"
TRANSLATION_CSV = TRANSLATION_DIR / "sbi_translations.csv"

sbi_dutch_to_english: dict[str, str] = {}

if not TRANSLATION_PICKLE.exists():
    if not TRANSLATION_CSV.exists():
        raise FileNotFoundError(
            f"Translation CSV {TRANSLATION_CSV} not found. Please provide the translation file."
        )

    dutch = "Title_Dutch"
    english = "Title_English"

    translations = pd.read_csv(TRANSLATION_CSV, usecols=[dutch, english])

    sbi_dutch_to_english = dict(zip(translations[dutch], translations[english]))

    with open(TRANSLATION_PICKLE, "wb") as f:
        pickle.dump(sbi_dutch_to_english, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(TRANSLATION_PICKLE, "rb") as f:
        sbi_dutch_to_english = pickle.load(f)


def sbi_translate(dutch_title: str) -> str:
    return sbi_dutch_to_english.get(dutch_title, dutch_title)
