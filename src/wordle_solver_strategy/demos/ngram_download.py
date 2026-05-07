import time
from pathlib import Path

import pandas as pd
import requests

from wordle_solver_strategy.paths import allowed_words_dir


def load_answers():
    answers_path = allowed_words_dir() / "words_answers.txt"
    with open(answers_path, "r") as answers_file:
        return answers_file.read().splitlines()


def main() -> None:
    answers = load_answers()
    guess_data_list = []

    for idx, guess in enumerate(answers):
        print(idx, guess)
        time.sleep(1)
        while True:
            try:
                resp = requests.get(
                    url=f"https://books.google.com/ngrams/json?content={guess}&year_start=1800&year_end=2019&corpus=26&smoothing=0"
                )
                data = resp.json()
                guess_data_list.append(data[0])
                break
            except Exception:
                time.sleep(10)

    out_path = Path(__file__).resolve().parent / "ngram_data.csv"
    pd.DataFrame(guess_data_list).to_csv(out_path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
