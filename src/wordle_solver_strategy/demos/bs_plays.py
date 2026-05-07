import ast
from pathlib import Path

import pandas as pd

from wordle_solver_strategy import MaxInfoSolver as Solver
from wordle_solver_strategy.paths import allowed_words_dir


def load_word_lists():
    answers_path = allowed_words_dir() / "words_answers.txt"
    guesses_path = allowed_words_dir() / "words_guesses.txt"
    with open(answers_path, "r") as answers_file:
        answers = answers_file.read().splitlines()
    with open(guesses_path, "r") as guesses_file:
        guesses = guesses_file.read().splitlines()
    return answers, guesses


def main() -> None:
    answers, guesses = load_word_lists()
    ngram_path = Path(__file__).resolve().parent / "ngram_data.csv"
    ngram_data = pd.read_csv(ngram_path)

    ngram_data["timeseries"] = ngram_data["timeseries"].apply(
        lambda x: ast.literal_eval(x)[-1]
    )

    plays = [
        "reast,00100",
        "align,11000",
        "caulk,22222",
    ]

    solver = Solver(answers, guesses, mode="standard")

    for play in plays[:-1]:
        user_in_list = play.split(",")

        guess = user_in_list[0]
        code = user_in_list[1]

        _, remaining_answers = solver.step(code, guess)

    ranked = ngram_data[ngram_data["ngram"].isin(remaining_answers)].sort_values(
        by="timeseries", ascending=False
    )
    print(ranked[["ngram", "timeseries"]])


if __name__ == "__main__":
    raise SystemExit(main())
