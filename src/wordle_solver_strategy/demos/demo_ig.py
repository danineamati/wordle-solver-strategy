import numpy as np
from scipy.stats import entropy as scipy_entropy

from wordle_solver_strategy import (
    bin_table_to_counts,
    get_bin_table,
    get_numeric_representations,
    information_gain,
)
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

    guesses_numba, guesses_char_counts = get_numeric_representations(guesses)
    answers_numba, answers_char_counts = get_numeric_representations(answers)

    bin_table = get_bin_table(guesses_numba, answers_numba, answers_char_counts)

    bin_counts = bin_table_to_counts(
        bin_table, np.full(len(guesses), True), np.full(len(answers), True)
    )

    e = scipy_entropy(bin_counts)
    entropy_sorting = np.argsort(e)

    ig = information_gain(bin_counts)
    ig_sorting = np.argsort(ig)

    print(np.array_equal(entropy_sorting, ig_sorting))


if __name__ == "__main__":
    raise SystemExit(main())
