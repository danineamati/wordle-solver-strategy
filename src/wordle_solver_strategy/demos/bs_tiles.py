import numpy as np

from wordle_solver_strategy import get_numeric_representations, mask_candidates
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

    codes = [
        "00100",
        "11000",
    ]

    solution = "caulk"

    solution_idx = answers.index(solution)

    answers_numba, answers_char_counts = get_numeric_representations(answers)

    for code in codes:
        match_int = 0
        for idx, c in enumerate(code):
            match_int += int(c) << idx * 2

        total_mask = np.ones(len(answers)).astype(bool)

        total_mask = np.invert(
            mask_candidates(
                match_int,
                answers_numba[solution_idx, :],
                answers_numba,
                answers_char_counts,
                total_mask,
            )
        )

        print(np.count_nonzero(total_mask.astype(int)))

        remaining_guesses = np.array(answers)[total_mask]
        print(remaining_guesses)


if __name__ == "__main__":
    raise SystemExit(main())
