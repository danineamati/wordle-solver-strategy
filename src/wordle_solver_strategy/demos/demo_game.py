import sys
import time

from wordle_solver_strategy import Game, MaxInfoAgent as Agent
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
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    answers, guesses = load_word_lists()
    g = Game(word="liken", verbose=True)
    start = time.time()
    agent = Agent(answers, guesses, mode="standard", first_guess="crane")
    final_guess, n_guesses = agent.play(g)
    end = time.time()
    print(
        f"Solution: '{final_guess}' found with {n_guesses} guesses in {end-start:.2f} seconds"
    )


if __name__ == "__main__":
    raise SystemExit(main())
