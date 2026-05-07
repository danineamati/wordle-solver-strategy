from wordle import Game, MaxInfoAgent as Agent
import pathlib
import time

CURR_FILE_DIR = pathlib.Path(__file__).parent
ROOT_DIR = CURR_FILE_DIR.parent
WORDS_ANSWERS_PATH = ROOT_DIR / "words_answers.txt"
WORDS_GUESSES_PATH = ROOT_DIR / "words_guesses.txt"

with open(WORDS_ANSWERS_PATH, "r") as answers_file:
    answers = answers_file.read().splitlines()
with open(WORDS_GUESSES_PATH, "r") as guesses_file:
    guesses = guesses_file.read().splitlines()

g = Game(word="liken", verbose=True)

start = time.time()
agent = Agent(answers, guesses, mode="standard", first_guess="crane")
(
    final_guess,
    n_guesses,
) = agent.play(g)
end = time.time()

print(
    f"Solution: '{final_guess}' found with {n_guesses} guesses in {end-start:.2f} seconds"
)
