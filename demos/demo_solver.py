import pathlib

from wordle import MaxInfoSolver as Solver

CURR_FILE_DIR = pathlib.Path(__file__).parent
ROOT_DIR = CURR_FILE_DIR.parent
WORDS_ANSWERS_PATH = ROOT_DIR / "words_answers.txt"
WORDS_GUESSES_PATH = ROOT_DIR / "words_guesses.txt"

with open(WORDS_ANSWERS_PATH, "r") as answers_file:
    answers = answers_file.read().splitlines()
with open(WORDS_GUESSES_PATH, "r") as guesses_file:
    guesses = guesses_file.read().splitlines()

solver = Solver(answers, guesses, mode="standard")

print("-----------------")
print("WORDLE SOLVER")
print("-----------------")
print(
    "Code Format\n",
    "No Match (Gray): 0\n",
    "Partial Match (Yellow) 1\n",
    "Exact Match (Green): 2",
)
print("-----------------")

print("Optimal starting guess: reast")

while True:
    print("-----------------")
    user_in = input("Enter guess,code: ")
    if len(user_in) == 0:
        break

    user_in_list = user_in.split(",")

    try:
        guess = user_in_list[0]
        code = user_in_list[1]
        if not len(guess) == 5 and not len(code) == 5:
            raise Exception
    except Exception:
        print("invalid input")
        continue

    suggested_guess, remaining_answers = solver.step(code, guess)
    if (len(remaining_answers)) > 1:
        print(
            f"Next guess: {suggested_guess}, {len(remaining_answers)} answers remaining: {remaining_answers}",
        )
    else:
        print(f"Answer is {suggested_guess}")
        break
