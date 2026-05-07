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
        if len(remaining_answers) > 1:
            print(
                f"Next guess: {suggested_guess}, {len(remaining_answers)} answers remaining: {remaining_answers}",
            )
        else:
            print(f"Answer is {suggested_guess}")
            break


if __name__ == "__main__":
    raise SystemExit(main())
