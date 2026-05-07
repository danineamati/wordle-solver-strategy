import time
from multiprocessing import Pool, cpu_count

import pandas as pd

from wordle_solver_strategy import Game, MaxInfoAgent, MaxPruneAgent, MaxSplitsAgent
from wordle_solver_strategy.paths import allowed_words_dir, experiments_csv_dir


def job(job_dict, answers, guesses):
    g = Game(word="DUMMY_WORD", max_plays=1, verbose=False)

    start_time = time.time()

    agent = job_dict["agent"](answers, guesses, first_guess=None)
    final_guess, _ = agent.play(g)

    end_time = time.time()

    job_dict.update(
        {
            "agent_name": job_dict["agent"].__name__,
            "first_word": final_guess,
            "time_taken": end_time - start_time,
        }
    )

    return job_dict


def main() -> None:
    answers_path = allowed_words_dir() / "words_answers.txt"
    guesses_path = allowed_words_dir() / "words_guesses.txt"
    with open(answers_path, "r") as answers_file:
        answers = answers_file.read().splitlines()
    with open(guesses_path, "r") as guesses_file:
        guesses = guesses_file.read().splitlines()

    agent_list = [MaxInfoAgent, MaxSplitsAgent, MaxPruneAgent]

    pool = Pool(cpu_count())

    results = pool.starmap(
        job, [({"agent": agent}, answers, guesses) for agent in agent_list]
    )

    out = experiments_csv_dir() / "first_guess_shallow.csv"
    pd.DataFrame(results).to_csv(out, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
