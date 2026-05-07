import itertools
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from wordle_solver_strategy import (
    Game,
    MaxInfoAgent,
    MaxPruneAgent,
    MaxSplitsAgent,
    get_numeric_representations,
)
from wordle_solver_strategy.paths import allowed_words_dir, experiments_csv_dir


def sub_job(agent_cls, sub_answers, answers, guesses):
    guesses_numba, guesses_char_counts = get_numeric_representations(guesses)
    answers_numba, answers_char_counts = get_numeric_representations(answers)

    results = []

    for answer_idx, answer in enumerate(sub_answers):
        g = Game(word=answer, verbose=False)

        start_time = time.time()

        agent = agent_cls(
            answers,
            guesses,
            mode="standard",
            guess_data=(guesses_numba, guesses_char_counts),
            answer_data=(answers_numba, answers_char_counts),
        )
        final_guess, n_guesses = agent.play(g)

        end_time = time.time()

        results.append((answer, final_guess, n_guesses, end_time - start_time))

    return results


def job(job_dict, answers, guesses):
    pool = Pool(cpu_count())

    split_answers = np.array_split(answers, cpu_count())

    total_result_lists = pool.starmap(
        sub_job,
        (
            (job_dict["agent"], sub_answers, answers, guesses)
            for sub_answers in split_answers
        ),
    )

    total_results = list(itertools.chain.from_iterable(total_result_lists))

    final_guesses = np.array([result[1] for result in total_results])
    failed_mask = np.array(answers) != final_guesses

    failed_words = np.array(answers)[failed_mask]

    n_plays = np.array([result[2] for result in total_results], dtype=float)
    n_plays[failed_mask] = np.NaN

    mean_n_plays = np.nanmean(n_plays)

    play_times = np.array([result[3] for result in total_results])
    play_times[failed_mask] = np.NaN

    mean_play_time = np.nanmean(play_times)

    job_dict.update(
        {
            "agent_name": job_dict["agent"].__name__,
            "mean_n_plays": mean_n_plays,
            "failed_words": failed_words,
            "mean_play_time": mean_play_time,
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

    results = [job({"agent": agent}, answers, guesses) for agent in agent_list]

    out = experiments_csv_dir() / "comparison_standardmode.csv"
    pd.DataFrame(results).to_csv(out, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
