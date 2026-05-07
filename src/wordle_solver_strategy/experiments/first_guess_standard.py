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
    bin_table_to_counts,
    get_bin_table,
    get_numeric_representations,
)
from wordle_solver_strategy.paths import allowed_words_dir, experiments_csv_dir


def sub_job(agent_cls, sub_guesses, answers, guesses):
    guesses_numba, guesses_char_counts = get_numeric_representations(guesses)
    answers_numba, answers_char_counts = get_numeric_representations(answers)

    bin_table = get_bin_table(guesses_numba, answers_numba, answers_char_counts)

    sub_guess_results = []

    for guess_idx, first_guess in enumerate(sub_guesses):
        print(first_guess)
        for answer_idx, answer in enumerate(answers):
            g = Game(word=answer, verbose=False)

            start_time = time.time()
            agent = agent_cls(
                answers,
                guesses,
                mode="standard",
                first_guess=first_guess,
                guess_data=(guesses_numba, guesses_char_counts),
                answer_data=(answers_numba, answers_char_counts),
                bin_table=bin_table,
            )
            final_guess, n_guesses = agent.play(g)
            end_time = time.time()

            sub_guess_results.append(
                (first_guess, answer, final_guess, n_guesses, end_time - start_time)
            )

    return sub_guess_results


def job(job_dict, answers, guesses):
    agent_cls = job_dict["agent"]
    print(agent_cls.__name__)

    top_n = np.minimum(len(guesses), job_dict["top_n"])

    guesses_numba, guesses_char_counts = get_numeric_representations(guesses)
    answers_numba, answers_char_counts = get_numeric_representations(answers)

    bin_table = get_bin_table(guesses_numba, answers_numba, answers_char_counts)

    bin_counts = bin_table_to_counts(
        bin_table, np.full(len(guesses), True), np.full(len(answers), True)
    )

    sort_idx = agent_cls.order_guesses(bin_counts)

    guesses_sorted = np.array(guesses)[sort_idx]
    guesses_sorted = guesses_sorted[:top_n]

    split_guesses = np.array_split(guesses_sorted, cpu_count())

    pool = Pool(cpu_count())

    total_result_lists = pool.starmap(
        sub_job,
        (
            (
                job_dict["agent"],
                sub_guesses,
                answers,
                guesses,
            )
            for sub_guesses in split_guesses
        ),
    )

    total_results = list(itertools.chain.from_iterable(total_result_lists))

    return total_results


def main() -> None:
    answers_path = allowed_words_dir() / "words_answers.txt"
    guesses_path = allowed_words_dir() / "words_guesses.txt"
    with open(answers_path, "r") as answers_file:
        answers = answers_file.read().splitlines()
    with open(guesses_path, "r") as guesses_file:
        guesses = guesses_file.read().splitlines()

    agent_list = [MaxInfoAgent, MaxSplitsAgent, MaxPruneAgent]

    start = time.time()
    results_per_agent = [
        job({"agent": agent, "top_n": 100}, answers, guesses) for agent in agent_list
    ]
    end = time.time()
    print("completed in", end - start)

    cols = ["first_guess", "answer", "final_guess", "n_guesses", "play_time"]

    dataframes = [
        pd.DataFrame(agent_result, columns=cols) for agent_result in results_per_agent
    ]

    for dataframe_idx, dataframe in enumerate(dataframes):
        dataframe["agent"] = agent_list[dataframe_idx].__name__

    results = pd.concat(dataframes, axis=0)

    results[results["answer"] != "final_guess"][["n_guesses", "play_time"]] = np.NaN

    results.to_csv(
        experiments_csv_dir() / "first_guess_deep_standardmode.csv", index=False
    )

    summary = results.groupby(["agent", "first_guess"]).mean().sort_values("n_guesses")
    summary.to_csv(experiments_csv_dir() / "first_guess_deep_standardmode_summary.csv")


if __name__ == "__main__":
    raise SystemExit(main())
