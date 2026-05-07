import itertools
import string
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

from wordle_solver_strategy import (
    Game,
    MaxInfoAgent,
    MaxPruneAgent,
    MaxSplitsAgent,
    get_bin_table,
    get_numeric_representations,
)


def sub_job(agent_cls, sub_answers, guesses):
    guesses_numba, guesses_char_counts = get_numeric_representations(guesses)
    answers_numba, answers_char_counts = get_numeric_representations(sub_answers)

    bin_table = get_bin_table(guesses_numba, answers_numba, answers_char_counts)

    sub_answer_results = []

    for sub_answer_idx, sub_answer in enumerate(sub_answers):
        g = Game(word=sub_answer, verbose=False)
        agent = agent_cls(
            sub_answers,
            guesses,
            mode="standard",
            first_guess=None,
            guess_data=(guesses_numba, guesses_char_counts),
            answer_data=(answers_numba, answers_char_counts),
            bin_table=bin_table,
        )

        final_guess, n_guesses = agent.play(g)

        sub_answer_results.append((sub_answer, final_guess, n_guesses))

    return sub_answer_results


def job(job_dict, guesses):
    agent_cls = job_dict["agent"]
    print(agent_cls.__name__)

    pool = Pool(cpu_count())

    job_results = []

    counter = 0
    for train_index, test_index in job_dict["splitter"].split(guesses):
        print(counter + 1)

        split_answers = list(np.array(guesses)[test_index])

        worker_split_answers = np.array_split(split_answers, cpu_count())

        start = time.time()
        results = pool.starmap(
            sub_job,
            ((agent_cls, sub_answers, guesses) for sub_answers in worker_split_answers),
        )
        end = time.time()

        cv_results = list(itertools.chain.from_iterable(results))

        result_answers = np.array([result[0] for result in cv_results])
        result_final_guesses = np.array([result[1] for result in cv_results])
        n_plays = np.array([result[2] for result in cv_results], dtype=float)

        n_plays[result_answers != result_final_guesses] = np.NaN

        job_results.append(
            (agent_cls.__name__, counter, np.nanmean(n_plays), end - start)
        )
        counter += 1

    return job_results


def main() -> None:
    alphabet = string.ascii_lowercase
    guesses_all = list(itertools.product(alphabet, repeat=5))

    guess_idxs = np.random.choice(len(guesses_all), size=(24000))

    guesses = ["".join(guesses_all[idx]) for idx in guess_idxs]

    agent_list = [MaxInfoAgent, MaxSplitsAgent, MaxPruneAgent]

    kf = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

    results_per_agent = [
        job({"agent": agent, "splitter": kf}, guesses) for agent in agent_list
    ]

    total_results = list(itertools.chain.from_iterable(results_per_agent))

    cols = ["agent", "split_idx", "mean_plays", "time"]
    print(pd.DataFrame(total_results, columns=cols).groupby("agent").mean())


if __name__ == "__main__":
    raise SystemExit(main())
