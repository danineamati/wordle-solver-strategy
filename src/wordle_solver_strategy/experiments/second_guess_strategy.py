import argparse
import time

import numpy as np
import pandas as pd
from numba import jit

from wordle_solver_strategy import (
    code_to_emoji,
    entropy,
    get_bin_table_inline,
    get_numeric_representations,
)
from wordle_solver_strategy.paths import allowed_words_dir, experiments_csv_dir


N_CODES = 1024


@jit(nopython=True, nogil=True, cache=True)
def subset_bin_counts(bin_table, answer_idxs, candidate_guess_idxs):
    counts = np.zeros((N_CODES, candidate_guess_idxs.shape[0]), dtype=np.intc)
    for candidate_idx in range(candidate_guess_idxs.shape[0]):
        guess_idx = candidate_guess_idxs[candidate_idx]
        for subset_answer_idx in range(answer_idxs.shape[0]):
            answer_idx = answer_idxs[subset_answer_idx]
            code = bin_table[answer_idx, guess_idx]
            counts[code, candidate_idx] += 1
    return counts


def code_int_to_string(code_int):
    chars = ["0", "0", "0", "0", "0"]
    for letter_idx in range(5):
        chars[letter_idx] = str((code_int >> (letter_idx * 2)) & 0b11)
    return "".join(chars)


def load_word_lists():
    answers_path = allowed_words_dir() / "words_answers.txt"
    guesses_path = allowed_words_dir() / "words_guesses.txt"
    with open(answers_path, "r") as answers_file:
        answers = answers_file.read().splitlines()
    with open(guesses_path, "r") as guesses_file:
        guesses = guesses_file.read().splitlines()
    return answers, guesses


def resolve_candidate_indices(guesses, top_n_candidates, candidate_words=None):
    if candidate_words:
        guess_to_idx = {word: idx for idx, word in enumerate(guesses)}
        normalized_words = []
        seen = set()
        for word in candidate_words:
            normalized_word = word.lower().strip()
            if normalized_word not in seen:
                normalized_words.append(normalized_word)
                seen.add(normalized_word)

        invalid_words = [word for word in normalized_words if word not in guess_to_idx]
        if invalid_words:
            raise ValueError(
                "candidate_words contains unknown guesses: "
                + ", ".join(invalid_words)
            )

        return np.array([guess_to_idx[word] for word in normalized_words], dtype=np.intc)

    if top_n_candidates is None:
        return np.arange(len(guesses), dtype=np.intc)
    top_n = max(1, min(int(top_n_candidates), len(guesses)))
    return np.arange(top_n, dtype=np.intc)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a capped second-guess strategy for a fixed first word "
            "using information-theoretic scoring."
        )
    )
    parser.add_argument("first_word", type=str, help="First guessed word.")
    parser.add_argument(
        "--max-num-second-word-in-strategy",
        type=int,
        default=None,
        help="Maximum number of second guesses allowed in the strategy. Defaults to 1.",
    )
    parser.add_argument(
        "--top-n-candidates",
        type=int,
        default=None,
        help=(
            "Limit candidate second guesses to first N words in guess list for speed. "
            "Defaults to all guesses."
        ),
    )
    parser.add_argument(
        "--candidate-words",
        nargs="+",
        default=None,
        help=(
            "Explicit list of candidate second guesses. "
            "When provided, this list is used instead of --top-n-candidates."
        ),
    )
    return parser.parse_args()


def build_bin_table(answers, guesses):
    guesses_numba, _ = get_numeric_representations(guesses)
    answers_numba, answers_char_counts = get_numeric_representations(answers)
    return get_bin_table_inline(guesses_numba, answers_numba, answers_char_counts)


def compute_utility_by_first_code(bin_table, first_guess_idx, answers, candidate_guess_idxs):
    first_codes = bin_table[:, first_guess_idx].astype(np.intc)
    active_codes, code_sizes = np.unique(first_codes, return_counts=True)
    code_probabilities = code_sizes / len(answers)

    utility_by_code = np.zeros((active_codes.shape[0], candidate_guess_idxs.shape[0]))
    for active_idx, code in enumerate(active_codes):
        answer_idxs = np.nonzero(first_codes == code)[0].astype(np.intc)
        counts = subset_bin_counts(bin_table, answer_idxs, candidate_guess_idxs)
        utility_by_code[active_idx, :] = entropy(counts)

    return active_codes, code_sizes, code_probabilities, utility_by_code


def select_second_guess_set(utility_by_code, code_probabilities, k):
    selected_candidate_positions = []
    used = np.zeros(utility_by_code.shape[1], dtype=bool)
    current_best = np.zeros(utility_by_code.shape[0])

    for _ in range(k):
        improved = np.maximum(current_best[:, None], utility_by_code)
        gains = np.sum((improved - current_best[:, None]) * code_probabilities[:, None], axis=0)
        gains[used] = -np.inf
        best_position = int(np.argmax(gains))
        if not np.isfinite(gains[best_position]):
            break
        selected_candidate_positions.append(best_position)
        used[best_position] = True
        current_best = np.maximum(current_best, utility_by_code[:, best_position])

    if len(selected_candidate_positions) == 0:
        raise RuntimeError("No second guesses selected. Check candidate pool.")

    return np.array(selected_candidate_positions, dtype=np.intc)


def build_output_rows(
    first_word,
    k,
    guesses,
    candidate_guess_idxs,
    active_codes,
    code_sizes,
    code_probabilities,
    utility_by_code,
    selected_candidate_positions,
):
    assignment_in_selected = np.argmax(
        utility_by_code[:, selected_candidate_positions], axis=1
    )

    rows = []
    for active_idx, code in enumerate(active_codes):
        assigned_local_idx = assignment_in_selected[active_idx]
        assigned_guess_position = selected_candidate_positions[assigned_local_idx]
        assigned_guess_idx = candidate_guess_idxs[assigned_guess_position]
        assigned_guess = guesses[assigned_guess_idx]
        assigned_score = utility_by_code[active_idx, assigned_guess_position]

        selected_guess_words = [
            guesses[candidate_guess_idxs[idx]] for idx in selected_candidate_positions
        ]
        unique_selected_guess_words = list(set(selected_guess_words))

        code_string = code_int_to_string(int(code))
        rows.append(
            {
                "first_word": first_word,
                "first_code_emoji": code_to_emoji(code_string),
                "first_code": code_string,
                "max_num_second_word_in_strategy": k,
                "assigned_second_guess": assigned_guess,
                "is_bin_solved": bool(code_sizes[active_idx] == 1),
                "n_answers_in_bin": int(code_sizes[active_idx]),
                "bin_probability": float(code_probabilities[active_idx]),
                "assigned_score_entropy": float(assigned_score),
                "selected_second_guess_list": ",".join(unique_selected_guess_words)
            }
        )
    return rows, assignment_in_selected


def sort_output_df(output_df):
    sort_columns = []
    ascending = []
    if "n_answers_in_bin" in output_df.columns:
        sort_columns.append("n_answers_in_bin")
        ascending.append(False)
    if "first_code_int" in output_df.columns:
        sort_columns.append("first_code_int")
        ascending.append(True)
    elif "first_code" in output_df.columns:
        sort_columns.append("first_code")
        ascending.append(True)

    if sort_columns:
        return output_df.sort_values(sort_columns, ascending=ascending)
    return output_df


def print_selected_guess_usage(
    guesses,
    candidate_guess_idxs,
    selected_candidate_positions,
    assignment_in_selected,
    code_sizes,
    code_probabilities,
):
    print("Selected second-guess usage by first-code bin:")
    for selected_pos_idx, selected_pos in enumerate(selected_candidate_positions):
        guess_idx = candidate_guess_idxs[selected_pos]
        guess_word = guesses[guess_idx]
        matched_bins = assignment_in_selected == selected_pos_idx
        n_bins = int(np.sum(matched_bins))
        n_answers = int(np.sum(code_sizes[matched_bins]))
        probability_mass = float(np.sum(code_probabilities[matched_bins]))
        print(
            f"  {guess_word}: bins={n_bins}, answers={n_answers}, "
            f"probability_mass={probability_mass:.4f}"
        )


def main():
    args = parse_args()

    # 1) Parse and validate user inputs.
    first_word = args.first_word.lower().strip()
    k = 1 if args.max_num_second_word_in_strategy is None else int(
        args.max_num_second_word_in_strategy
    )
    if k < 1:
        raise ValueError("max_num_second_word_in_strategy must be >= 1")

    timings = {}
    phase_start = time.time()
    answers, guesses = load_word_lists()
    timings["load_words"] = time.time() - phase_start

    if first_word not in guesses:
        raise ValueError(f"first_word '{first_word}' is not in guess word list.")

    candidate_guess_idxs = resolve_candidate_indices(
        guesses, args.top_n_candidates, args.candidate_words
    )
    k = min(k, candidate_guess_idxs.shape[0])

    # 2) Build global feedback table once (numba hot path).
    phase_start = time.time()
    bin_table = build_bin_table(answers, guesses)
    timings["build_bin_table"] = time.time() - phase_start

    # 3) Score all second-guess candidates for each first-feedback bin.
    first_guess_idx = guesses.index(first_word)
    phase_start = time.time()
    active_codes, code_sizes, code_probabilities, utility_by_code = (
        compute_utility_by_first_code(
            bin_table, first_guess_idx, answers, candidate_guess_idxs
        )
    )
    timings["score_candidates"] = time.time() - phase_start

    # 4) Greedily select the capped second-guess strategy.
    phase_start = time.time()
    selected_candidate_positions = select_second_guess_set(
        utility_by_code, code_probabilities, k
    )
    selected_guess_idxs = candidate_guess_idxs[selected_candidate_positions]
    selected_guess_words = [guesses[idx] for idx in selected_guess_idxs]
    timings["select_strategy"] = time.time() - phase_start

    # 5) Build and save row-level CSV output.
    rows, assignment_in_selected = build_output_rows(
        first_word,
        k,
        guesses,
        candidate_guess_idxs,
        active_codes,
        code_sizes,
        code_probabilities,
        utility_by_code,
        selected_candidate_positions,
    )
    out_path = (
        experiments_csv_dir()
        / f"second_guess_strategy_for_{first_word}_max_{k}.csv"
    )
    output_df = sort_output_df(pd.DataFrame(rows))
    output_df.to_csv(out_path, index=False, float_format="%.4f")

    timings["write_csv"] = time.time() - phase_start

    print(f"Saved strategy to: {out_path}")
    print(f"Selected second guesses ({len(selected_guess_words)}): {selected_guess_words}")
    print_selected_guess_usage(
        guesses,
        candidate_guess_idxs,
        selected_candidate_positions,
        assignment_in_selected,
        code_sizes,
        code_probabilities,
    )
    print("Timings (seconds):")
    for phase_name, duration in timings.items():
        print(f"  {phase_name}: {duration:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
