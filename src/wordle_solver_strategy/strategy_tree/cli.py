"""Console entry: fit a shallow tree on a second-guess strategy CSV and print rules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wordle_solver_strategy.paths import experiments_csv_dir

from .features import build_feature_matrix, load_strategy_csv
from .humanize import humanize_decision_tree_export
from .tree import fit_second_guess_tree, tree_rules_as_text, weighted_accuracy


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Fit a weighted decision tree that approximates assigned_second_guess "
            "from first_code features (see second-guess strategy CSV)."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "Path to second_guess_strategy_for_* CSV. "
            "If omitted, uses experiments_csv/second_guess_strategy_for_{first}_max_{k}.csv"
        ),
    )
    parser.add_argument(
        "--first-word",
        type=str,
        default="crane",
        help="First guess word (used to build default CSV path). Default: crane",
    )
    parser.add_argument(
        "--max-num-second-word-in-strategy",
        type=int,
        default=8,
        help="k in default CSV filename. Default: 8",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum depth of the decision tree. Default: 4",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random state for the tree. Default: 0",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples per leaf. Default: 1",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help=(
            "Write the same report that is printed to stdout into "
            "experiments_csv/ next to the strategy CSVs "
            "(see wordle_solver_strategy.paths.experiments_csv_dir)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory to save the report as a .txt file (created if missing). "
            "Overrides --save. Typical: experiments_csv at the repository root."
        ),
    )
    parser.add_argument(
        "--human-readable",
        action="store_true",
        help=(
            "Rewrite tree split lines in plain English (e.g. 'e_tile > 0.50' -> "
            "'E is yellow or green'). Applies to stdout and saved reports."
        ),
    )
    return parser.parse_args(argv)


def default_csv_path(first_word: str, k: int) -> Path:
    fw = first_word.lower().strip()
    return experiments_csv_dir() / f"second_guess_strategy_for_{fw}_max_{k}.csv"


def interpretation_guide(first_word: str, *, human_readable: bool = False) -> str:
    fw = first_word.lower().strip()
    letters = [c for c in fw]
    letter_slots = ", ".join(
        f"'{letters[i]}' -> {letters[i]}_tile (position {i + 1})"
        for i in range(5)
    )
    hr_note = ""
    if human_readable:
        hr_note = (
            "\nThe exported decision tree uses plain-English split labels instead of "
            "feature names and numeric thresholds (--human-readable).\n"
        )
    return f"""How to interpret this tree
-------------------------
Context: The first guess is fixed as "{fw}". Each row in the training data is one possible
Wordle feedback pattern for that guess (one row per `first_code` in the CSV).

Per-position feedback (ordinal features):
  The first guess has five letters. Each slot uses one feature named <letter>_tile with a
  numeric code for that slot's tile:
    0 = ⬛ absent (gray; letter not in the solution, or extra copy not placed)
    1 = 🟨 present elsewhere (yellow)
    2 = 🟩 correct spot (green)

  Letter to feature for this run: {letter_slots}

Count features (integers 0..5, sum to 5):
  n_green   — how many positions are green (2)
  n_yellow  — how many are yellow (1)
  n_absent  — how many are gray (0)

How sklearn splits (thresholds like "e_tile <= 0.50"):
  The tree tests a numeric feature against a threshold. For tile codes 0/1/2:
    "<= 0.5"  -> value is 0 (⬛ gray) vs 🟨/🟩
    "<= 1.5"  -> values 0 or 1 (⬛/🟨) vs 🟩 (2)
  For n_green / n_yellow / n_absent, splits such as "<= 3.5" sit halfway between whole counts.

Leaves: "class: <word>" is the second guess the tree assigns for all patterns that reach
that leaf. It approximates the CSV's `assigned_second_guess`; with a small depth, multiple
patterns share one leaf so the label is the majority / weighted choice from the solver.
{hr_note}"""


def report_text(
    csv_path: Path,
    first_word: str,
    n_rows: int,
    n_features: int,
    acc: float,
    max_depth: int,
    n_leaves: int,
    rules: str,
    *,
    human_readable: bool = False,
) -> str:
    tree_title = "Decision tree (sklearn export)"
    if human_readable:
        tree_title = "Decision tree (plain-English splits; see --human-readable)"

    lines = [
        f"CSV: {csv_path.resolve()}",
        f"Rows: {n_rows}  Features: {n_features}",
        (
            "Weights: normalized bin_probability (sum=1); "
            "accuracy is expected match rate vs oracle labels on this table."
        ),
        f"Weighted accuracy (resubstitution): {acc:.4f}",
        f"Tree depth cap: {max_depth}  Leaves: {n_leaves}",
        "",
        interpretation_guide(first_word, human_readable=human_readable).rstrip(),
        "",
        tree_title,
        "----------------------------",
        rules.rstrip() + "\n",
    ]
    return "\n".join(lines)


def output_report_path(
    csv_path: Path, max_depth: int, seed: int, *, human_readable: bool = False
) -> str:
    stem = csv_path.stem
    if seed != 0:
        core = f"{stem}_decision_tree_depth_{max_depth}_seed_{seed}"
    else:
        core = f"{stem}_decision_tree_depth_{max_depth}"
    if human_readable:
        core += "_human"
    return f"{core}.txt"


def resolve_output_dir(args) -> Path | None:
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser().resolve()
    if args.save:
        return experiments_csv_dir()
    return None


def print_report(text: str) -> None:
    """Print report text, handling terminals that cannot encode emoji."""
    try:
        print(text, end="")
    except UnicodeEncodeError:
        # Keep UTF-8 on saved files; degrade stdout rendering only when needed.
        encoding = sys.stdout.encoding or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding)
        print(safe_text, end="")
        print(
            "Note: terminal encoding cannot render some emoji; saved report keeps them.",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    k = args.max_num_second_word_in_strategy
    if k < 1:
        raise SystemExit("--max-num-second-word-in-strategy must be >= 1")

    csv_path = args.csv
    if csv_path is None:
        csv_path = default_csv_path(args.first_word, k)
    else:
        csv_path = Path(csv_path)

    df = load_strategy_csv(csv_path)
    x, feature_names, sample_weight, y_labels = build_feature_matrix(df)

    model, encoder = fit_second_guess_tree(
        x,
        y_labels,
        sample_weight=sample_weight,
        max_depth=args.max_depth,
        random_state=args.seed,
        min_samples_leaf=args.min_samples_leaf,
    )

    y_pred_enc = model.predict(x)
    y_pred = encoder.inverse_transform(y_pred_enc)

    acc = weighted_accuracy(y_labels, y_pred, sample_weight)
    n_leaves = model.get_n_leaves()

    class_names = [str(c) for c in encoder.classes_]
    rules = tree_rules_as_text(
        model,
        feature_names,
        class_names=class_names,
    )
    if args.human_readable:
        rules = humanize_decision_tree_export(rules)

    first_word = str(df["first_word"].iloc[0]).lower().strip()
    text = report_text(
        csv_path,
        first_word,
        len(df),
        len(feature_names),
        acc,
        args.max_depth,
        n_leaves,
        rules,
        human_readable=args.human_readable,
    )
    print_report(text)
    sys.stdout.flush()

    out_dir = resolve_output_dir(args)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = output_report_path(
            csv_path, args.max_depth, args.seed, human_readable=args.human_readable
        )
        out_path = out_dir / out_name
        out_path.write_text(text, encoding="utf-8")
        print(f"Saved: {out_path.resolve()}", file=sys.stderr)
        sys.stderr.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
