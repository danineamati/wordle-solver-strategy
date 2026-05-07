from pathlib import Path


def repo_root() -> Path:
    """Repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def allowed_words_dir() -> Path:
    return repo_root() / "allowed_words"


def experiments_csv_dir() -> Path:
    return repo_root() / "experiments_csv"
