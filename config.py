from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
LONG_TERM_WATCHLISTS_DIR = RESULTS_DIR / "long_term_watchlists"

LONG_TERM_SCREEN_VERSION = "lt_v1"


def ensure_results_dirs():
    for path in (RESULTS_DIR, LONG_TERM_WATCHLISTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
