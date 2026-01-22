import os
from dotenv import load_dotenv


load_dotenv()

def _parse_comma_separated_integers(s):
    """
    Converts a comma-separated string of integers into a list of ints.
    Example: "100,200,300..." -> [100, 200, 300, ...]
    """
    if s.strip() == "":
        return []
    return [int(x.strip()) for x in s.strip().split(',')]

def _get_seeds_from_env_or_else_default() -> list[int]:
    seeds_str = os.getenv('RANDOM_SEEDS', '100,200,300')
    return _parse_comma_separated_integers(seeds_str)

RANDOM_SEEDS = _get_seeds_from_env_or_else_default()
