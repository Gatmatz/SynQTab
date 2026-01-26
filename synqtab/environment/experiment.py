import os
from dotenv import load_dotenv

def _parse_comma_separated_integers(s) -> list[int]:
    """
    Converts a comma-separated string of integers into a list of ints.
    Example: "100,200,300..." -> [100, 200, 300, ...]
    """
    if s.strip() == "":
        return []
    return [int(x.strip()) for x in s.strip().split(',')]

def _parse_comma_separated_floats(s) -> list[float]:
    """
    Converts a comma-separated string of floating point numbers into a list of floats.
    Example: "0.1, 0.2, 0.4..." -> [0.1, 0.2, 0.4, ...]
    """
    if s.strip() == "":
        return []
    return [float(x.strip()) for x in s.strip().split(',')]

def _get_seeds_from_env_or_else_default() -> list[int]:
    seeds_str = os.getenv('RANDOM_SEEDS', '100,200,300')
    return _parse_comma_separated_integers(seeds_str)

def _get_pollution_rates_from_env_or_else_default() -> list[float]:
    pollution_rates_str = os.getenv('POLLUTION_RATES', '0.1, 0.2, 0.4')
    return _parse_comma_separated_floats(pollution_rates_str)


load_dotenv()
RANDOM_SEEDS = _get_seeds_from_env_or_else_default()
ERROR_RATES = _get_pollution_rates_from_env_or_else_default()
EXECUTION_PROFILE = os.getenv('EXECUTION_PROFILE', 'NOT FOUND IN ENV')
