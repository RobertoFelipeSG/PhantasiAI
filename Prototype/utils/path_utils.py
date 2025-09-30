import sys
from pathlib import Path

def add_parent_to_syspath(levels_up=1):
    """
    Add the parent (or ancestor) directory to sys.path to allow imports from sibling folders.

    Parameters:
    -----------
    levels_up : int
        How many levels up from the current file to go before adding to sys.path.
    """
    current = Path(__file__).resolve()
    target = current
    for _ in range(levels_up):
        target = target.parent

    if str(target) not in sys.path:
        sys.path.insert(0, str(target))
