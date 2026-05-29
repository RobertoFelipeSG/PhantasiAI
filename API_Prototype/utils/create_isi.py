import json
import random
from collections import Counter
from pathlib import Path

# CONSTANTS
ISI_BINS = [0.30, 0.70, 1.10, 1.50]   # seconds
OCCURRENCES_PER_BIN = 10
NUM_GROUPS = 10
GROUP_SIZE = 40
TOTAL_INTERVALS = NUM_GROUPS * GROUP_SIZE 

def generate_intervals():
    ''' Create a pseudo-randomized list of 400 ISIs, with normal distribution of different intervals '''
    rng = random.Random(42) # ensures reproducibility if script is reran

    all_intervals = []
    seen_groups = set()

    # Create object of 400 ISIs
    for _ in range(NUM_GROUPS):
        group = ISI_BINS * OCCURRENCES_PER_BIN # Created group of 40 normalized intervals

        # Shuffle until order is unique 
        while True:
            rng.shuffle(group)
            group_tuple = tuple(group)
            if group_tuple not in seen_groups:
                seen_groups.add(group_tuple)
                break
        
        all_intervals.extend(group) # Add to global ISI array

    return all_intervals

if __name__ == "__main__":
    intervals = generate_intervals()

    # Save the Python list as a JSON array
    isi_path = Path(__file__).parent / "dynamic_intervals.json"
    with open(isi_path, 'w') as f:
        json.dump({"intervals": intervals}, f, indent=2)

    # Sanity check
    print(f"Total intervals generated : {len(intervals)}  (expected {TOTAL_INTERVALS})")
 
    counts = Counter(intervals)
    print("Per-group distribution:")
    for g in range(NUM_GROUPS):
        group_slice = intervals[g * GROUP_SIZE : (g + 1) * GROUP_SIZE]
        group_counts = Counter(group_slice)
        print(
            f"  Group {g + 1:02d}: "
            + "  ".join(f"{b:.2f}×{group_counts[b]}" for b in ISI_BINS)
        )