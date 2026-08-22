import json
import os


# Add new metric files here as they arrive
METRICS_FILES = [
    "all_case_metrics_raw.json",
    "all_case_metrics_raw2.json",
]


def load_all_metrics():
    """
    Load and combine all BraTS metric JSON files.

    Each JSON file is expected to have the format:

    {
        "results": [
            {
                "case_id": "...",
                "metrics": {...}
            }
        ]
    }

    Returns:
        list: Combined list of case records.
    """

    all_cases = {}

    for filename in METRICS_FILES:

        # Make path relative to this Python file
        filepath = os.path.join(
            os.path.dirname(__file__),
            filename
        )

        if not os.path.exists(filepath):
            print(f"WARNING: File not found: {filename}")
            continue

        print(f"Loading: {filename}")

        with open(filepath, "r") as f:
            data = json.load(f)

        # New JSON format
        if isinstance(data, dict) and "results" in data:
            cases = data["results"]

        # Support old format too, just in case
        elif isinstance(data, list):
            cases = data

        else:
            print(f"WARNING: Unknown JSON format in {filename}")
            continue

        print(f"  Cases loaded: {len(cases)}")

        for case in cases:

            case_id = case.get("case_id")

            if not case_id:
                print("WARNING: Case without case_id skipped")
                continue

            # Store by case_id to avoid duplicates
            all_cases[case_id] = case

    combined_cases = list(all_cases.values())

    print("-----------------------------------")
    print(f"Total unique cases loaded: {len(combined_cases)}")
    print("-----------------------------------")

    return combined_cases