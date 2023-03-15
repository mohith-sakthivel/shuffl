import os
import csv
import numpy as np


NAMES_FILE = "names.txt"
SHUFFL_STATUS_FILE = "status.txt"
MATCH_OUTPUT = "match.txt"
MATCH_OUTPUT_VERBOSE = "match_verbose.txt"

PAST_MATCHES_DIR = "past_matches"
GROUP_SIZE = 3


def read_data(file_path, lambda_fn=str):
    data = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            if len(row) == 1:
                data.append(lambda_fn(row[0]))
            else:
                row_data = []
                for i in row:
                    row_data.append(lambda_fn(i))
                data.append(row_data)

    return data


def write_data(file_path, data):
    with open(file_path, mode="w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerows(data)


def process_status(str_data):
    if str_data == "Y" or str_data == "y":
        return True
    elif str_data == "N" or str_data == "n":
        return False
    else:
        raise ValueError


def main():
    # Load names
    names = read_data(NAMES_FILE, lambda_fn=str)
    status = read_data(SHUFFL_STATUS_FILE, lambda_fn=process_status)
    match_history = np.eye(len(names), dtype=bool)

    ids = np.arange(len(names))
    available_mask = np.array(status, dtype=bool)

    # Read past data
    for item in os.listdir(PAST_MATCHES_DIR):
        if item[-4:] == ".txt":
            match_data = read_data(PAST_MATCHES_DIR + "/" + item, int)
            for match in match_data:
                for id1 in match:
                    for id2 in match:
                        match_history[id1, id2] = True

    # Create groupings for current week
    current_matches = []
    num_repeats = 0

    while np.any(available_mask):
        # Choose 1 available person
        anchor_id = np.random.choice(ids[available_mask])
        available_mask[anchor_id] = False
        # If no person is available for matching, add to previous group
        if not np.any(available_mask):
            current_matches[-1].append(anchor_id)
            break

        match_ids = [anchor_id]
        # Create a new mask of available and previously unmatched folks for this person
        curr_mask = np.logical_and(available_mask, ~match_history[anchor_id])
        while len(match_ids) < GROUP_SIZE:
            if np.any(curr_mask):
                # Choose an unmatched person
                new_id = np.random.choice(ids[curr_mask])
            else:
                # If an unmatched person is not available, do a repeat
                new_id = np.random.choice(ids[available_mask])
                num_repeats += 1

            available_mask[new_id] = False
            if new_id in match_ids:
                print("here")
            match_ids.append(new_id)
            # Update mask to use match history of currently selected person
            curr_mask = np.logical_and(curr_mask, ~match_history[new_id])

            if not np.any(available_mask):
                break

        current_matches.append(match_ids)

    write_data(MATCH_OUTPUT, current_matches)
    print("Matching done...")
    print(f"Matching has {num_repeats} groups with some repeats...")

    current_matches_verbose = []
    for row in current_matches:
        row_verbose = []
        for i in row:
            row_verbose.append(names[i])
        current_matches_verbose.append(row_verbose)

    write_data(MATCH_OUTPUT_VERBOSE, current_matches_verbose)


if __name__ == "__main__":
    main()
