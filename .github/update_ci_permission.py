"""
Sort the CI permissions configuration file.

Usage:
    python3 .github/update_ci_permission.py --sort-only
"""

import argparse
import json
import os
import sys

FILE_NAME = os.path.join(os.path.dirname(__file__), "CI_PERMISSIONS.json")


def sort_permissions_file():
    if not os.path.exists(FILE_NAME):
        print(f"{FILE_NAME} not found. Nothing to sort.")
        return

    try:
        with open(FILE_NAME, "r") as f:
            permissions = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Error: {FILE_NAME} is invalid JSON: {exc}")
        sys.exit(1)

    sorted_permissions = dict(sorted(permissions.items()))

    with open(FILE_NAME, "w") as f:
        json.dump(sorted_permissions, f, indent=4)
        f.write("\n")

    print(f"Sorted {FILE_NAME}. Total users: {len(sorted_permissions)}")


def main():
    parser = argparse.ArgumentParser(description="Sort CI permissions.")
    parser.add_argument(
        "--sort-only",
        action="store_true",
        help="Sort CI_PERMISSIONS.json alphabetically.",
    )
    args = parser.parse_args()

    if not args.sort_only:
        parser.error("only --sort-only is supported")

    sort_permissions_file()


if __name__ == "__main__":
    main()
