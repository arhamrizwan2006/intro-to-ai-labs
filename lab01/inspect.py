"""
Lab 01 - Home Assignment
Introduction to Artificial Intelligence (AI-101L)

A small reusable function that loads a CSV (from a local path or a URL)
and prints one compact "first look" report: shape, dtypes, missing
values (count + percentage), and the numeric summary.
"""

import pandas as pd


def inspect(path_or_url):
    """Load a CSV and print a compact, standard first-look report.

    Parameters
    ----------
    path_or_url : str
        A local file path or an http(s) URL pointing at a CSV file.
    """
    df = pd.read_csv(path_or_url)

    print("=" * 70)
    print("SOURCE :", path_or_url)
    print("SHAPE  :", df.shape[0], "rows x", df.shape[1], "columns")
    print("=" * 70)

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print("None.")
    else:
        report = pd.DataFrame({
            "missing": miss,
            "percent": (miss / len(df) * 100).round(1),
        })
        print(report.sort_values("missing", ascending=False))

    print("\n--- NUMERIC SUMMARY ---")
    print(df.describe())

    return df


if __name__ == "__main__":
    # Dataset 1: Titanic passenger list (used throughout Lab 01)
    print("\n\n########## DATASET 1: TITANIC ##########\n")
    _ = inspect("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")

    # Dataset 2: Iris flower measurements (a second, different public dataset)
    print("\n\n########## DATASET 2: IRIS ##########\n")
    _ = inspect("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
