import pandas as pd

from preprocess import clean_text


DATA_PATH = "data/spam.csv"


def load_data():

    df = pd.read_csv(DATA_PATH)

    return df


if __name__ == "__main__":

    df = load_data()

    print("Original messages:")
    print(df["message"].head())

    print()
    print("-----------------------------")
    print()

    df["clean_message"] = df["message"].apply(
        clean_text
    )

    print("Cleaned messages:")
    print(df[["message", "clean_message"]].head())