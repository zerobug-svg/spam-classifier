import re


def clean_text(text: str) -> str:
    """
    Clean an SMS message before giving it to the ML model.
    """

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":

    message = "Congratulations! You WON ₹50,000!!!"

    cleaned = clean_text(message)

    print("Original:")
    print(message)

    print()

    print("Cleaned:")
    print(cleaned)