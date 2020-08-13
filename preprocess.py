
import pprint
from sklearn import datasets
import pandas as pandas
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


INPUT_FILE_NAME = "tweets.csv"
OUTPUT_FILE_NAME = "cleaned_tweets.csv"

ROW_TWEET_TEXT = 0
ROW_LABEL = 2

tweets = pandas.read_csv(INPUT_FILE_NAME, header=None, skiprows=[0])[
    [ROW_TWEET_TEXT, ROW_LABEL]]


def clean_text(input_text: str) -> str:

    # Remove all the special characters
    input_text = re.sub(r'\W', ' ', input_text)

    # remove all single characters
    input_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', input_text)

    # Remove single characters from the start
    input_text = re.sub(r'\^[a-zA-Z]\s+', ' ', input_text)

    # Substituting multiple spaces with single space
    input_text = re.sub(r'\s+', ' ', input_text, flags=re.I)

    # Removing prefixed 'b'
    input_text = re.sub(r'^b\s+', '', input_text)

    # Fold case
    input_text = input_text.lower()

    return input_text


# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()


def stem_text(input_text: str) -> str:
    return stemmer.stem(input_text)


tweets["cleaned"] = tweets[ROW_TWEET_TEXT].apply(clean_text).apply(stem_text)

tweets.to_csv(OUTPUT_FILE_NAME)
