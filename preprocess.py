
import pprint
from sklearn import datasets
import pandas as pandas
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

INPUT_FILE_NAME = "tweets.csv" # Input data sebelum preprocessing
OUTPUT_FILE_NAME = "cleaned_tweets.csv" # Input data setelah preprocessing

ROW_TWEET_TEXT = 0
ROW_LABEL = 2
tweets = pandas.read_csv(INPUT_FILE_NAME, header=None, skiprows=[0])[
    [ROW_TWEET_TEXT, ROW_LABEL]]
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(input_text: str) -> str:
    # Filtering, menghapus semua karakter non teks
    input_text = re.sub(r'\W', ' ', input_text)

    # Menghapus semua karakter tunggal pada bagian tengah teks
    input_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', input_text)

    # Menghapus semua karakter tunggal pada awal teks
    input_text = re.sub(r'\^[a-zA-Z]\s+', ' ', input_text)

    # Mengganti spasi berurutan dengan ' '
    input_text = re.sub(r'\s+', ' ', input_text, flags=re.I)

    # Case folding
    input_text = input_text.lower()

    # Stemming
    input_text = stemmer.stem(input_text)
    return input_text


tweets["cleaned"] = tweets[ROW_TWEET_TEXT].apply(clean_text)
tweets.to_csv(OUTPUT_FILE_NAME)
