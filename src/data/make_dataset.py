# -*- coding: utf-8 -*-
import gzip
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
#import wget
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import wget
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
2
def parse(path):
    g = gzip.open(path, "rb")
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


def to_sentiment(rating):
    rating = int(rating)
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2


class_names = ["negative", "neutral", "positive"]


class AmazonData(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_length):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        target = self.targets[idx]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }


@click.command()
@click.argument("input_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
<<<<<<< HEAD
    logger.info('making final data set from raw data')
    '''if 'reviews_Amazon_Instant_Video_5.json.gz' not in os.listdir(input_filepath):
        print('Raw data folder appears to be empty. Downloading the data to raw data folder.')
        url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz'
        filepath = wget.download(url, out=input_filepath)
        print(filepath, 'Download finished!')'''
    df = getDF(input_filepath + '/reviews_Amazon_Instant_Video_5.json.gz')
    data = df['reviewText'].to_numpy()
    labels = df['overall'].apply(to_sentiment).to_list()
    X_train, X_test, Y_train, Y_test = train_test_split(data,labels, train_size=0.75, test_size=0.25, random_state=42, shuffle=True)
    np.savez(input_filepath + '/../interim/train.npz', x=X_train, y=Y_train)
    np.savez(input_filepath + '/../interim/test.npz', x=X_test, y=Y_test)
    tokenizer = tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
=======
    logger.info("making final data set from raw data")
    if "reviews_Amazon_Instant_Video_5.json.gz" not in os.listdir(input_filepath):
        print(
            "Raw data folder appears to be empty. Downloading the data to raw data folder."
        )
        url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz"
        filepath = wget.download(url, out=input_filepath)
        print(filepath, "Download finished!")
    df = getDF(input_filepath + "/reviews_Amazon_Instant_Video_5.json.gz")
    data = df["reviewText"].to_numpy()
    labels = df["overall"].apply(to_sentiment).to_list()
    X_train, X_test, Y_train, Y_test = train_test_split(
        data, labels, train_size=0.75, test_size=0.25, random_state=42, shuffle=True
    )
    np.savez(input_filepath + "/../interim/train.npz", x=X_train, y=Y_train)
    np.savez(input_filepath + "/../interim/test.npz", x=X_test, y=Y_test)
    tokenizer = tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
>>>>>>> 5e1453eb872e5d745356f8a91b1c6aa74027ef72
    train_data = AmazonData(
        reviews=X_train,
        targets=Y_train,
        tokenizer=tokenizer,
        max_length=20,
    )
    test_data = AmazonData(
        reviews=X_test,
        targets=Y_test,
        tokenizer=tokenizer,
        max_length=20,
    )
    torch.save(train_data, output_filepath + "/train.pth")
    torch.save(test_data, output_filepath + "/test.pth")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
