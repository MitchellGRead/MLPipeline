import json
import re
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import ray
from nltk.stem import PorterStemmer as stemmer
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from config import config
from config.config import logger


def load_data(dataset_loc: str, num_samples: int = None) -> Dataset:
    """Load data from source into a Ray Dataset

    Args:
        dataset_loc (str): Location of the dataset
        num_samples (int, optional): number of samples to load. Defaults to None (use all data).

    Returns:
        Dataset: Data represented as a Ray Dataset
    """
    data = ray.data.read_csv(dataset_loc)
    data = data.random_shuffle(seed=1234)
    data = ray.data.from_items(data.take(num_samples)) if num_samples else data
    return data


def stratify_split(
    data: Dataset, stratify: str, test_size: float, shuffle: bool = True, seed: int = 1234
) -> tuple[Dataset, Dataset]:
    """Split a dataset into train and test splits with equal amounts of data points from each class in the column we want to stratify on

    Args:
        data (Dataset): Input Ray Dataset to split
        stratify (str): Name of column to split on
        test_size (float): Proportion of dataset to split for test set
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        seed (int, optional): Seed for shuffling. Defaults to 1234.

    Returns:
        typle[Dataset, Dataset]: The stratified train and test Ray Datasets
    """

    def _add_split(df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover, used in parent function
        """Naively split a dataframe into train and test splits. Add a column specifying whether is is a train or test split"""
        train, test = train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=seed)
        train["_split"] = "train"
        test["_split"] = "test"
        return pd.concat([train, test])

    def _filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
        """Filter by data points that match the split column's value and return the dataframe with the _split column dropped"""
        return df[df["_split"] == split].drop("_split", axis=1)

    # Train, test split with stratify
    grouped = data.groupby(stratify).map_groups(
        _add_split, batch_format="pandas"
    )  # group by each unique value in the column we want to stratify on
    train_data = grouped.map_batches(
        _filter_split, fn_kwargs={"split": "train"}, batch_format="pandas"
    )
    test_data = grouped.map_batches(
        _filter_split, fn_kwargs={"split": "test"}, batch_format="pandas"
    )

    # Shuffle each split
    train_data = train_data.random_shuffle(seed=seed)
    test_data = test_data.random_shuffle(seed=seed)

    return train_data, test_data


def preprocess_data(df, lower, stem, min_freq):
    """Preprocess the data."""
    logger.info(f"Preprocessing data with lower={lower} - stem={stem} - min_freq={min_freq}")

    df["text"] = df.title + " " + df.description  # feature engineering
    df.text = df.text.apply(clean_text, lower=lower, stem=stem)  # clean text
    df = replace_oos_labels(
        df=df, labels=config.ACCEPTED_TAGS, label_col="tag", oos_label="other"
    )  # replace OOS labels
    df = replace_minority_labels(
        df=df, label_col="tag", min_freq=min_freq, new_label="other"
    )  # replace labels below min freq

    return df


def replace_oos_labels(df, labels, label_col, oos_label="other"):
    """Replace out of scope (oos) labels."""
    oos_tags = [item for item in df[label_col].unique() if item not in labels]
    df[label_col] = df[label_col].apply(lambda x: oos_label if x in oos_tags else x)
    return df


def replace_minority_labels(df, label_col, min_freq, new_label="other"):
    """Replace minority labels with another label."""
    labels = Counter(df[label_col].values)
    labels_above_freq = Counter(label for label in labels.elements() if (labels[label] >= min_freq))
    df[label_col] = df[label_col].apply(lambda label: label if label in labels_above_freq else None)
    df[label_col] = df[label_col].fillna(new_label)
    return df


def get_data_splits(X, y, train_size=0.7):
    """Generate balanced data splits."""
    logger.info(f"Generating test splits with train_size={train_size}")
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess(df: pd.DataFrame, class_to_index: dict) -> dict:
    """Preprocess the data of our Pandas dataframe

    Args:
        df (pd.DataFrame): Raw dataframe to preprocess
        class_to_index (dict): Mapping of class names to indices

    Returns:
        dict: Preprocessed data (ids, masks, targets)
    """
    df["text"] = f"{df.title} {df.description}"  # feature engineering
    df["text"] = df.text.apply(clean_text)  # data cleaning
    df = df.drop(
        columns=["id", "created_on", "title", "description"], errors="ignore"
    )  # clean dataframe
    df = df[["text", "tag"]]  # rearrange columns
    df["tag"] = df["tag"].map(class_to_index)  # label encoding
    outputs = tokenize(df)
    return outputs


def tokenize(batch: dict) -> dict:
    """Tokenize the text input in our batch using a tokenizer

    Args:
        batch (dict): batch of data with the text inputs to tokenize

    Returns:
        dict: batch of data with the results tokenized (`input_ids` and `attention_mask`) on the text inputs.
    """
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    encoded_inputs = tokenizer(batch["text"].tolist(), return_tensors="np", padding="longest")
    return dict(
        ids=encoded_inputs["input_ids"],
        masks=encoded_inputs["attention_mask"],
        targets=np.array(batch["tag"]),
    )


def clean_text(
    text: str, lower: bool = True, stem: bool = False, stopwords: List = config.STOPWORDS
):
    """Clean raw text string

    Args:
        text (str): Text to be cleaned
        lower (bool, optional): Whether text should be lower cased. Defaults to True.
        stem (bool, optional): Whether we want to stem text. Defaults to False.
        stopwords (List, optional): List of words to filter out. Defaults to config.STOPWORDS.

    Returns:
        str: cleaned text
    """
    # Lower
    if lower:
        text = text.lower()

    # Remove stopwords
    if len(stopwords):
        pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
        text = pattern.sub("", text)

    # Spacing and filters
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)  # add spacing
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends
    text = re.sub(r"http\S+", "", text)  # Remove links

    # Stemming
    if stem:
        text = " ".join([stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")])

    return text


class CustomPreprocessor(Preprocessor):
    def _fit(self, data: Dataset):
        tags = data.unique(column="tag")
        self.class_to_index = {tag: i for i, tag in enumerate(tags)}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}

    def _transform_pandas(self, batch):
        return preprocess(batch, class_to_index=self.class_to_index)


class LabelEncoder(object):
    """Encode labels into unique indices."""

    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index or {}  # mutable defaults ;)
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y):
        classes = np.unique(y)
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y):
        encoded = np.zeros((len(y)), dtype=int)
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded

    def decode(self, y):
        classes = []
        for i, item in enumerate(y):
            classes.append(self.index_to_class[item])
        return classes

    def save(self, fp):
        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)
