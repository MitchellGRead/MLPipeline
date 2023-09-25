import pandas as pd
from ray.data import Dataset
from sklearn.model_selection import train_test_split

from ml.api.data_handler_interface import DataHandlerInterface

from . import utils


class TagifaiDataHandler(DataHandlerInterface):
    def __init__(self, data_loc: str, num_samples: int = None) -> None:
        super().__init__(data_loc, num_samples)
        self.dataset = utils.load_csv_data(data_loc, num_samples)

    def get_data(self) -> Dataset:
        return self.dataset

    def split_data(
        self, test_size: float, shuffle: bool = True, seed: int = 1234
    ) -> tuple[Dataset, Dataset]:
        def _add_split(
            df: pd.DataFrame,
        ) -> pd.DataFrame:  # pragma: no cover, used in parent function
            """Naively split a dataframe into train and test splits.
            Add a column specifying whether it's the train or test split."""
            train, test = train_test_split(
                df, test_size=test_size, shuffle=shuffle, random_state=seed
            )
            train["_split"] = "train"
            test["_split"] = "test"
            return pd.concat([train, test])

        def _filter_split(
            df: pd.DataFrame, split: str
        ) -> pd.DataFrame:  # pragma: no cover, used in parent function
            """Filter by data points that match the split column's value
            and return the dataframe with the _split column dropped."""
            return df[df["_split"] == split].drop("_split", axis=1)

        # Train, test split with stratify
        stratify_on = "tag"
        grouped = self.dataset.groupby(stratify_on).map_groups(
            _add_split, batch_format="pandas"
        )  # group by each unique value in the column we want to stratify on
        train_ds = grouped.map_batches(
            _filter_split, fn_kwargs={"split": "train"}, batch_format="pandas"
        )  # combine
        test_ds = grouped.map_batches(
            _filter_split, fn_kwargs={"split": "test"}, batch_format="pandas"
        )  # combine

        # Shuffle each split (required)
        train_ds = train_ds.random_shuffle(seed=seed)
        test_ds = test_ds.random_shuffle(seed=seed)

        return train_ds, test_ds

    def add_to_config(self, train_loop_config: dict) -> dict:
        tags = self.dataset.unique("tag")
        train_loop_config["num_classes"] = len(tags)
        return train_loop_config
