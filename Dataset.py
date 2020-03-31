import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from Vectorizer import ReviewVectorizer
from pathlib import Path


class ReviewDataset(Dataset):
    def __init__(self, predictor_df, vectorizer):
        """
        Args:
            predictor_df (pandas.DataFrame): the dataset
            vectorizer (ReviewVectorizer): vectorizer instantiated from dataset
        """
        self.predictor_df = predictor_df
        self._vectorizer = vectorizer

        self.train_df = self.predictor_df[self.predictor_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.predictor_df[self.predictor_df.split == 'val']
        self.validation_size = len(self.val_df)

        self.test_df = self.predictor_df[self.predictor_df.split == 'test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

    @classmethod
    def load_dataset_and_make_vectorizer(cls, predictor_csv):
        """Load dataset and make a new vectorizer from scratch

        Args:
            predictor_csv (str): location of the dataset
        Returns:
            an instance of ReviewDataset
        """
        predictor_df = pd.read_csv(Path().joinpath('data', predictor_csv))
        train_predictor_df = predictor_df[predictor_df.split == 'train']
        return cls(predictor_df, ReviewVectorizer.from_dataframe(train_predictor_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, predictor_csv, vectorizer_filepath):
        """Load dataset and the corresponding vectorizer.
        Used in the case in the vectorizer has been cached for re-use

        Args:
            predictor_csv (str): location of the dataset
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of ReviewDataset
        """
        predictor_df = pd.read_csv(Path().joinpath('data', predictor_csv))
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(predictor_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """a static method for loading the vectorizer from file

        Args:
            vectorizer_filepath (str): the location of the serialized vectorizer
        Returns:
            an instance of ReviewVectorizer
        """
        with open(vectorizer_filepath) as fp:
            return ReviewVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """saves the vectorizer to disk using json

        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe

        Args:
            split (str): one of "train", "val", or "test"
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        predictor_vector = \
            self._vectorizer.vectorize(row.predictor)

        target_index = \
            self._vectorizer.target_vocab.lookup_token(row.target)

        return {'x_data': predictor_vector,
                'y_target': target_index}

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size