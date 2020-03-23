import collections
import pandas as pd

from argparse import Namespace
from pathlib import Path
from utils import *


if __name__ == '__main__':

    args = Namespace(
        raw_train_dataset_csv="train.csv",
        raw_test_dataset_csv="test.csv",
        train_proportion=0.7,
        val_proportion=0.15,
        test_proportion=0.15,
        output_munged_csv="tweets_with_splits_full.csv",
        seed=1337
    )

    # Read raw data
    train_reviews = pd.read_csv(Path().joinpath('data', args.raw_train_dataset_csv))[['target', 'text']]
    train_reviews = train_reviews[~pd.isnull(train_reviews.text)]
    train_reviews.columns = ['target', 'predictor']

    '''
        test_reviews = pd.read_csv(Path().joinpath('data', args.raw_test_dataset_csv))[['text']]
        test_reviews = test_reviews[~pd.isnull(test_reviews.text)]
        test_reviews['target'] = ""
        test_reviews = test_reviews[['target', 'text']]
        test_reviews.columns = ['target', 'predictor']
    '''

    # Splitting the subset by target to create our new train, val, and test splits
    by_rating = collections.defaultdict(list)
    for _, row in train_reviews.iterrows():
        by_rating[row.target].append(row.to_dict())

    # Unique classes
    print(set(train_reviews.target))

    # Create split data
    final_list = []
    np.random.seed(args.seed)

    for _, item_list in sorted(by_rating.items()):

        np.random.shuffle(item_list)

        n_total = len(item_list)
        n_train = int(args.train_proportion * n_total)
        n_val = int(args.val_proportion * n_total)
        n_test = int(args.test_proportion * n_total)

        # Give data point a split attribute
        for item in item_list[:n_train]:
            item['split'] = 'train'

        for item in item_list[n_train:n_train + n_val]:
            item['split'] = 'val'

        for item in item_list[n_train + n_val:n_train + n_val + n_test]:
            item['split'] = 'test'

        # Add to final list
        final_list.extend(item_list)

    # Write split data to file
    final_predictors = pd.DataFrame(final_list)

    for i in final_predictors[pd.isnull(final_predictors.split)].index:
        final_predictors.loc[i, 'split'] = final_predictors.loc[i-1, 'split']

    final_predictors.predictor = final_predictors.predictor.apply(preprocess_text)

    final_predictors['target'] = final_predictors.target.apply({0: 'fake', 1: 'real'}.get)

    # final_predictors.to_csv(args.output_munged_csv, index=False)
    final_predictors.to_csv(Path().joinpath('data', args.output_munged_csv), index=False)