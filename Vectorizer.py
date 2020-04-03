import numpy as np
import string
from utils import *
from Vocabulary import Vocabulary
from collections import Counter


class ReviewVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""

    def __init__(self, predictor_vocab, target_vocab, max_predictor_length=0):
        """
        Args:
            predictor_vocab (Vocabulary): maps words to integers
            target_vocab (Vocabulary): maps class labels to integers
        """
        self.predictor_vocab = predictor_vocab
        self.target_vocab = target_vocab
        self._max_predictor_length = max_predictor_length
    '''
    def vectorize(self, predictor):
        """Create a collapsed one-hit vector for the predictor

        Args:
            predictor (str): the predictor
        Returns:
            one_hot (np.ndarray): the collapsed one-hot encoding
        """
        one_hot_matrix = np.zeros(len(self.predictor_vocab), dtype=np.float32)

        for token in predictor.split(" "):
            if token not in string.punctuation:
                one_hot_matrix[self.predictor_vocab.lookup_token(token)] = 1

        return one_hot_matrix
    '''
    def vectorize(self, predictor, classifier_class):
        if classifier_class == 'CNN':
            one_hot_matrix_size = (len(self.predictor_vocab), self._max_predictor_length)
            one_hot_matrix = np.zeros(one_hot_matrix_size, dtype=np.float32)

            for position_index, word in enumerate(remove_punctuation(predictor)):
                word_index = self.predictor_vocab.lookup_token(word)
                one_hot_matrix[word_index][position_index] = 1
        else:
            one_hot_matrix = np.zeros(len(self.predictor_vocab), dtype=np.float32)

            for token in predictor.split(" "):
                if token not in string.punctuation:
                    one_hot_matrix[self.predictor_vocab.lookup_token(token)] = 1

        return one_hot_matrix


    @classmethod
    def from_dataframe(cls, predictor_df, cutoff=25):
        """Instantiate the vectorizer from the dataset dataframe

        Args:
            predictor_df (pandas.DataFrame): the predictor dataset
            cutoff (int): the parameter for frequency-based filtering
        Returns:
            an instance of the ReviewVectorizer
        """
        predictor_vocab = Vocabulary(add_unk=True)
        target_vocab = Vocabulary(add_unk=False)
        max_predictor_length = 0

        # Add targets
        for target in sorted(set(predictor_df.target)):
            target_vocab.add_token(target)

        # Add top words if count > provided count
        word_counts = Counter()
        for index, row in predictor_df.iterrows():
            vector = remove_punctuation(row.predictor)
            max_predictor_length = max(max_predictor_length, len(vector))
            for word in vector:
                word_counts[word] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                predictor_vocab.add_token(word)

        return cls(predictor_vocab, target_vocab, max_predictor_length) # for CNN

    @classmethod
    def from_serializable(cls, contents):
        """Instantiate a ReviewVectorizer from a serializable dictionary

        Args:
            contents (dict): the serializable dictionary
        Returns:
            an instance of the ReviewVectorizer class
        """
        predictor_vocab = Vocabulary.from_serializable(contents['predictor_vocab'])
        target_vocab = Vocabulary.from_serializable(contents['target_vocab'])

        return cls(predictor_vocab=predictor_vocab, target_vocab=target_vocab,
                  max_predictor_length=contents['max_predictor_length'])

    def to_serializable(self):
        """Create the serializable dictionary for caching

        Returns:
            contents (dict): the serializable dictionary
        """
        return {'predictor_vocab': self.predictor_vocab.to_serializable(),
                'target_vocab': self.target_vocab.to_serializable(),
                'max_predictor_length': self._max_predictor_length}