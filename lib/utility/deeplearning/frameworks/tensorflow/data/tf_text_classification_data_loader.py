from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


@dataclass
class TFTextClassificationDataBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_train_onehot: np.ndarray
    y_test_onehot: np.ndarray
    tokenizer: Tokenizer
    vocab_size: int
    max_sequence_length: int
    max_nb_words: int


class TFTextClassificationDataLoader:
    """
    Build tokenized and padded text datasets for TensorFlow text classification.
    """

    @staticmethod
    def from_text_and_target(
        texts: pd.Series,
        targets: pd.Series,
        max_nb_words: int = 20000,
        max_sequence_length: int = 150,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> TFTextClassificationDataBundle:
        cleaned_texts = (
            texts.fillna("")
            .astype("string")
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        y = targets.astype(int).to_numpy()

        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            cleaned_texts,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        tokenizer = Tokenizer(
            num_words=max_nb_words,
            oov_token="<OOV>",
        )
        tokenizer.fit_on_texts(X_train_texts.tolist())

        X_train_seq = tokenizer.texts_to_sequences(X_train_texts.tolist())
        X_test_seq = tokenizer.texts_to_sequences(X_test_texts.tolist())

        X_train = pad_sequences(
            X_train_seq,
            maxlen=max_sequence_length,
            padding="post",
            truncating="post",
        )
        X_test = pad_sequences(
            X_test_seq,
            maxlen=max_sequence_length,
            padding="post",
            truncating="post",
        )

        y_train_onehot = np.eye(2, dtype=np.float32)[y_train]
        y_test_onehot = np.eye(2, dtype=np.float32)[y_test]

        vocab_size = min(max_nb_words, len(tokenizer.word_index) + 1)

        return TFTextClassificationDataBundle(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            y_train_onehot=y_train_onehot,
            y_test_onehot=y_test_onehot,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            max_nb_words=max_nb_words,
        )
