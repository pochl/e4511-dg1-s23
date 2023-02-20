import os
import shutil
from itertools import compress
from typing import List

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging

from src.libs.utils import read_pickle, save_json, save_pickle
from src.preprocessing.utils import get_answer_span, read_squad


class Preparer:
    """A Class to prepare a given dataset."""

    def __init__(self, data_path: str, tokenizer: AutoTokenizer, out_dir: str) -> None:
        """Init.
        Args:
            data_path (str): Path to data. The data must be in SQuAD format.
            tokenizer (AutoTokenizer): Hugging Face's Tokenizer object.
            out_dir (str): Output directory to store the tokenized data.
        """

        self.tokenizer = tokenizer
        self.out_dir = out_dir

        logging.get_logger("transformers.tokenization_utils_base").setLevel(
            logging.ERROR
        )

        self.data = read_squad(data_path)

        self.seperators = self.get_seperators()

        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)

        os.makedirs(self.out_dir)

        self.get_tokenizer_info()

    def get_seperators(self) -> List[int]:
        """Retrieves sperators of tokenizer.

        Returns:
            List[int]: Token ID of seperator. -1 indicate non-seperator token.
        """

        example = self.tokenizer("test", "test", padding="do_not_pad")
        is_seperator = [x is None for x in example.sequence_ids(0)]
        seperators_idx = np.array(range(len(is_seperator)))[is_seperator]

        idx_ = 0
        seperators = []
        for idx in seperators_idx:
            if idx - idx_ <= 1:
                seperators.append(example["input_ids"][idx])
            else:
                seperators.append(-1)
                seperators.append(example["input_ids"][idx])

            idx_ = idx

        return seperators

    def get_answer_span(self):
        """Gets and saves answer span to csv file."""

        answer_starts = []
        answer_ends = []
        for _, row in tqdm(
            self.data.iterrows(), total=len(self.data), desc="computing answer spans"
        ):

            context = read_pickle(f"{self.out_dir}/context/{row.context_id}.pickle")
            answer = row["answer"]

            start_position, end_position = get_answer_span(context, answer)
            answer_starts.append(start_position)
            answer_ends.append(end_position)

        answers_span = self.data[["question_id", "context_id"]].copy()
        answers_span.loc[:, "answer_start"] = answer_starts
        answers_span.loc[:, "answer_end"] = answer_ends
        answers_span.to_csv(self.out_dir + "/answers_span.csv", index=False)

    def tokenize(self, instance_type: str):
        """Tokenizes the data and saves to pickle file.

        Args:
            instance_type (str): Type of the data. Either "question" or "context".
        """

        assert instance_type in (
            "question",
            "context",
        ), 'instance_type must be either "question" or "context".'

        os.makedirs(f"{self.out_dir}/{instance_type}")

        instances = self.data[[f"{instance_type}_id", instance_type]].drop_duplicates()

        for _, row in tqdm(
            instances.iterrows(),
            total=len(instances),
            desc=f"tokenizing {instance_type}s",
        ):

            instance_id = row[f"{instance_type}_id"]

            encode_dict = {}
            encode = self.tokenizer(
                text=row[instance_type],
                max_length=None,
                truncation=False,
                stride=None,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="do_not_pad",
            )

            not_seperator = [x is not None for x in encode.sequence_ids(0)]
            encode_dict["input_ids"] = list(
                compress(encode["input_ids"][0], not_seperator)
            )
            encode_dict["attention_mask"] = list(
                compress(encode["attention_mask"][0], not_seperator)
            )
            encode_dict["offset_mapping"] = list(
                compress(encode["offset_mapping"][0], not_seperator)
            )
            encode_dict["sequence_ids"] = list(
                compress(encode.sequence_ids(0), not_seperator)
            )

            save_pickle(
                encode_dict, f"{self.out_dir}/{instance_type}/{instance_id}.pickle"
            )

    def get_tokenizer_info(self):
        """Gets and saves tokenizer information."""

        padding_id = self.tokenizer.pad_token_id
        seperators = self.get_seperators()
        info = {"seperators": seperators, "padding_id": padding_id}
        save_json(info, f"{self.out_dir}/tokenizer_info.json")
