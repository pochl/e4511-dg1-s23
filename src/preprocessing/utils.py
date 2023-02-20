from typing import List, Tuple

import pandas as pd

from src.libs.utils import read_json


def read_squad(path: str) -> pd.DataFrame:
    """Read dataset in SQuAD format.

    Args:
        path (str): Path to dataset.

    Returns:
        pd.DataFrame: Dataset.
    """

    squad_dict = read_json(path)

    titles = []
    questions_id = []
    contexts = []
    questions = []
    answers = []
    for group in squad_dict["data"]:
        title = group["title"]
        for passage in group["paragraphs"]:
            context = passage["context"]
            for ques_ans in passage["qas"]:
                question = ques_ans["question"]
                question_id = (
                    ques_ans["id"].split("__")[-1].split("_")[0].replace("/", "-")
                )
                for answer in ques_ans["answers"]:
                    titles.append(title)
                    questions_id.append(question_id)
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return pd.DataFrame.from_records(
        {
            "context_id": titles,
            "question_id": questions_id,
            "context": contexts,
            "question": questions,
            "answer": answers,
        }
    )


def get_answer_span(context: dict, answer: dict) -> Tuple[int, int]:
    """Generate answer span, which is the indexes of start and end tokens (inclusive) in the context.

    Args:
        context (dict): Dictionary containing tokenized information of the context, with the following keys:
            "input_ids", "attention_mask", "offset_mapping", "sequence_ids". This dictionary is processed from
            Hugging Face's tokenizer object.
        answer (dict): Dictionary containing information of the answer in SQuAD format.

    Returns:
        Tuple[int, int]: (start_position, end_position index)
    """

    offset = context["offset_mapping"]

    start_char = answer["answer_start"]
    end_char = answer["answer_start"] + len(answer["text"])
    sequence_ids = context["sequence_ids"]

    context_end = len(sequence_ids) - 1

    idx = 0
    while idx <= context_end and offset[idx][0] <= start_char:
        idx += 1
    start_position = idx - 1

    idx = context_end
    while idx >= 0 and offset[idx][1] >= end_char:
        idx -= 1
    end_position = idx + 1

    return start_position, end_position


def calculate_overlap(x_start: int, x_end: int, y_start: int, y_end: int) -> int:
    """Calculates overlapping span between 2 spans (x, y). For example, if x's span is (5, 15), and
    y's span is (9, 20), the the overlapping span is 7.

    Args:
        x_start (int): Start position of span x.
        x_end (int): Ens position of span x.
        y_start (int): Start position of span y.
        y_end (int): End position of span y.

    Returns:
        int: Overlaping span.
    """

    return max(0, min(x_end, y_end) - max(x_start, y_start))


def resample(data: pd.DataFrame, contain_answer: pd.Series) -> pd.DataFrame:
    """Resample data such that the numbers of positive samples (the sample contain the answer to the question), and
    negative samples (the sample does not contain the answer to the question), are equal.

    Args:
        data (pd.DataFrame): Data to be sample
        contain_answer (pd.Series): Whether each sample contains the answer or not.

    Returns:
        pd.DataFrame: Resampled data.
    """

    negative = data[~contain_answer].sample(sum(contain_answer), random_state=2023)
    positive = data[contain_answer]

    return pd.concat([positive, negative])
