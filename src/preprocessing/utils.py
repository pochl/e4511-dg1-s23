from typing import List

import pandas as pd

from src.libs.utils import read_json


def read_squad(path: str):

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
            for qa in passage["qas"]:
                question = qa["question"]
                question_id = qa["id"].split("__")[-1].split("_")[0].replace("/", "-")
                for answer in qa["answers"]:
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


def get_answer_span(context, answer):

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


def calculate_overlap(x_start, x_end, y_start, y_end):
    return max(0, min(x_end, y_end) - max(x_start, y_start))


def resample(data: pd.DataFrame, contain_answer: List[bool]):

    negative = data[~contain_answer].sample(sum(contain_answer), random_state=2023)
    positive = data[contain_answer]

    return pd.concat([positive, negative])
