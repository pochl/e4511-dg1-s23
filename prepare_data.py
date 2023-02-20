import pandas as pd
from transformers import AutoTokenizer
from src.preprocessing.preparer import Preparer


model_name = "roberta-base"
data_path = 'resources/data/cuad/train_separate_questions.json'


model_dir = f"resources/models/{model_name}"
tokenized_dir = f"resources/tokenized_data/{model_name}"

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

preparer = Preparer(data_path, tokenizer, tokenized_dir)

preparer.tokenize(instance_type='question')
preparer.tokenize(instance_type='context')
preparer.get_answer_span()