import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    squad_convert_examples_to_features
)
from torch.utils.data import DataLoader, SequentialSampler
from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample
from transformers.data.metrics.squad_metrics import compute_predictions_logits

def predictor(questions, context, model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)

    max_length = 512
    stride = 128

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu") #this

    model.to(device)

    # Create SquadExample for each question
    # SquadExample is a single training/test example for the Squad dataset
    examples = []
    for i, question_text in enumerate(questions):
        example = SquadExample(
            qas_id=str(i),
            question_text=question_text,
            context_text=context,
            answer_text=None,
            start_position_character=None,
            title="Predict",
            answers=None,
        )
        examples.append(example)
    # Converts a list of examples into a list of features
    # that can be directly given as input to a model.
    features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_length,
            doc_stride=stride,
            max_query_length=max_length,
            is_training=False,
            return_dataset="pt", # returns a torch.data.TensorDataset
            threads=1,
    )
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=8)

    results = []
    for batch in dataloader: #each sample
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad(): #disabled gradient calculation
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            example_indices = batch[3]
            outputs = model(**inputs)
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()] #question
                unique_id = int(eval_feature.unique_id)

                output = [output[i].detach().cpu().tolist() for output in outputs.to_tuple()]
                start_positions, end_positions = output
                result = SquadResult(unique_id, start_positions, end_positions )
                results.append(result)

    # get best answer
    predictions = compute_predictions_logits(
            all_examples=examples,
            all_features=features,
            all_results=results,
            n_best_size=1,
            max_answer_length=max_length,
            do_lower_case=False,
            output_prediction_file=None,
            output_nbest_file=None,
            output_null_log_odds_file=None,
            verbose_logging=False,
            version_2_with_negative=True,
            null_score_diff_threshold=0.0,
            tokenizer=tokenizer
    )
    return predictions
