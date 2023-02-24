from transformers import AdamW, AutoModelForQuestionAnswering

from src.libs.utils import read_yaml
from src.preprocessing.dataset import Dataset
from src.training.trainer import Trainer

config = read_yaml("configs/config.yaml")

model_dir = f"resources/models/{config['model']}"
tokenized_dir = f"resources/tokenized_data/{config['model']}"

train_dataset = Dataset(
    tokenized_dir,
    config["max_length"],
    config["stride"],
    config["min_answer_length"],
    selected_questions=config["selected_questions"],
)
model = AutoModelForQuestionAnswering.from_pretrained(model_dir)

trainer = Trainer(
    model=model,
    optimizer=AdamW,
    batch_size=config["batch_size"],
    learning_rate=config["learning_rate"],
    epochs=config["epochs"],
)

trainer.train(train_dataset)
