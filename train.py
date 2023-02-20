import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, AutoModelForQuestionAnswering

from src.preprocessing.dataset import Dataset

tokenized_dir = "resources/tokenized_data/roberta-base"
max_length = 512
stride = 128
min_answer_length = 100
model_name = "roberta-base"
selected_questions = [
    "Cap On Liability",
    "Governing Law",
    "No-Solicit Of Employees",
    "Non-Compete",
]

model_dir = f"resources/models/{model_name}"


train_dataset = Dataset(
    tokenized_dir,
    max_length,
    stride,
    min_answer_length,
    selected_questions=selected_questions,
)
model = AutoModelForQuestionAnswering.from_pretrained(model_dir)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in tqdm(
        train_loader, total=train_loader.__len__(), desc=f"epoch {epoch}"
    ):
        optim.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()
