from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering

from src.preprocessing.dataset import Dataset


class Trainer:
    """Trainer class."""

    def __init__(
        self,
        model: AutoModelForQuestionAnswering,
        optimizer: Callable,
        batch_size: int,
        learning_rate: float,
        epochs: int,
    ) -> None:
        """Trainer class

        Args:
            model (AutoModelForQuestionAnswering): Question-Answering Model object from Hugging Face.
            Callable (Optimizer): Optimizer object from Hugging Face or PyTorch.
            batch_size (int): Batch size.
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of epochs for training.
        """

        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.train()

        self.optim = optimizer(self.model.parameters(), lr=learning_rate)

    def train(self, data: Dataset):
        """Train the model.

        Args:
            data (Dataset): Training data.
        """

        data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)

        for epoch in range(3):
            for batch in tqdm(
                data_loader, total=len(data_loader), desc=f"epoch {epoch}"
            ):
                self.optim.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                start_positions = batch["start_positions"].to(self.device)
                end_positions = batch["end_positions"].to(self.device)
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )
                loss = outputs[0]
                loss.backward()
                self.optim.step()

        self.model.eval()
