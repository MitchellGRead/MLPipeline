import torch
import torch.nn as nn


class FinetunedLLM(nn.Module):  # pragma: no cover, torch model
    """Model architecture for a Large Language Model (LLM) that we will fine-tune."""

    def __init__(self, llm, threshold, embedding_dim, num_classes):
        super(FinetunedLLM, self).__init__()
        self.llm = llm
        self.dropout = torch.nn.Dropout(threshold)
        self.fc1 = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, batch):
        ids, masks = batch["ids"], batch["masks"]
        _, pool = self.llm(input_ids=ids, attention_mask=masks)
        z = self.dropout(pool)
        z = self.fc1(z)
        return z
