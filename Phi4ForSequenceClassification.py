import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class Phi4ForSequenceClassification(nn.Module):
    def __init__(self, model_name, backbone=None, num_classes=2, loss_fn=None, dtype=torch.bfloat16):
        super().__init__()

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        hidden_size = self.backbone.config.hidden_size

        self.norm = nn.LayerNorm(hidden_size).to(dtype)
        self.classifier = nn.Linear(hidden_size, num_classes).to(dtype)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

    def mean_pool(self, hidden_states, attention_mask):
        #attention_mask: [B, T]
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype) #[B, T, 1]
        summed = (hidden_states * mask).sum(dim=1) #[B,T, D] * [B, T, 1] = [B, T, D].sum(dim=1) = [B, D]
        counts = mask.sum(dim=1).clamp(min=1e-6)   #[B, T, 1].sum(dim=1) = [B, 1]
        return summed / counts                     #[B, D] / [B, 1] = [B, D]

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        pooled = self.norm(pooled)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}
