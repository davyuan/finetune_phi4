import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from Phi4ForSequenceClassification import Phi4ForSequenceClassification
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             average_precision_score)

MODEL_NAME = "microsoft/phi-4"
BATCH_SIZE = 1
EPOCHS = 1
MAX_LEN = 1024
NUM_CLASSES = 2
OUTPUT_DIR = "./finetuned_phi4_content_safety_QLoRA"
GRADIENT_ACCUMULATION_STEPS = 4

def create_model_and_tokenizer(tokenizer, device, local_rank, loss_fn=None): 
    # for QLoRA we load the backbone in 4-bit NF4 + LoRA adapters
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",  # IMPORTANT
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    backbone = AutoModel.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    backbone = get_peft_model(backbone, lora_config)

    model = Phi4ForSequenceClassification(None, backbone=backbone).to(device)

    # Ensure classifier + norm are trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    for param in model.norm.parameters():
        param.requires_grad = True


    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    return model, tokenizer

def create_dataloaders(tokenizer):
    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")

    train_labels = torch.tensor(dataset["train"]["toxicity"], dtype=torch.long)
    class_counts = torch.bincount(train_labels, minlength=NUM_CLASSES).float()
    class_weights = train_labels.numel() / (NUM_CLASSES * class_counts.clamp(min=1.0))
    class_weights = class_weights.to(torch.float16)

    def preprocess(example):
        encoded = tokenizer(
            example["user_input"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
        )
        encoded["labels"] = int(example["toxicity"])
        return encoded

    dataset = dataset.map(
        preprocess,
        batched=False,
        remove_columns=dataset["train"].column_names,
    )

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset["train"],
        shuffle=True,
    )

    train_loader = DataLoader(
        dataset["train"],
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4,
    )

    eval_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset["test"],
        shuffle=False,
    )

    eval_loader = DataLoader(
        dataset["test"],
        batch_size=BATCH_SIZE,
        sampler=eval_sampler,
        pin_memory=True,
        num_workers=4,
    )

    return train_loader, eval_loader, train_sampler, eval_sampler, class_weights


def train_one_epoch(model, loader, optimizer, device, log_every=100, gradient_accumulation_steps=1):
    model.train()

    running_loss = torch.zeros(1, device=device)
    running_correct = torch.zeros(1, device=device)
    running_total = torch.zeros(1, device=device)
    optimizer.zero_grad()
    last_step = 0

    for step, batch in enumerate(loader, start=1):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs["loss"]
        logits = outputs["logits"]

        scaled_loss = loss / gradient_accumulation_steps
        scaled_loss.backward()

        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        last_step = step

        preds = logits.argmax(dim=-1)
        running_loss += loss.detach()
        running_correct += (preds == batch["labels"]).sum()
        running_total += batch["labels"].size(0)

        if step % log_every == 0:
            # reduce across all GPUs
            dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(running_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(running_total, op=dist.ReduceOp.SUM)

            if dist.get_rank() == 0:
                avg_loss = running_loss.item() / log_every / dist.get_world_size()
                acc = running_correct.item() / running_total.item()

                print(
                    f"Step {step:6d} | "
                    f"loss={avg_loss:.4f} | "
                    f"acc={acc:.4f}"
                )

            running_loss.zero_()
            running_correct.zero_()
            running_total.zero_()

    if last_step and last_step % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs["logits"]

        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)

        all_labels.append(batch["labels"])
        all_preds.append(preds)
        all_probs.append(probs)

    labels = torch.cat(all_labels)
    preds = torch.cat(all_preds)
    probs = torch.cat(all_probs)

    # gather across GPUs
    labels_list = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
    preds_list  = [torch.empty_like(preds) for _ in range(dist.get_world_size())]
    probs_list  = [torch.empty_like(probs) for _ in range(dist.get_world_size())]

    dist.all_gather(labels_list, labels)
    dist.all_gather(preds_list, preds)
    dist.all_gather(probs_list, probs)

    if dist.get_rank() == 0:
        y_true = torch.cat(labels_list).float().cpu().numpy()
        y_pred = torch.cat(preds_list).cpu().numpy()
        y_prob = torch.cat(probs_list).float().cpu().numpy()

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": roc_auc_score(y_true, y_prob),
            "auprc": average_precision_score(y_true, y_prob),
        }

    return None


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()


def save_finetuned_model(model, tokenizer, output_dir):
    if dist.get_rank() != 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if isinstance(model, DDP) else model
    model_to_save.backbone.save_pretrained(output_dir)
    head_state = {
        "classifier": model_to_save.classifier.state_dict(),
        "norm": model_to_save.norm.state_dict(),
    }
    torch.save(head_state, os.path.join(output_dir, "head.pt"))
    tokenizer.save_pretrained(output_dir)
    print(f"Saved fine-tuned model to {output_dir}")


def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    train_loader, eval_loader, train_sampler, eval_sampler, class_weights = create_dataloaders(tokenizer)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    model, tokenizer = create_model_and_tokenizer(tokenizer, device, local_rank, loss_fn=loss_fn)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4,      
        weight_decay=0.01,
    )

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)

        train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        )

        eval_sampler.set_epoch(epoch)
        metrics = evaluate(model, eval_loader, device)

        if metrics and dist.get_rank() == 0:
            print(f"\nEpoch {epoch}")
            for k, v in metrics.items():
                print(f"{k:10s}: {v:.4f}")

    save_finetuned_model(model, tokenizer, OUTPUT_DIR)
    cleanup_ddp()

if __name__ == "__main__":
    main()

