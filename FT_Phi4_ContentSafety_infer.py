import argparse
import os

import torch
from transformers import AutoTokenizer

from Phi4ForSequenceClassification import Phi4ForSequenceClassification

MAX_LEN = 1024
MODEL_NAME = "microsoft/phi-4-mini-instruct"

def load_finetuned_model(model_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    model = Phi4ForSequenceClassification(
        model_dir,
        num_classes=2,
    )

    head_path = os.path.join(model_dir, "head.pt")
    if os.path.exists(head_path):
        head_state = torch.load(head_path, map_location=device)
        model.classifier.load_state_dict(head_state["classifier"])
        model.norm.load_state_dict(head_state["norm"])
    else:
        raise FileNotFoundError(f"Missing head weights at {head_path}")

    model.to(device)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, texts, device):
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

    return [
        {
            "text": text,
            "score": float(probs[i, 1].item()),
            "pred": int(preds[i].item()),
        }
        for i, text in enumerate(texts)
    ]


def main():
    parser = argparse.ArgumentParser(description="Run inference on the finetuned Phi4 content-safety head")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./finetuned_phi4_content_safety",
        help="Directory where the backbone and head checkpoints were saved",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for inference",
    )
    parser.add_argument(
        "--examples",
        nargs="+",
        default=[
            "Hello there, how are you?",
            "I hate everyone who disagrees with me.",
        ],
        help="Text strings that should be scored",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer = load_finetuned_model(args.model_dir, device)

    results = predict(model, tokenizer, args.examples, device)
    for result in results:
        label = "unsafe" if result["pred"] == 1 else "safe"
        print(f"Input: {result['text']}")
        print(f"  prediction: {label}")
        print(f"  unsafe probability: {result['score']:.4f}\n")


if __name__ == "__main__":
    main()