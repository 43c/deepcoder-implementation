#!/usr/bin/env python3
import sys
import os
from pathlib import Path

current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
sys.path.append(str(project_root))

import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    BertConfig, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    TrainerCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
import numpy as np

class CleanPrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            if "eval_loss" in logs:
                epoch = logs.get("epoch", 0)
                val_loss = logs.get("eval_loss", 0)
                f1 = logs.get("eval_f1_micro", 0)
                recall = logs.get("eval_recall_micro", 0)

                print(f"Epoch: {epoch:5.1f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"F1: {f1:.4f} | "
                      f"Recall: {recall:.4f}")

# components same as the successor in enumerative-search
COMPONENTS = [
    "ZIPWITH", "*", "MAP", "SQR", "MUL4", "DIV4", "-",
    "MUL3", "DIV3", "MIN", "+", "SCANL1", "SHR", "SHL",
    "MAX", "HEAD", "DEC", "SUM", "doNEG", "isNEG",
    "INC", "LAST", "MINIMUM", "isPOS", "SORT", "FILTER",
    "isODD", "REVERSE", "ACCESS", "isEVEN", "COUNT",
    "TAKE", "MAXIMUM", "DROP",
]

# Integer range
INT_MIN = -256
INT_MAX = 255
VOCAB_OFFSET = 4

# special token IDs
PAD_ID = 0
SEP_ID = 1
CLS_ID = 2
UNK_ID = 3
VOCAB_SIZE = (INT_MAX - INT_MIN + 1) + VOCAB_OFFSET

def encode_integer(n):
    if n < INT_MIN or n > INT_MAX:
        return UNK_ID
    return (n - INT_MIN) + VOCAB_OFFSET

def process_entry(example, max_len=128):
    input_ids = [CLS_ID]
    token_type_ids = [0]
    
    inp_val = example.inputs 
    out_val = example.output

    if isinstance(inp_val, (list, tuple)):
        args = inp_val
    else:
        args = [inp_val]

    for x in args:
        if isinstance(x, (list, tuple)):
            tokens = [encode_integer(i) for i in x]
        else:
            tokens = [encode_integer(x)]
        
        input_ids.extend(tokens)
        token_type_ids.extend([0] * len(tokens))
        
        input_ids.append(SEP_ID)
        token_type_ids.append(0)

    if isinstance(out_val, (list, tuple)):
        tokens = [encode_integer(x) for x in out_val]
    else:
        tokens = [encode_integer(out_val)]
    
    input_ids.extend(tokens)
    token_type_ids.extend([1] * len(tokens))
    
    input_ids.append(SEP_ID)
    token_type_ids.append(1)

    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
        token_type_ids = token_type_ids[:max_len]
            
    return input_ids, token_type_ids

class DeepCoderIntegerDataset(Dataset):
    def __init__(self, flat_data, max_length = 128):
        self.data = flat_data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example, labels = self.data[idx]
        input_ids, token_type_ids = process_entry(example, self.max_length)
        
        attention_mask = [1] * len(input_ids)
        
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [PAD_ID] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            token_type_ids = token_type_ids + [0] * padding_length
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
        }

def calculate_score(prediction):
    logits, labels = prediction
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs > 0.5).astype(int)
    f1 = f1_score(labels, predictions, average='micro', zero_division=0)
    recall = recall_score(labels, predictions, average='micro', zero_division=0)
    return {"f1_micro": f1, "recall_micro": recall}

def flatten_dataset(entries):
    flat_data = []
    for entry in entries:
        labels = [1.0 if entry.attribute.get(comp, False) else 0.0 for comp in COMPONENTS]
        for example in entry.examples:
            flat_data.append((example, labels))
    return flat_data

if __name__ == "__main__":
    model_dir = current_script_path.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset (Bickle 100k)...")
    dataset_path = project_root / "dataset" / "train" / "bickle100k.pickle"
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "rb") as f:
        d = pickle.load(f)
    
    if hasattr(d, "dataset"):
        all_data = list(d.dataset)
    else:
        all_data = d 

    print(f"Total programs loaded: {len(all_data)}")

    train_entries, test_entries = train_test_split(all_data, test_size=0.1, random_state=42)
    
    print("Flattening dataset...")
    flat_train = flatten_dataset(train_entries)
    flat_test = flatten_dataset(test_entries)
    
    print(f"Training examples: {len(flat_train)}")
    print(f"Testing examples:  {len(flat_test)}")
    
    train_dataset = DeepCoderIntegerDataset(flat_train, max_length=128)
    test_dataset = DeepCoderIntegerDataset(flat_test, max_length=128)

    print("Initializing Custom Model from Scratch...")
    
    config = BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
        num_labels=len(COMPONENTS),
        problem_type="multi_label_classification",
        hidden_dropout_prob=0.1, 
        attention_probs_dropout_prob=0.1,
    )

    model = BertForSequenceClassification(config)

    print(f"Model Parameters: {model.num_parameters() / 1e6:.2f} Million")

    # train
    args = TrainingArguments(
        output_dir=str(model_dir),
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-4,
        num_train_epochs=100,
        weight_decay=0.01,
        logging_strategy="epoch",
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=None,
        load_best_model_at_end=False,
        metric_for_best_model="f1_micro",
        
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=calculate_score,
        callbacks=[CleanPrinterCallback()],
    )

    print("Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. All checkpoints have been saved.")
    trainer.save_model(model_dir)
    print(f"Model saved to {model_dir}")