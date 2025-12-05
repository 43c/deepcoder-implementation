import pickle, sys, os
import time
sys.path.append(os.path.abspath(".."))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from pathlib import Path

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
VOCAB_OFFSET = 2

# special token IDs
PAD_ID = 0
UNK_ID = 1

# vocabulary size
VOCAB_SIZE = (INT_MAX - INT_MIN + 1) + VOCAB_OFFSET
TYPE_INT_TOKEN = VOCAB_SIZE
TYPE_ARRAY_TOKEN = VOCAB_SIZE + 1
FULL_VOCAB_SIZE = VOCAB_SIZE + 2

# hyperparameters
EMBEDDING_DIM = 20      
HIDDEN_SIZE = 256       
NUM_LAYERS = 3          
MAX_INPUTS = 3
MAX_ARRAY_LEN = 20
EXAMPLE_MAX_LEN = (MAX_INPUTS + 1) * (MAX_ARRAY_LEN + 1)
MAX_EXAMPLES = 5

# tokenizer
def encode_integer(n):
    if n < INT_MIN or n > INT_MAX:
        return UNK_ID
    return (n - INT_MIN) + VOCAB_OFFSET

def get_type_token(x):
    if isinstance(x, (list, tuple)):
        return TYPE_ARRAY_TOKEN
    else:
        return TYPE_INT_TOKEN
    
def encode_value(x, max_array_len):
    vals = []
    if isinstance(x, (list, tuple)):
        for v in x:
            vals.append(encode_integer(v))
            if len(vals) >= max_array_len:
                break
    else:
        vals.append(encode_integer(x))
    if len(vals) < max_array_len:
        vals.extend([PAD_ID] * (max_array_len - len(vals)))
    return vals

def process_entry(example, max_length):
    input_ids = []
    inp_val = example.inputs
    out_val = example.output

    if isinstance(inp_val, (list, tuple)):
        inputs_list = list(inp_val)
    else:
        inputs_list = [inp_val]
    inputs_list = inputs_list[:MAX_INPUTS]

    for x in inputs_list:
        input_ids.append(get_type_token(x))
        input_ids.extend(encode_value(x, MAX_ARRAY_LEN))

    input_ids.append(get_type_token(out_val))
    input_ids.extend(encode_value(out_val, MAX_ARRAY_LEN))

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    padding_len = max_length - len(input_ids)
    if padding_len > 0:
        input_ids = input_ids + [PAD_ID] * padding_len      
    return input_ids

def calculate_score(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    f1 = f1_score(labels_np, preds_np, average='micro')
    return f1


class DeepCoderDataset(Dataset):
    def __init__(self, dataset, max_examples = 5, max_length = 20):
        self.dataset = dataset
        self.max_examples = max_examples
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dataset = self.dataset[idx]
        processed_examples = []
        cur_examples = dataset.examples[:self.max_examples]
        for example in cur_examples:
            processed_example = process_entry(example, self.max_length)
            processed_examples.append(processed_example)
        while len(processed_examples) < self.max_examples:
            processed_examples.append([PAD_ID] * self.max_length)
        labels = [1.0 if dataset.attribute.get(comp, False) else 0.0 for comp in COMPONENTS]
        return {
            # tensor shape (5, 20)
            "examples": torch.tensor(processed_examples, dtype = torch.long), 
            # tensor shape (components true/false)
            "labels": torch.tensor(labels, dtype = torch.float),
        }
    
class DeepCoderEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, input_length):
        super(DeepCoderEncoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx = PAD_ID)
        self.flat_dim = input_length * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0) #64
        num_examples = x.size(1) #5
        # [batch, 5, 20] -> [batch, 5, 20, 20]
        x = self.embedding(x)
        # [batch, 5, 20, 20] -> [batch, 5, 400]
        x = x.view(batch_size, num_examples, -1)
        # [64, 5, 400] -> [320, 400]
        x = x.view(batch_size * num_examples, -1)
        # [320, 400] -> [320, 256]
        x = self.mlp(x)
        # [320, 256] -> [64, 5, 256]
        x = x.view(batch_size, num_examples, -1)
        return x

class DeepCoderDecoder(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(DeepCoderDecoder, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, encoder_features):
        pooled_features = encoder_features.mean(dim=1)
        logits = self.classifier(pooled_features)
        return logits
    
class DeepCoderModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, input_length, num_classes):
        super(DeepCoderModel, self).__init__()
        self.encoder = DeepCoderEncoder(num_embeddings, embedding_dim, hidden_size, input_length)
        self.decoder = DeepCoderDecoder(hidden_size, num_classes)
    def forward(self, x):
        encoded_features = self.encoder(x)
        logits = self.decoder(encoded_features)
        return logits

    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dataset_path = "../dataset/train/bickle100k.pickle"
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    with open(dataset_path, "rb") as f:
        d = pickle.load(f)
    if hasattr(d, "dataset"):
        all_data = list(d.dataset)
    else:
        all_data = d 

    print(f"Total dataset loaded")

    train_entries, test_entries = train_test_split(all_data, test_size=0.1, random_state=42)

    processed_train_data = DeepCoderDataset(train_entries, max_examples=MAX_EXAMPLES, max_length=EXAMPLE_MAX_LEN)
    processed_test_data = DeepCoderDataset(test_entries, max_examples=MAX_EXAMPLES, max_length=EXAMPLE_MAX_LEN)
    
    batch_size = 64
    train_loader = DataLoader(processed_train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(processed_test_data, batch_size=batch_size, shuffle=False)

    model = DeepCoderModel(
        num_embeddings=FULL_VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        input_length=EXAMPLE_MAX_LEN,
        num_classes=len(COMPONENTS)
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # train
    num_epochs = 100
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            inputs = batch["examples"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        #test
        model.eval()
        total_test_loss = 0.0
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["examples"].to(device)
                labels = batch["labels"].to(device)
                
                logits = model(inputs)
                loss = criterion(logits, labels)
                
                total_test_loss += loss.item()
                all_logits.append(logits)
                all_labels.append(labels)
        
        avg_test_loss = total_test_loss / len(test_loader)
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        val_f1 = calculate_score(all_logits, all_labels)
        
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1:03d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_test_loss:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"time={elapsed:.0f}s"
        )

        save_name = model_dir / f"epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_name)

    print("models saved")
