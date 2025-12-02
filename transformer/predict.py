import torch
from transformers import BertForSequenceClassification
from pathlib import Path
import sys

INT_MIN = -256
INT_MAX = 255
VOCAB_OFFSET = 4
PAD_ID = 0
SEP_ID = 1
CLS_ID = 2
UNK_ID = 3
MAX_LEN = 128
EXAMPLES_PER_PROBLEM = 5

COMPONENTS = [
    "ZIPWITH", "*", "MAP", "SQR", "MUL4", "DIV4", "-",
    "MUL3", "DIV3", "MIN", "+", "SCANL1", "SHR", "SHL",
    "MAX", "HEAD", "DEC", "SUM", "doNEG", "isNEG",
    "INC", "LAST", "MINIMUM", "isPOS", "SORT", "FILTER",
    "isODD", "REVERSE", "ACCESS", "isEVEN", "COUNT",
    "TAKE", "MAXIMUM", "DROP",
]

def encode_integer(n):
    if n < INT_MIN or n > INT_MAX:
        return UNK_ID
    return (n - INT_MIN) + VOCAB_OFFSET

def safe_parse_line(line_str):
    line_str = line_str.strip()
    if not line_str:
        return []
    
    parts = line_str.split('|')
    args = []
    for p in parts:
        nums = [int(x) for x in p.split()]
        args.append(nums)
    return args

def process_entry_data(input_str, output_str):
    input_ids = [CLS_ID]
    token_type_ids = [0]

    raw_inputs = safe_parse_line(input_str)
    
    for x in raw_inputs:
        tokens = [encode_integer(i) for i in x]
        
        input_ids.extend(tokens)
        token_type_ids.extend([0] * len(tokens))
        
        input_ids.append(SEP_ID)
        token_type_ids.append(0)

    raw_outputs = safe_parse_line(output_str)
    
    for x in raw_outputs:
        tokens = [encode_integer(i) for i in x]
        input_ids.extend(tokens)
        token_type_ids.extend([1] * len(tokens))
    
    input_ids.append(SEP_ID)
    token_type_ids.append(1)

    if len(input_ids) >= MAX_LEN:
        input_ids = input_ids[:MAX_LEN]
        token_type_ids = token_type_ids[:MAX_LEN]
    
    attention_mask = [1] * len(input_ids)
    padding_length = MAX_LEN - len(input_ids)
    
    if padding_length > 0:
        input_ids = input_ids + [PAD_ID] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        token_type_ids = token_type_ids + [0] * padding_length

    return input_ids, attention_mask, token_type_ids

def write_predictions(output_path, scores):
    combined = list(zip(COMPONENTS, scores))
    combined.sort(key=lambda x: x[1], reverse=True)
    
    with open(output_path, "w") as f:
        for name, s in combined:
            f.write(f"{s:.6f} {name}\n")

if __name__ == "__main__":
    current_script_dir = Path(__file__).parent.resolve()
    project_root = current_script_dir.parent

    MODEL_PATH = current_script_dir / "models" / "deepcoder_transformer" / "checkpoint-759402"
    DATA_DIR = project_root / "enumerative-search" / "data" / "example"
    OUTPUT_DIR = DATA_DIR / "predictions_transformer"

    print(f"Model Path: {MODEL_PATH}")
    print(f"Data Dir:   {DATA_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    input_file = DATA_DIR / "input_values.txt"
    output_file = DATA_DIR / "output_values.txt"

    if not MODEL_PATH.exists():
        print(f"Error: Model checkpoint not found at {MODEL_PATH}")
        sys.exit(1)

    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    print("Loading data...")
    all_inputs = input_file.read_text().splitlines()
    all_outputs = output_file.read_text().splitlines()
    
    total_lines = len(all_inputs)
    num_problems = total_lines // EXAMPLES_PER_PROBLEM
    print(f"Total lines: {total_lines}")
    print(f"Total Problems: {num_problems}")

    print("Starting prediction...")
    with torch.no_grad():
        for problem_idx in range(num_problems):
            if problem_idx % 100 == 0:
                print(f"Processing problem {problem_idx}/{num_problems}...")

            start_line = problem_idx * EXAMPLES_PER_PROBLEM
            
            batch_input_ids = []
            batch_masks = []
            batch_type_ids = []

            for i in range(EXAMPLES_PER_PROBLEM):
                line_idx = start_line + i
                inp_ids, mask, type_ids = process_entry_data(all_inputs[line_idx], all_outputs[line_idx])
                
                batch_input_ids.append(inp_ids)
                batch_masks.append(mask)
                batch_type_ids.append(type_ids)

            input_tensor = torch.tensor(batch_input_ids, dtype=torch.long).to(device)
            mask_tensor = torch.tensor(batch_masks, dtype=torch.long).to(device)
            type_tensor = torch.tensor(batch_type_ids, dtype=torch.long).to(device)

            tokens = {
                'input_ids': input_tensor, 
                'attention_mask': mask_tensor,
                'token_type_ids': type_tensor
            }
            
            logits = model(**tokens).logits 
            
            probs = torch.sigmoid(logits)     
            avg_probs = torch.mean(probs, dim=0)
            
            out_filename = OUTPUT_DIR / f"{problem_idx}.txt"
            write_predictions(out_filename, avg_probs.cpu().tolist())

    print(f"Predictions saved to {OUTPUT_DIR}")