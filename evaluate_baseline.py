import os
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
BASE_DATASET_PATH = "/public/home/sjtu_intern/users/yilu.cao/dataset"
GSM8K_CACHE_DIR = os.path.join(BASE_DATASET_PATH, "gsm8k")
INSTRUCT_MODEL_PATH = "/public/share/model/Qwen2.5-7B-Instruct"
NUM_SAMPLES_TO_EVALUATE = 50

# Load dataset
print("Loading GSM8K dataset...")
gsm8k_dataset_full = load_dataset("gsm8k", "main", cache_dir=GSM8K_CACHE_DIR)
gsm8k_test = gsm8k_dataset_full['test']

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    INSTRUCT_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

# Verify model loaded correctly
print(f"Model loaded. Type: {type(model)}")
print(f"Tokenizer chat template available: {tokenizer.chat_template is not None}")

def extract_answer_simple(text):
    """Extract numerical answer from model output"""
    # Look for #### pattern first
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1)
    
    # Look for common answer patterns
    patterns = [
        r"answer is:?\s*(-?\d+(?:\.\d+)?)",
        r"total is:?\s*(-?\d+(?:\.\d+)?)",
        r"result is:?\s*(-?\d+(?:\.\d+)?)",
        r"=\s*(-?\d+(?:\.\d+)?)\s*$"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1)
    
    # Last resort: find last number
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None

def normalize_number(num_str):
    """Normalize number for comparison"""
    if num_str is None:
        return None
    try:
        num = float(num_str)
        # Handle integer conversion
        if num.is_integer():
            return str(int(num))
        else:
            # For decimals, round to avoid floating point issues
            return f"{num:.6f}".rstrip('0').rstrip('.')
    except:
        return num_str

# Test with official chat template
def test_with_chat_template():
    print("\n=== Testing with Official Chat Template (Random Sampling) ===")
    
    correct = 0
    # --- 核心修改：改为随机抽样 ---
    total = min(NUM_SAMPLES_TO_EVALUATE, len(gsm8k_test))
    
    # 随机抽样需要评估的样本，确保每次运行结果可复现
    sampled_gsm8k_test = gsm8k_test.shuffle(seed=42).select(range(total))
    print(f"Using random sampling with seed=42, testing {total} samples")
    # --- 修改结束 ---
    
    for i in range(total):
        example = sampled_gsm8k_test[i]  # 从抽样后的数据集中获取样本
        question = example['question']
        true_answer = extract_answer_simple(example['answer'])
        true_answer = normalize_number(true_answer)
        
        # Method 1: Use chat template if available
        if tokenizer.chat_template:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Solve math problems step by step and end your answer with #### followed by the numerical answer."},
                {"role": "user", "content": question}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Method 2: Manual format
            prompt = f"<|im_start|>system\nYou are a helpful assistant. Solve math problems step by step and end your answer with #### followed by the numerical answer.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
                max_new_tokens=512,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred_answer = extract_answer_simple(generated)
        pred_answer = normalize_number(pred_answer)
        
        is_correct = (true_answer == pred_answer)
        if is_correct:
            correct += 1
        
        if i < 3 or not is_correct:  # Show first 3 and errors
            print(f"\nExample {i+1}:")
            print(f"Question: {question[:100]}...")
            print(f"True answer: {true_answer}")
            print(f"Predicted: {pred_answer}")
            print(f"Generated (last 200): ...{generated[-200:]}")
            print(f"Correct: {is_correct}")
    
    print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.1f}%")
    return correct/total

# Alternative: Test with different generation strategies
def test_generation_strategies():
    print("\n=== Testing Different Generation Strategies ===")
    
    test_question = "John has 5 apples. He buys 3 more apples and then gives 2 apples to his friend. How many apples does John have now?"
    
    strategies = [
        {"name": "Greedy", "do_sample": False},
        {"name": "Beam Search", "do_sample": False, "num_beams": 4},
        {"name": "Low Temperature", "do_sample": True, "temperature": 0.1, "top_p": 0.95},
    ]
    
    for strategy in strategies:
        print(f"\n{strategy['name']}:")
        
        prompt = f"<|im_start|>user\n{test_question}\nSolve step by step and end with #### answer<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        gen_kwargs = {k: v for k, v in strategy.items() if k != "name"}
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids.to(model.device),
                max_new_tokens=200,
                pad_token_id=tokenizer.pad_token_id,
                **gen_kwargs
            )
        
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Output: {generated}")
        answer = extract_answer_simple(generated)
        print(f"Extracted answer: {answer}")

# Run tests
print("\n" + "="*50)
accuracy = test_with_chat_template()

if accuracy < 0.6:  # If accuracy is still low
    print("\n" + "="*50)
    test_generation_strategies()
    
    # Additional diagnostic
    print("\n=== Model Diagnostic ===")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Vocab size: {model.config.vocab_size}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Test tokenizer
    print("\n=== Tokenizer Test ===")
    test_text = "The answer is #### 42"
    tokens = tokenizer.tokenize(test_text)
    print(f"Tokenization of '{test_text}': {tokens}")
    
    # Check for special tokens
    print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")