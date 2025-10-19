import os
import re
import torch
import json
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
BASE_DATASET_PATH = "/public/home/sjtu_intern/users/yilu.cao/dataset"
MATH_CACHE_DIR = os.path.join(BASE_DATASET_PATH, "competition-math")
INSTRUCT_MODEL_PATH = "/public/home/sjtu_intern/users/yilu.cao/Qwen2.5-Math-7B-Instruct"
NUM_SAMPLES_TO_EVALUATE = 20

# Load MATH dataset
print("Loading MATH dataset...")
math_dataset_full = load_dataset("competition_math", cache_dir=MATH_CACHE_DIR)
math_test = math_dataset_full['test']

print(f"MATH test set size: {len(math_test)}")
print(f"Available subjects: {set(math_test['type'])}")

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

def extract_answer_from_math_solution(text):
    """Extract answer from MATH dataset solution format"""
    # MATH dataset answers are often in LaTeX format, need to extract the final numerical or algebraic answer
    
    # Look for #### pattern first (if we add it in our prompt)
    match = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if match:
        return match.group(1).strip()
    
    # Look for common answer patterns in mathematical text
    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is:?\s*(.+?)(?:\.|$|\n)",
        r"(?:therefore|thus|hence),?\s*(.+?)(?:\.|$|\n)",
        r"(?:so|then),?\s*(.+?)(?:\.|$|\n)",
        r"(?:result|solution)\s*(?:is)?\s*:?\s*(.+?)(?:\.|$|\n)",
        r"=\s*(.+?)(?:\.|$|\n)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1).strip()
            # Clean up common mathematical notation
            answer = answer.replace('$', '').replace('\\', '').strip()
            return answer
    
    # Last resort: try to find the last mathematical expression
    # Look for standalone numbers or simple expressions at the end
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.lower().startswith(('let', 'we', 'since', 'because', 'if', 'given')):
            # Try to extract a clean answer from this line
            clean_line = re.sub(r'[^\w\s\d\.\-\+\*/\(\)=,]', '', line)
            if clean_line.strip():
                return clean_line.strip()
    
    return None

def normalize_math_answer(answer_str, true_answer_str):
    """Normalize mathematical answers for comparison"""
    if answer_str is None or true_answer_str is None:
        return answer_str, true_answer_str
    
    def clean_answer(ans):
        if ans is None:
            return None
        # Remove common mathematical formatting
        ans = str(ans).strip()
        ans = ans.replace('$', '').replace('\\', '').replace(' ', '')
        ans = ans.replace('frac', '').replace('{', '').replace('}', '')
        
        # Try to evaluate simple numerical expressions
        try:
            # Handle simple fractions like "1/2"
            if '/' in ans and ans.count('/') == 1:
                parts = ans.split('/')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    result = float(parts[0]) / float(parts[1])
                    if result.is_integer():
                        return str(int(result))
                    return f"{result:.6f}".rstrip('0').rstrip('.')
            
            # Try direct float conversion
            num = float(ans)
            if num.is_integer():
                return str(int(num))
            else:
                return f"{num:.6f}".rstrip('0').rstrip('.')
        except:
            pass
        
        return ans.lower()
    
    return clean_answer(answer_str), clean_answer(true_answer_str)

def format_math_problem(problem, level, type_):
    """Format MATH problem for the model"""
    prompt = f"""Solve this {level}-level {type_} problem step by step:

{problem}

Please provide a detailed solution and end your answer with #### followed by the final answer."""
    
    return prompt

def test_math_dataset():
    """Test model performance on MATH dataset"""
    print("\n=== Testing Qwen2.5-Math-7B-Instruct on MATH Dataset ===")
    
    correct = 0
    total = min(NUM_SAMPLES_TO_EVALUATE, len(math_test))
    
    # Random sampling for reproducible results
    sampled_math_test = math_test.shuffle(seed=42).select(range(total))
    print(f"Using random sampling with seed=42, testing {total} samples")
    
    # Track performance by subject
    subject_stats = {}
    
    # Store detailed results for output file
    detailed_results = []
    
    for i in range(total):
        example = sampled_math_test[i]
        problem = example['problem']
        true_solution = example['solution']
        level = example['level']
        subject = example['type']
        
        # Initialize subject stats
        if subject not in subject_stats:
            subject_stats[subject] = {'correct': 0, 'total': 0}
        subject_stats[subject]['total'] += 1
        
        # Extract true answer from solution
        true_answer = extract_answer_from_math_solution(true_solution)
        
        # Format the problem
        formatted_problem = format_math_problem(problem, level, subject)
        
        # Create chat messages
        if tokenizer.chat_template:
            messages = [
                {"role": "system", "content": "You are a mathematical expert. Solve problems step by step with clear reasoning and provide the final answer."},
                {"role": "user", "content": formatted_problem}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Manual format
            prompt = f"<|im_start|>system\nYou are a mathematical expert. Solve problems step by step with clear reasoning and provide the final answer.<|im_end|>\n<|im_start|>user\n{formatted_problem}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize with longer max_length for complex math problems
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
                max_new_tokens=1024,  # Longer generation for detailed solutions
                do_sample=False,  # Use greedy decoding for mathematical accuracy
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred_answer = extract_answer_from_math_solution(generated)
        
        # Normalize answers for comparison
        norm_pred, norm_true = normalize_math_answer(pred_answer, true_answer)
        
        is_correct = (norm_pred == norm_true) if (norm_pred and norm_true) else False
        if is_correct:
            correct += 1
            subject_stats[subject]['correct'] += 1
        
        # Store detailed result
        result_entry = {
            "example_id": i + 1,
            "subject": subject,
            "level": level,
            "problem": problem,
            "true_solution": true_solution,
            "true_answer": true_answer,
            "predicted_answer": pred_answer,
            "normalized_true": norm_true,
            "normalized_pred": norm_pred,
            "is_correct": is_correct,
            "model_output": generated,
            "formatted_prompt": formatted_problem
        }
        detailed_results.append(result_entry)
        
        # Show details for all examples (since we only have 20)
        print(f"\n{'='*80}")
        print(f"Example {i+1}/{total} - {subject} (Level {level}) - {'✅ CORRECT' if is_correct else '❌ WRONG'}")
        print(f"{'='*80}")
        print(f"Problem:\n{problem}")
        print(f"\nTrue answer: {true_answer}")
        print(f"Predicted answer: {pred_answer}")
        print(f"Normalized - True: {norm_true}, Predicted: {norm_pred}")
        print(f"\nModel's full response:")
        print("-" * 60)
        print(generated)
        print("-" * 60)
        
        # Progress indicator
        current_accuracy = correct / (i + 1) * 100
        print(f"Progress: {i+1}/{total}, Current accuracy: {current_accuracy:.1f}%")
    
    # Save detailed results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"qwen_math_evaluation_{timestamp}.json"
    
    evaluation_summary = {
        "model_path": INSTRUCT_MODEL_PATH,
        "dataset": "competition-math",
        "timestamp": timestamp,
        "total_samples": total,
        "correct_answers": correct,
        "overall_accuracy": correct / total * 100,
        "subject_statistics": subject_stats,
        "detailed_results": detailed_results
    }
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_filename}")
    
    # Final results
    print(f"\n{'='*80}")
    print("FINAL RESULTS - Qwen2.5-Math-7B-Instruct")
    print(f"{'='*80}")
    
    overall_accuracy = correct / total * 100
    print(f"Overall Accuracy: {correct}/{total} = {overall_accuracy:.1f}%")
    
    print(f"\nPerformance by Subject:")
    print("-" * 50)
    for subject, stats in sorted(subject_stats.items()):
        subject_acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"{subject:20}: {stats['correct']:2}/{stats['total']:2} = {subject_acc:5.1f}%")
    
    return overall_accuracy / 100

def test_specific_subject(subject_name, num_samples=10):
    """Test on a specific subject"""
    print(f"\n=== Testing {subject_name} Problems ===")
    
    # Filter by subject
    subject_data = math_test.filter(lambda x: x['type'] == subject_name)
    
    if len(subject_data) == 0:
        print(f"No problems found for subject: {subject_name}")
        return
    
    print(f"Found {len(subject_data)} {subject_name} problems")
    
    # Sample problems
    sample_size = min(num_samples, len(subject_data))
    sampled_data = subject_data.shuffle(seed=42).select(range(sample_size))
    
    correct = 0
    for i, example in enumerate(sampled_data):
        problem = example['problem']
        true_solution = example['solution']
        level = example['level']
        
        print(f"\n{subject_name} Problem {i+1} (Level {level}):")
        print("-" * 50)
        print(problem)
        print(f"\nTrue solution preview: {true_solution[:200]}...")
        
        # You can add evaluation logic here if needed
    
    return subject_data

# Main execution
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🧮 Testing Qwen2.5-Math-7B-Instruct on Competition Math Dataset")
    print("="*80)
    
    # Show available subjects first
    subjects = list(set(math_test['type']))
    print(f"Available subjects in MATH dataset: {subjects}")
    print(f"Total problems in dataset: {len(math_test)}")
    print(f"Testing {NUM_SAMPLES_TO_EVALUATE} samples...")
    
    # Run evaluation
    accuracy = test_math_dataset()
    
    print(f"\n{'='*80}")
    print("🎯 EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Model: Qwen2.5-Math-7B-Instruct")
    print(f"Dataset: Competition Math")
    print(f"Samples tested: {NUM_SAMPLES_TO_EVALUATE}")
    print(f"Final Accuracy: {accuracy*100:.1f}%")
    
    if accuracy < 0.4:
        print("\nNote: MATH dataset is extremely challenging.")
        print("Performance interpretation:")
        print("  < 20%: Below average for 7B models")
        print("  20-30%: Average performance")
        print("  30-40%: Good performance")
        print("  > 40%: Excellent performance")
    
    print(f"\n✅ Detailed results saved as JSON file for further analysis.")