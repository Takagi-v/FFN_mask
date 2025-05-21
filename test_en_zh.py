import torch
from safetensors import safe_open
import sys
from tqdm import tqdm

# 确保能导入您项目的模型模块
project_dir = "/public/home/sjtu_intern/users/yilu.cao"  # 修改为您项目的路径
if project_dir not in sys.path:
    sys.path.append(project_dir)

# 导入您自定义的模型类
from models.modeling_qwen2 import Qwen2ForCausalLM # Assuming this is correctly located
from transformers import AutoTokenizer

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 加载掩码文件 (路径已修改为 en_zh)
# IMPORTANT: Ensure this path and checkpoint number are correct for your en_zh mask
mask_file = "./output/qwen2-3b/xnli_local_wo/en_zh/checkpoint-12380/model.safetensors" # MODIFIED
try:
    with safe_open(mask_file, framework="pt") as f:
        mask_tensor = f.get_tensor("tensor")
except Exception as e:
    print(f"错误：无法加载掩码文件 {mask_file}。请检查路径和文件是否存在。")
    print(f"详细错误: {e}")
    sys.exit(1)


print(f"掩码张量形状: {mask_tensor.shape}")

# 确保掩码是一维的
mask_tensor = mask_tensor.flatten()
# 应用sigmoid
mask_tensor = torch.sigmoid(mask_tensor)
binary_mask = (mask_tensor >= 0.5).float()

# 加载模型和分词器
model_path = "/public/share/model/Qwen2.5-3B-Instruct"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Qwen2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
except Exception as e:
    print(f"错误：无法加载模型或分词器从 {model_path}。请检查路径和模型文件。")
    print(f"详细错误: {e}")
    sys.exit(1)

# Qwen2的指令模板
QWEN2_TEMPLATE = "<|im_start|>user\n{src}<|im_end|>\n<|im_start|>assistant\n"

# 多条英文测试样例 (已修改)
test_texts = [
    "The book is on the table.",
    "The weather is beautiful and sunny today.",
    "I enjoy reading historical stories.",
    "Coffee is my favorite drink in the morning.",
    "Traveling broadens horizons and enriches experiences.",
    "Learning a new language is challenging, but also very interesting.",
    "Technology is changing our lives at an unprecedented speed.",
    "What do you want to eat tonight?",
    "Let's work together to achieve our common goals.",
    "The scenery of Guilin is the best under heaven."
]


# 准备掩码张量，只需要计算一次
weights_tensor = binary_mask.unsqueeze(0).to(device)  # 添加批量维度

print("\n\n开始正式测试 (en_zh):") # MODIFIED
# 测试每个样例
for i, text in enumerate(tqdm(test_texts, desc="处理测试样例"), 1):
    formatted_input = QWEN2_TEMPLATE.format(src=text)
    print(f"\n第{i}个输入: {text}")
    # print(f"格式化后: {formatted_input}") # Can be verbose, optionally uncomment
    
    # 标记化输入
    inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
    
    # 无掩码生成
    with torch.no_grad():
        outputs_no_mask = model.generate(
            **inputs,
            max_length=256, 
            do_sample=False,
            num_beams=5,
            pad_token_id=tokenizer.eos_token_id 
        )
    translation_no_mask = tokenizer.decode(outputs_no_mask[0, inputs.input_ids.shape[1]:], skip_special_tokens=True) 
    print(f"  模型无掩码输出: {translation_no_mask.strip()}")
    
    # 使用掩码生成
    with torch.no_grad():
        outputs_with_mask = model.generate(
            **inputs,
            max_length=256, 
            weight_tensor=weights_tensor,
            do_sample=False,
            num_beams=5,
            pad_token_id=tokenizer.eos_token_id
        )
    translation_with_mask = tokenizer.decode(outputs_with_mask[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  模型有掩码输出: {translation_with_mask.strip()}")
    
    if i < len(test_texts):
        print("-" * 80)

print("\n测试完成。")