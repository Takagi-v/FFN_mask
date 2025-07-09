import torch
from safetensors import safe_open
import sys
from tqdm import tqdm
import os
import csv

# 确保能导入您项目的模型模块
project_dir = "/public/home/sjtu_intern/users/yilu.cao"  # 修改为您项目的路径
if project_dir not in sys.path:
    sys.path.append(project_dir)

# 导入您自定义的模型类
try:
    from models.modeling_qwen2 import Qwen2ForCausalLM
except ImportError as e:
    print(f"错误：无法从 models.modeling_qwen2 导入 Qwen2ForCausalLM。")
    print(f"请确保 '{project_dir}' 是正确的，并且模型文件位于其预期的位置。")
    print(f"详细信息: {e}")
    sys.exit(1)

from transformers import AutoTokenizer

# --- 配置 ---
MASK_BASE_PATH = "./output/qwen2-3b/xnli_local/"
MASK_CHECKPOINT_SUBDIR = "checkpoint-12380/model.safetensors"
MODEL_PATH = "/public/share/model/Qwen2.5-3B-Instruct"
OUTPUT_CSV_DIR = "./translation_results_with"

LANGUAGE_PAIRS = [
    "en_ar", "en_de", "en_es", "en_fr", "en_ru", "en_zh",
    "zh_ar", "zh_de", "zh_en", "zh_es", "zh_fr", "zh_ru",
]

# 语言代码到名称的映射 (当前在 get_test_sentences 中未直接用于格式化句子)
LANG_NAMES = {
    "en": "English", "de": "German", "fr": "French",
    "es": "Spanish", "ar": "Arabic", "ru": "Russian", "zh": "Chinese"
}

QWEN2_TEMPLATE = "<|im_start|>user\n{src}<|im_end|>\n<|im_start|>assistant\n"

# --- 测试句子 ---

# 英语源句子
# 10个“普通文本”（期望模型进行翻译）
EN_GENERAL_SENTENCES = [
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
# 10个“指令文本”（直接的指令或问题）
EN_INSTRUCTIONAL_PROMPTS = [
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms.",
    "Write a short poem about a starry night.",
    "Can you summarize a complex topic for me?",
    "Tell me a fun fact about the ocean.",
    "What are the main ingredients in a Margherita pizza?",
    "Give me three tips for learning a new language.",
    "Who wrote 'Hamlet'?",
    "Describe a typical day for an astronaut in space.",
    "If I have 5 apples and I give away 2, how many do I have left?"
]

# 中文源句子
# 10个“普通文本”（期望模型进行翻译）
ZH_GENERAL_SENTENCES = [
    "书在桌子上。",
    "今天天气晴朗，阳光明媚。",
    "我喜欢读历史故事。",
    "咖啡是我早上最喜欢的饮料。",
    "旅行能开阔眼界，丰富阅历。",
    "学习一门新语言很有挑战性，但也很有趣。",
    "科技正以前所未有的速度改变我们的生活。",
    "你今晚想吃什么？",
    "让我们共同努力实现我们的共同目标。",
    "桂林山水甲天下。"
]
# 10个“指令文本”（直接的指令或问题）
ZH_INSTRUCTIONAL_PROMPTS = [
    "法国的首都是哪里？",
    "用简单的语言解释一下相对论。",
    "写一首关于星空的短诗。",
    "你能帮我总结一个复杂的主题吗？", # 修正：之前英文版为 "Can you summarize a complex topic for me?"
    "告诉我一个关于海洋的有趣事实。",
    "玛格丽特披萨的主要成分是什么？",
    "给我三个学习新语言的建议。",
    "《哈姆雷特》是谁写的？",
    "描述一下宇航员在太空中的典型一天。",
    "如果我有5个苹果，送出去了2个，我还剩几个？"
]

def get_test_sentences(src_lang_code):
    """为给定的源语言生成20个测试句子。"""
    sentences = []
    if src_lang_code == "en":
        sentences.extend(EN_GENERAL_SENTENCES)
        sentences.extend(EN_INSTRUCTIONAL_PROMPTS)
    elif src_lang_code == "zh":
        sentences.extend(ZH_GENERAL_SENTENCES)
        sentences.extend(ZH_INSTRUCTIONAL_PROMPTS)
    else:
        print(f"警告：源语言 '{src_lang_code}' 没有预定义的句子。将跳过。")
        return []
    return sentences

# --- 主脚本 ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    print(f"CSV 结果将保存在: {OUTPUT_CSV_DIR}")

    print(f"\n正在从 {MODEL_PATH} 加载分词器和模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = Qwen2ForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        model.eval()
    except Exception as e:
        print(f"错误：无法从 {MODEL_PATH} 加载模型或分词器。")
        print(f"详细错误: {e}")
        sys.exit(1)
    print("分词器和模型加载成功。")

    for lang_pair_str in LANGUAGE_PAIRS:
        src_lang, tgt_lang = lang_pair_str.split('_') # tgt_lang 在此用于标识语言对，而非格式化句子
        print(f"\n\n{'='*20} 正在处理语言对: {src_lang.upper()} -> {tgt_lang.upper()} {'='*20}")

        mask_file_path = os.path.join(MASK_BASE_PATH, lang_pair_str, MASK_CHECKPOINT_SUBDIR)
        print(f"尝试从以下路径加载掩码文件: {mask_file_path}")
        try:
            with safe_open(mask_file_path, framework="pt") as f:
                mask_tensor_raw = f.get_tensor("tensor")
        except Exception as e:
            print(f"错误：无法加载掩码文件 {mask_file_path}。")
            print(f"由于掩码加载错误，跳过语言对 {lang_pair_str}。")
            print(f"详细错误: {e}")
            continue

        print(f"原始掩码张量形状: {mask_tensor_raw.shape}")
        mask_tensor = mask_tensor_raw.flatten()
        mask_tensor = torch.sigmoid(mask_tensor)
        binary_mask = (mask_tensor >= 0.5).float()
        weights_tensor = binary_mask.unsqueeze(0).to(device)
        print(f"用于模型的已处理掩码张量形状: {weights_tensor.shape}")

        test_texts = get_test_sentences(src_lang) # 现在只传入源语言代码
        if not test_texts:
            print(f"{src_lang} -> {tgt_lang} 没有测试句子。跳过。")
            continue

        csv_file_name = f"results_{lang_pair_str}.csv"
        csv_file_path = os.path.join(OUTPUT_CSV_DIR, csv_file_name)
        
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Original Input", "Model Output (No Mask)", "Model Output (With Mask)"])

            print(f"\n开始测试 {src_lang.upper()} -> {tgt_lang.upper()}:")
            for i, text in enumerate(tqdm(test_texts, desc=f"处理 {lang_pair_str} ({src_lang}->{tgt_lang})"), 1):
                formatted_input = QWEN2_TEMPLATE.format(src=text)
                inputs = tokenizer(formatted_input, return_tensors="pt", truncation=True, max_length=512).to(device)
                current_results = [text]

                try:
                    with torch.no_grad():
                        outputs_no_mask = model.generate(
                            **inputs, max_new_tokens=256, do_sample=False,
                            num_beams=5, pad_token_id=tokenizer.eos_token_id
                        )
                    translation_no_mask = tokenizer.decode(outputs_no_mask[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    current_results.append(translation_no_mask.strip())
                except Exception as e:
                    print(f"\n无掩码生成时发生错误，输入: {text}")
                    print(f"详细错误: {e}")
                    current_results.append(f"错误: {e}")

                try:
                    with torch.no_grad():
                        outputs_with_mask = model.generate(
                            **inputs, max_new_tokens=256, weight_tensor=weights_tensor,
                            do_sample=False, num_beams=5, pad_token_id=tokenizer.eos_token_id
                        )
                    translation_with_mask = tokenizer.decode(outputs_with_mask[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    current_results.append(translation_with_mask.strip())
                except Exception as e:
                    print(f"\n带掩码生成时发生错误，输入: {text}")
                    print(f"详细错误: {e}")
                    current_results.append(f"错误: {e}")
                
                csv_writer.writerow(current_results)

                if i <= 2 or i == 10 or i == 11 or i == 20 : # 打印几条样本结果，便于观察
                    print(f"\n输入 ({lang_pair_str} - {i}): {text}")
                    print(f"  无掩码输出: {current_results[1]}")
                    print(f"  有掩码输出: {current_results[2]}")
                    if i < len(test_texts): print("-" * 30)
        print(f"\n{lang_pair_str} 的结果已保存到 {csv_file_path}")
    print("\n\n所有语言对测试完成。")

if __name__ == "__main__":
    main()