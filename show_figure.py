import numpy as np
import matplotlib.pyplot as plt
import torch
from safetensors import safe_open
import os
import seaborn as sns # seaborn 似乎未在原脚本的绘图函数中使用，但保留导入以防万一
import pandas as pd # 将 pandas 导入移到脚本开头

# --- 配置 ---
MASK_BASE_PATH = "./output/qwen2-3b/xnli_local_wo/"  # 与您上一个脚本一致
MASK_CHECKPOINT_SUBDIR = "checkpoint-12380/model.safetensors" # 与您上一个脚本一致
# 假设语言对与您上一个脚本中的定义相同
LANGUAGE_PAIRS = [
    "en_ar", "en_de", "en_es", "en_fr", "en_ru", "en_zh",
    "zh_ar", "zh_de", "zh_en", "zh_es", "zh_fr", "zh_ru",
]
N_LAYERS = 36  # Qwen2.5-3B-Instruct 模型的层数 (根据您的脚本)
INTERMEDIATE_SIZE = 11008  # 每层的 intermediate_size (根据您的脚本)

FIGURE_OUTPUT_DIR = "./figure"
STATS_OUTPUT_DIR = "./figure_stats" # 用于存放统计数据的CSV文件

# --- 函数定义 ---

def load_mask_tensor(file_path):
    """
    加载掩码张量文件，支持.safetensors和.pt格式
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"掩码文件未找到: {file_path}")
        
    if file_path.endswith('.safetensors'):
        with safe_open(file_path, framework="pt") as f:
            # 大多数情况下，张量名为"tensor"
            tensor = f.get_tensor("tensor")
    elif file_path.endswith('.pt') or file_path.endswith('.bin'):
        tensor = torch.load(file_path, map_location='cpu')
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")
    
    return tensor

def reshape_mask_tensor(tensor, n_layers, intermediate_size):
    """
    将掩码张量重塑为适合热力图显示的形式
    
    Args:
        tensor: 原始掩码张量
        n_layers: 模型层数
        intermediate_size: 每层的intermediate_size
        
    Returns:
        重塑后的tensor (经过sigmoid)，形状为[n_layers, intermediate_size]
    """
    original_n_layers = n_layers # 保存原始层数以备后用

    if tensor.dim() > 1:
        # 假设如果维度大于1，第一个维度是批次或其他，我们只取第一个有效掩码
        # 如果您的张量结构不同 (例如，已经是 [layers, neurons] 或其他)，您可能需要调整这里
        # 基于您的原始脚本，它直接 flatten()，所以我们先展平
        tensor_flat = tensor.flatten()
        # 检查是否是多个掩码的堆叠，例如LoRA中的多个任务掩码
        # 如果 tensor[0] 是预期的单个掩码，则使用 tensor[0].flatten()
        # 为安全起见，如果总元素数量是 n_layers * intermediate_size 的倍数，取第一个
        if tensor.shape[0] > 1 and tensor.numel() == tensor.shape[0] * original_n_layers * intermediate_size:
             print(f"警告: 输入张量形状为 {tensor.shape}，可能包含多个掩码。将使用第一个掩码 (tensor[0])。")
             tensor_flat = tensor[0].flatten()
        elif tensor.numel() != original_n_layers * intermediate_size and tensor.numel() % (original_n_layers * intermediate_size) == 0:
             # 可能是其他情况，例如合并的权重，这里只是一个猜测
             print(f"警告: 张量元素数量 {tensor.numel()} 是预期单个掩码大小的倍数。将使用第一个块。")
             tensor_flat = tensor.flatten()[:original_n_layers * intermediate_size]
        else:
             tensor_flat = tensor.flatten() # 默认行为
    else:
        tensor_flat = tensor.flatten()
    
    expected_length = original_n_layers * intermediate_size
    current_length = tensor_flat.shape[0]

    if current_length != expected_length:
        print(f"警告: 展平后的张量大小 {current_length} 与预期大小 {expected_length} ({original_n_layers}x{intermediate_size}) 不符。")
        if current_length % intermediate_size == 0:
            adjusted_n_layers = current_length // intermediate_size
            print(f"将调整层数为: {adjusted_n_layers} (基于 intermediate_size: {intermediate_size})")
            n_layers_to_reshape = adjusted_n_layers
        else:
            # 如果不能整除，这是一个更严重的问题，可能无法正确重塑
            print(f"错误: 无法根据 intermediate_size={intermediate_size} 重塑张量，长度为 {current_length}。将尝试使用原始层数，可能会失败或产生错误结果。")
            # 尝试截断或填充到预期长度 - 但这通常是不正确的，除非您知道为什么会这样
            # 为了避免错误，这里选择一个策略，例如截断或报错
            if current_length > expected_length:
                print(f"将截断张量至预期长度 {expected_length}。")
                tensor_flat = tensor_flat[:expected_length]
            else: # current_length < expected_length
                # 无法从此重塑，除非填充，但这没有意义
                raise ValueError(f"张量长度 {current_length} 小于预期长度 {expected_length} 且无法调整层数。")
            n_layers_to_reshape = original_n_layers
    else:
        n_layers_to_reshape = original_n_layers

    try:
        reshaped_tensor = tensor_flat.reshape(n_layers_to_reshape, intermediate_size)
    except RuntimeError as e:
        raise RuntimeError(f"无法将形状为 {tensor_flat.shape} 的张量重塑为 ({n_layers_to_reshape}, {intermediate_size})。原始张量形状: {tensor.shape}。错误: {e}")

    sigmoid_values = torch.sigmoid(reshaped_tensor)
    
    return sigmoid_values

def plot_heatmap(data, title_suffix=""):
    """
    绘制掩码热力图
    参数:
    - data: sigmoid后的掩码数据，形状为[n_layers, intermediate_size]
    - title_suffix: 添加到主标题后的后缀 (例如语言对信息)
    """
    data_np = data.cpu().numpy()
    n_layers, n_intermediate = data_np.shape
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [1, 1]})
    
    heatmap_data = data_np[::-1, :] # 翻转数据，使得层0在底部
    
    cax1 = ax1.imshow(heatmap_data, cmap='viridis', aspect='auto', vmin=0., vmax=1.)
    fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_title("Sigmoid激活值", fontsize=16)
    
    ax1.set_yticks([])
    ax1.set_xticks(np.arange(0, n_intermediate, max(1, n_intermediate//10))) # 确保步长至少为1
    
    ax_left1 = ax1.twinx()
    ax_left1.yaxis.tick_left()
    ax_left1.set_ylim(0, n_layers)
    ax_left1.set_yticks(np.arange(n_layers) + 0.5)
    ax_left1.set_yticklabels(np.arange(n_layers)[::-1], fontsize=10)
    ax_left1.set_ylabel("层索引", fontsize=14)
    ax_left1.yaxis.set_label_position("left")
    
    cax2 = ax2.imshow((heatmap_data >= 0.5).astype(float), cmap='viridis', aspect='auto', vmin=0., vmax=1.)
    fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title("二值化掩码 (Sigmoid值 >= 0.5)", fontsize=16)
    
    ax2.set_yticks([])
    ax2.set_xticks(np.arange(0, n_intermediate, max(1, n_intermediate//10))) # 确保步长至少为1
    
    ax_left2 = ax2.twinx()
    ax_left2.yaxis.tick_left()
    ax_left2.set_ylim(0, n_layers)
    ax_left2.set_yticks(np.arange(n_layers) + 0.5)
    ax_left2.set_yticklabels(np.arange(n_layers)[::-1], fontsize=10)
    ax_left2.set_ylabel("层索引", fontsize=14)
    ax_left2.yaxis.set_label_position("left")
    
    fig.suptitle(f"Qwen2 掩码层热力图 {title_suffix}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应标题和标签
    
    return fig

def plot_layer_activation_rates(data, title_suffix=""):
    """
    绘制每层激活率的条形图
    参数:
    - data: sigmoid后的掩码数据
    - title_suffix: 添加到主标题后的后缀
    """
    data_np = data.cpu().numpy()
    n_layers_actual = data_np.shape[0] # 使用数据的实际层数
    
    activation_rates = (data_np >= 0.5).mean(axis=1) * 100
    
    fig = plt.figure(figsize=(15, 8)) # 创建新的figure对象
    bars = plt.bar(np.arange(n_layers_actual), activation_rates, color='royalblue')
    
    for i, v in enumerate(activation_rates):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=9)
    
    plt.title(f"每层掩码激活率 {title_suffix}", fontsize=18)
    plt.xlabel("层索引", fontsize=14)
    plt.ylabel("激活率 (%)", fontsize=14)
    plt.xticks(np.arange(0, n_layers_actual, max(1, n_layers_actual//20))) # 确保步长至少为1
    plt.ylim(0, 110) # 留出顶部空间给文本
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig # 返回创建的figure对象

# --- 主执行逻辑 ---
if __name__ == "__main__":
    # 创建输出目录
    os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(STATS_OUTPUT_DIR, exist_ok=True)

    all_pairs_summary_stats = []

    for lang_pair_str in LANGUAGE_PAIRS:
        print(f"\n--- 正在处理语言对: {lang_pair_str.upper()} ---")
        mask_file = os.path.join(MASK_BASE_PATH, lang_pair_str, MASK_CHECKPOINT_SUBDIR)

        if not os.path.exists(mask_file):
            print(f"警告: 掩码文件 {mask_file} 未找到。跳过语言对 {lang_pair_str}。")
            all_pairs_summary_stats.append({
                "language_pair": lang_pair_str,
                "status": "file_not_found",
                "overall_activation_rate_percent": np.nan,
                "active_layers_count": np.nan
            })
            continue
        
        try:
            print(f"加载掩码: {mask_file}")
            tensor = load_mask_tensor(mask_file)
            print(f"加载的张量形状: {tensor.shape}, 类型: {tensor.dtype}, 设备: {tensor.device}")

            # 注意：reshape_mask_tensor 内部可能会根据张量实际大小调整层数
            sigmoid_values = reshape_mask_tensor(tensor, N_LAYERS, INTERMEDIATE_SIZE)
            current_n_layers = sigmoid_values.shape[0] # 获取重塑后的实际层数
            print(f"重塑并应用Sigmoid后的张量形状: {sigmoid_values.shape}")

            # 绘制并保存热力图
            print("生成热力图...")
            heatmap_fig = plot_heatmap(sigmoid_values, title_suffix=f"- {lang_pair_str.upper()}")
            heatmap_filename = os.path.join(FIGURE_OUTPUT_DIR, f"heatmap_{lang_pair_str}.png")
            heatmap_fig.savefig(heatmap_filename)
            plt.close(heatmap_fig) # 关闭图像以释放内存
            print(f"热力图已保存到: {heatmap_filename}")

            # 绘制并保存激活率图表
            print("生成激活率图表...")
            activation_fig = plot_layer_activation_rates(sigmoid_values, title_suffix=f"- {lang_pair_str.upper()}")
            activation_filename = os.path.join(FIGURE_OUTPUT_DIR, f"activation_rates_{lang_pair_str}.png")
            activation_fig.savefig(activation_filename)
            plt.close(activation_fig) # 关闭图像
            print(f"激活率图表已保存到: {activation_filename}")

            # 计算并保存统计数据
            print("计算统计数据...")
            layer_indices = list(range(current_n_layers))
            activation_rates_percent = ((sigmoid_values >= 0.5).float().mean(dim=1) * 100).cpu().numpy().tolist()
            
            stats_data = {
                "层索引": layer_indices,
                "激活率(%)": activation_rates_percent,
                "平均值(Sigmoid)": sigmoid_values.mean(dim=1).cpu().numpy().tolist(),
                "最大值(Sigmoid)": sigmoid_values.max(dim=1)[0].cpu().numpy().tolist(),
                "最小值(Sigmoid)": sigmoid_values.min(dim=1)[0].cpu().numpy().tolist()
            }
            stats_df = pd.DataFrame(stats_data).set_index("层索引")
            
            stats_filename = os.path.join(STATS_OUTPUT_DIR, f"stats_{lang_pair_str}.csv")
            stats_df.to_csv(stats_filename)
            print(f"统计数据已保存到: {stats_filename}")

            # 汇总统计
            overall_activation_rate = (sigmoid_values >= 0.5).float().mean().item() * 100
            active_layers = sum(1 for rate in activation_rates_percent if rate > 0) # 假设激活率大于0即为活跃层
            all_pairs_summary_stats.append({
                "language_pair": lang_pair_str,
                "status": "success",
                "overall_activation_rate_percent": round(overall_activation_rate, 2),
                "active_layers_count": active_layers,
                "total_layers_processed": current_n_layers
            })

        except FileNotFoundError as e: # 已在前面检查，但以防万一
            print(f"文件未找到错误处理 {lang_pair_str}: {e}")
            all_pairs_summary_stats.append({
                "language_pair": lang_pair_str,
                "status": "file_not_found_during_processing",
                "overall_activation_rate_percent": np.nan,
                "active_layers_count": np.nan
            })
        except ValueError as e:
            print(f"值错误处理 {lang_pair_str} (通常是文件格式或重塑问题): {e}")
            all_pairs_summary_stats.append({
                "language_pair": lang_pair_str,
                "status": f"value_error: {str(e)[:100]}", # 记录部分错误信息
                "overall_activation_rate_percent": np.nan,
                "active_layers_count": np.nan
            })
        except RuntimeError as e:
            print(f"运行时错误处理 {lang_pair_str} (通常是重塑或设备问题): {e}")
            all_pairs_summary_stats.append({
                "language_pair": lang_pair_str,
                "status": f"runtime_error: {str(e)[:100]}",
                "overall_activation_rate_percent": np.nan,
                "active_layers_count": np.nan
            })
        except Exception as e:
            print(f"处理 {lang_pair_str} 时发生未知错误: {e}")
            import traceback
            traceback.print_exc()
            all_pairs_summary_stats.append({
                "language_pair": lang_pair_str,
                "status": f"unknown_error: {str(e)[:100]}",
                "overall_activation_rate_percent": np.nan,
                "active_layers_count": np.nan
            })
    
    # 保存所有语言对的汇总统计
    if all_pairs_summary_stats:
        summary_df = pd.DataFrame(all_pairs_summary_stats)
        summary_filename = os.path.join(STATS_OUTPUT_DIR, "summary_all_language_pairs.csv")
        summary_df.to_csv(summary_filename, index=False)
        print(f"\n所有语言对的汇总统计数据已保存到: {summary_filename}")
        print(summary_df)

    print("\n所有处理完成。")