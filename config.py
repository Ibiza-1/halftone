"""实验配置：训练、评估和推理统一从这里读取。数值是当前默认值，可自行修改。"""
from pathlib import Path
import torch

# ===== 共用配置 =====
PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 128         # 正方形图像边长，同时是 ONet 的输出类别数
DEVICE = "auto"          # auto / cpu / cuda / cuda:0 / mps
INPUT_IMAGE = PROJECT_DIR / "data" / "input.png"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
PREVIEW_SIZE = 512       # 保存预览图的边长，不改变实际计算尺寸

# ===== ONet 结构、训练和评估 =====
ONET_HIDDEN_SIZE = 256
ONET_SAMPLES_PER_BIN = 100
ONET_TRAIN_STEPS = 5000
ONET_LEARNING_RATE = 1e-4
ONET_LOG_INTERVAL = 500
ONET_RANDOM_SEED = 7
ONET_EVAL_SAMPLES_PER_BIN = 100
# None：根据结构和训练次数生成路径；也可填写自己指定的 Path。
# 加载旧权重时，IMAGE_SIZE 和 ONET_HIDDEN_SIZE 必须与其结构一致。
ONET_CHECKPOINT = None

# ===== PNet 结构、训练和推理 =====
PNET_CHANNELS = 64
PNET_TRAIN_STEPS = 100
PNET_LEARNING_RATE = 1e-4
PNET_LOG_INTERVAL = 10
PNET_RANDOM_SEED = 42
PNET_MODE = "train"      # train：从头训练并保存；infer：加载权重，只推理
PNET_CHECKPOINT = None   # None：自动生成路径；推理也读取这个路径

# ===== 渲染和损失 =====
RENDER_CHUNK_SIZE = 256  # 每批墨点数；训练仍需保留各批求导数据
NUM_DOTS = None          # None：按目标总墨量估计；正整数：指定墨点数
GAUSSIAN_KERNEL_SIZE = 11
GAUSSIAN_SIGMA = 1.5
# ===== 配置结束，下面是共用辅助函数 =====


def onet_checkpoint_path():
    if ONET_CHECKPOINT is not None:
        return Path(ONET_CHECKPOINT)
    # 默认宽度沿用已有文件名；其他宽度添加标识，避免互相覆盖。
    width_tag = "" if ONET_HIDDEN_SIZE == 256 else f"_h{ONET_HIDDEN_SIZE}"
    return CHECKPOINT_DIR / f"onet_{IMAGE_SIZE}{width_tag}_ce_{ONET_TRAIN_STEPS}.pth"


def pnet_checkpoint_path():
    if PNET_CHECKPOINT is not None:
        return Path(PNET_CHECKPOINT)
    return CHECKPOINT_DIR / f"pnet_{IMAGE_SIZE}_c{PNET_CHANNELS}_{PNET_TRAIN_STEPS}.pth"


def resolve_device(spec=DEVICE):
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("DEVICE 指定了 CUDA，但当前 PyTorch 无法使用 CUDA。")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("DEVICE 指定了 MPS，但当前环境无法使用 MPS。")
    return device


def validate_config():
    positive_integers = {
        "IMAGE_SIZE": IMAGE_SIZE, "PREVIEW_SIZE": PREVIEW_SIZE,
        "ONET_HIDDEN_SIZE": ONET_HIDDEN_SIZE,
        "ONET_SAMPLES_PER_BIN": ONET_SAMPLES_PER_BIN,
        "ONET_EVAL_SAMPLES_PER_BIN": ONET_EVAL_SAMPLES_PER_BIN,
        "ONET_TRAIN_STEPS": ONET_TRAIN_STEPS,
        "ONET_LOG_INTERVAL": ONET_LOG_INTERVAL,
        "PNET_CHANNELS": PNET_CHANNELS,
        "PNET_TRAIN_STEPS": PNET_TRAIN_STEPS,
        "PNET_LOG_INTERVAL": PNET_LOG_INTERVAL,
        "RENDER_CHUNK_SIZE": RENDER_CHUNK_SIZE,
        "GAUSSIAN_KERNEL_SIZE": GAUSSIAN_KERNEL_SIZE,
    }
    for name, value in positive_integers.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} 必须为正整数，当前为 {value!r}")
    if IMAGE_SIZE < 2:
        raise ValueError("IMAGE_SIZE 至少为 2。")
    if NUM_DOTS is not None and (type(NUM_DOTS) is not int or NUM_DOTS <= 0):
        raise ValueError("NUM_DOTS 必须为 None 或正整数。")
    for name, value in {
        "ONET_LEARNING_RATE": ONET_LEARNING_RATE,
        "PNET_LEARNING_RATE": PNET_LEARNING_RATE,
        "GAUSSIAN_SIGMA": GAUSSIAN_SIGMA,
    }.items():
        if not 0 < value < float("inf"):
            raise ValueError(f"{name} 必须为有限正数。")
    if GAUSSIAN_KERNEL_SIZE % 2 != 1 or GAUSSIAN_KERNEL_SIZE // 2 >= IMAGE_SIZE:
        raise ValueError("高斯核必须为奇数，且半径必须小于 IMAGE_SIZE。")
    if PNET_MODE not in ("train", "infer"):
        raise ValueError('PNET_MODE 必须是 "train" 或 "infer"。')
