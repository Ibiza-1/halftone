import torch
import torch.nn.functional as F
from pathlib import Path

from onet import ONet


# ======================== ONet训练超参数 ========================
# DIM必须与train_single.py中的IMAGE_SIZE相同。
DIM = 128                # 一个坐标轴上的位置类别数
HIDDEN_SIZE = 256        # ONet隐藏层宽度
SAMPLES_PER_BIN = 100    # 每一个位置区间采样多少个连续坐标
TRAIN_STEPS = 5000       # 参数更新次数
LEARNING_RATE = 1e-4     # Adam学习率
LOG_INTERVAL = 500       # 每隔多少次更新打印一次结果
RANDOM_SEED = 7          # 固定初始化，方便复现实验
# ------------------------------------------------------------
# DEVICE：训练设备选择，可填以下任一字符串：
#   "auto"   —— 自动选择（优先 CUDA → MPS → CPU）
#   "cuda"   —— 使用第一块 NVIDIA/AMD GPU
#   "cuda:0" —— 显式指定第 0 块 GPU
#   "cuda:1" —— 显式指定第 1 块 GPU（多卡机器）
#   "mps"    —— Apple Silicon (M1/M2/M3...) 的 Metal 后端
#   "cpu"    —— 纯 CPU（macbook 默认）
DEVICE = "auto"
# ===============================================================
def resolve_device(spec):
    """根据字符串解析训练设备，方便后续把张量/模型搬过去。"""
    if spec is None or str(spec).strip() == "" or str(spec).strip().lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


if __name__ == '__main__':
    device = resolve_device(DEVICE)
    print("使用训练设备：", device)

    sample_ids = torch.arange(DIM * SAMPLES_PER_BIN)
    xs = (sample_ids.float() / SAMPLES_PER_BIN).unsqueeze(1)

    indices = sample_ids // SAMPLES_PER_BIN
    targets = F.one_hot(indices, num_classes=DIM).float()

    # 把训练数据和模型都搬到选定的设备上
    xs = xs.to(device)
    indices = indices.to(device)
    targets = targets.to(device)
    sample_ids = sample_ids.to(device)


    # print("输入形状：", xs.shape)
    # print("标签形状：", targets.shape)

    # for i in [0, 1234, 6399]:
    #     print(
    #         "坐标：", xs[i].item(),
    #         "类别：", indices[i].item(),
    #         "标签中1的位置：", targets[i].argmax().item(),
    #     )

    # print("每行标签之和：", targets.sum(dim=1).unique())

    torch.manual_seed(RANDOM_SEED)

    model = ONet(dim=DIM, hidden=HIDDEN_SIZE).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    for step in range(TRAIN_STEPS):

        predictions = model(xs)
        # L1
        #loss = (predictions - targets).abs().mean()
        
        #交叉熵损失
        loss = F.cross_entropy(predictions, indices)

        if step == 0:
            print("训练开始时的损失：", loss.item())

        optimizer.zero_grad() # 清空旧梯度
        loss.backward() # 计算本次梯度
        optimizer.step() # 根据梯度更新参数

        if (step + 1) % LOG_INTERVAL == 0 or step + 1 == TRAIN_STEPS:
            with torch.no_grad():
                check_predictions = model(xs)
                #check_loss = (check_predictions - targets).abs().mean()
                check_loss = F.cross_entropy(check_predictions, indices)

                check_indices = check_predictions.argmax(dim=1)
                check_accuracy = (check_indices == indices).float().mean()

            print(
                "更新次数：", step + 1,
                "损失：", check_loss.item(),
                "准确率：", check_accuracy.item() * 100, "%"
            )
    with torch.no_grad():
        predictions = model(xs)
        predicted_indices = predictions.argmax(dim=1)
        correct = predicted_indices == indices
        accuracy = correct.float().mean()

        probabilities = F.softmax(predictions, dim=1)
        correct_probabilities = probabilities[sample_ids, indices]


    print(
        "正确类别的平均概率：", correct_probabilities.mean().item(),
        "训练后准确率：", accuracy.item() * 100, "%")

    check_sample_ids = [
        0,
        SAMPLES_PER_BIN,
        min(12 * SAMPLES_PER_BIN + 34, len(xs) - 1),
        (DIM // 2) * SAMPLES_PER_BIN,
        len(xs) - 1,
    ]
    for i in check_sample_ids:
        print(
            "输入坐标：", xs[i].item(),
            "正确类别：", indices[i].item(),
            "正确类别概率：", correct_probabilities[i].item(),
            "正确编号：", indices[i].item(),
            "预测编号：", predicted_indices[i].item(),
        )

    project_dir = Path(__file__).resolve().parent
    checkpoint_dir = project_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = (
        checkpoint_dir / f"onet_{DIM}_ce_{TRAIN_STEPS}.pth"
    )
    torch.save(model.state_dict(), checkpoint_path)
    print("模型参数已保存到：", checkpoint_path)

    # model = ONet()
    # x = torch.rand(5, 1) * 63
    # out = model(x)
    # print(out.shape)
