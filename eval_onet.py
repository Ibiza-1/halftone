"""评估 config.py 指定的 ONet，尺寸、隐藏层宽度和设备与训练保持一致。"""
import torch
from onet import ONet
from config import (
    IMAGE_SIZE, ONET_HIDDEN_SIZE, ONET_EVAL_SAMPLES_PER_BIN,
    DEVICE, resolve_device, validate_config, onet_checkpoint_path,
)


if __name__ == "__main__":
    validate_config()
    device = resolve_device(DEVICE)
    loaded_model = ONet(dim=IMAGE_SIZE, hidden=ONET_HIDDEN_SIZE).to(device)
    parameters = torch.load(
        onet_checkpoint_path(), map_location=device, weights_only=True,
    )
    loaded_model.load_state_dict(parameters)
    loaded_model.eval()

    # 每个区间均匀取点，并偏移半个采样间距，避开原先的训练网格点。
    count = ONET_EVAL_SAMPLES_PER_BIN
    test_ids = torch.arange(IMAGE_SIZE * count, device=device)
    xs = ((test_ids.float() + 0.5) / count).unsqueeze(1)
    labels = xs.squeeze(1).floor().long()
    with torch.no_grad():
        probabilities = loaded_model(xs).softmax(dim=1)
        predicted_labels = probabilities.argmax(dim=1)
    correct = predicted_labels == labels
    print("测试坐标数量：", len(xs))
    print("准确率：", correct.float().mean().item() * 100, "%")
    print("正确类别平均概率：", probabilities[test_ids, labels].mean().item())

    # 按相对位置选取示例，避免尺寸变化后固定下标越界。
    for i in sorted({0, len(xs) // 4, len(xs) // 2, len(xs) - 1}):
        print("坐标：", xs[i].item(), "正确类别：", labels[i].item(),
              "预测类别：", predicted_labels[i].item(),
              "正确类别概率：", probabilities[i, labels[i]].item())

    fraction = xs.squeeze(1) - labels.float()
    near_boundary = (fraction < 0.1) | (fraction > 0.9)
    for name, mask in [("边界附近", near_boundary), ("其他位置", ~near_boundary)]:
        if mask.any():
            print(name + "准确率：", correct[mask].float().mean().item() * 100, "%")
        else:
            print(name + "：当前采样设置下没有样本")

    # 选择图像中部的整数边界；任意 IMAGE_SIZE >= 2 都可使用。
    boundary = IMAGE_SIZE // 2
    boundary_xs = boundary + torch.tensor(
        [[-0.05], [-0.005], [0.005], [0.05]], device=device,
    )
    with torch.no_grad():
        boundary_probabilities = loaded_model(boundary_xs).softmax(dim=1)
    for i in range(len(boundary_xs)):
        print("边界坐标：", boundary_xs[i].item(),
              f"类别{boundary - 1}概率：", boundary_probabilities[i, boundary - 1].item(),
              f"类别{boundary}概率：", boundary_probabilities[i, boundary].item(),
              "预测类别：", boundary_probabilities[i].argmax().item())
