import torch
from pathlib import Path


from onet import ONet



if __name__ == "__main__":
    checkpoint_path = (
        Path(__file__).resolve().parent
        / "checkpoints"
        / "onet_64_ce_5000.pth"
    )


    loaded_model = ONet(dim=64)


    parameters = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )   
    loaded_model.load_state_dict(parameters)
    loaded_model.eval()


    test_ids = torch.arange(64 * 100)
    xs = (test_ids.float() / 100 + 0.005).unsqueeze(1)
    print("xs的形状：", xs.shape)
    print("第1号样本：", xs[1])


    labels = xs.squeeze(1).floor().long()

    with torch.no_grad():
        probabilities = loaded_model(xs).softmax(dim=1)
        predicted_labels = probabilities.argmax(dim=1)

    for i in [0, 1234, 3267, 6399]:
        print(
            "坐标：", xs[i].item(),
            "正确类别：", labels[i].item(),
            "预测类别：", predicted_labels[i].item(),
            "正确类别概率：", probabilities[i, labels[i]].item(),
        )

    
    correct = predicted_labels == labels
    accuracy = correct.float().mean()
    print("6400个测试坐标的准确率：", accuracy.item() * 100, "%")
    correct_probabilities = probabilities[test_ids, labels]
    print("正确类别平均概率：", correct_probabilities.mean().item())

    fraction = xs.squeeze(1) - labels.float()
    near_boundary = (fraction < 0.1) | (fraction > 0.9)

    print("边界附近准确率：", correct[near_boundary].float().mean().item() * 100, "%")
    print("其他位置准确率：", correct[~near_boundary].float().mean().item() * 100, "%")

    boundary_xs = torch.tensor([
        [12.95],
        [12.995],
        [13.005],
        [13.05],
    ])
    boundary_labels = boundary_xs.squeeze(1).floor().long()

    with torch.no_grad():
        boundary_probabilities = loaded_model(boundary_xs).softmax(dim=1)

    for i in range(len(boundary_xs)):
        print(
            "坐标：", boundary_xs[i].item(),
            "正确类别：", boundary_labels[i].item(),
            "类别12的概率：", boundary_probabilities[i, 12].item(),
            "类别13的概率：", boundary_probabilities[i, 13].item(),
            "预测类别：", boundary_probabilities[i].argmax().item(),
        )

