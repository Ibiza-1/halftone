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

    test_coordinate = torch.tensor([[32.675]], requires_grad=True)
    probability = loaded_model(test_coordinate).softmax(dim=1)[0,32]

    print("类别32的概率：", probability.item())


    print("反向求导前的坐标梯度：", test_coordinate.grad)

    probability.backward()

    print("反向求导后的坐标梯度：", test_coordinate.grad.item())

    with torch.no_grad():
        probability_no_grad = loaded_model(test_coordinate).softmax(dim=1)[0, 32]

    print("原先的 probability 能否求导：", probability.requires_grad)
    print("新算的 probability_no_grad 能否求导：", probability_no_grad.requires_grad)
    print("新算的类别32概率：", probability_no_grad.item())

    delta = 0.001

    with torch.no_grad():
        p_left = loaded_model(torch.tensor([[32.675 - delta]])).softmax(dim=1)[0, 32]
        p_right = loaded_model(torch.tensor([[32.675 + delta]])).softmax(dim=1)[0, 32]

    estimated_grad = (p_right - p_left) / (2 * delta)
    print("用左右概率估计的梯度：", estimated_grad.item())


