import torch
from render import render_one_dot
from pathlib import Path
from onet import ONet

x1_encoding = torch.tensor([0.0, 0.0, 1.0])
y1_encoding = torch.tensor([0.0, 1.0, 0.0])
x2_encoding = torch.tensor([0.0, 1.0, 0.0])
y2_encoding = torch.tensor([1.0, 0.0, 0.0])

dot1_image = render_one_dot(x1_encoding, y1_encoding)
dot2_image = render_one_dot(x2_encoding, y2_encoding)

combined_image = 1 - (1 - dot1_image) * (1 - dot2_image)
print("两个点合成后的图：\n", combined_image)

print("形状：", dot1_image.shape)
print("点图：\n", dot1_image)
print("第1行第2列：", dot1_image[1, 2])

checkpoint_path = Path(__file__).resolve().parent / "checkpoints" / "onet_64_ce_5000.pth"

loaded_model = ONet(dim=64)
parameters = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
loaded_model.load_state_dict(parameters)
loaded_model.eval()
loaded_model.requires_grad_(False)

x_coordinate = torch.tensor([[12.34]], requires_grad=True)
x1_encoding = loaded_model(x_coordinate).softmax(dim=1).squeeze(0)
y_coordinate = torch.tensor([[32.675]], requires_grad=True)
y1_encoding = loaded_model(y_coordinate).softmax(dim=1).squeeze(0)

neural_dot_image = render_one_dot(x1_encoding, y1_encoding)

print("横坐标概率最大的列编号：", x1_encoding.argmax().item())
print("纵坐标概率最大的行编号：", y1_encoding.argmax().item())
print("单点图形状：", neural_dot_image.shape)
print("第32行第12列的值：", neural_dot_image[32, 12].item())
print("整张图的值之和：", neural_dot_image.sum().item())
print("单点图能否求导：", neural_dot_image.requires_grad)

pixel_value = neural_dot_image[32,12]

print("反向求导前，x 的梯度：", x_coordinate.grad)
print("反向求导前，y 的梯度：", y_coordinate.grad)

pixel_value.backward()

print("反向求导后，x 的梯度：", x_coordinate.grad.item())
print("反向求导后，y 的梯度：", y_coordinate.grad.item())

overlap_x2_encoding = torch.tensor([0.0, 0.0, 1.0])
overlap_y2_encoding = torch.tensor([0.0, 1.0, 0.0])
overlap_dot2_image = render_one_dot(overlap_x2_encoding, overlap_y2_encoding)

same_place_add = dot1_image + overlap_dot2_image
same_place_combine = 1 - (1 - dot1_image) * (1 - overlap_dot2_image)

print("直接相加，重合位置的值：", same_place_add[1, 2].item())
print("按合成公式，重合位置的值：", same_place_combine[1, 2].item())

