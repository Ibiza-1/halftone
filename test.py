import torch


coords = torch.tensor([
    [12.34, 32.675],  # 第0个点：x=12.34，y=32.675
    [ 5.20, 10.100],  # 第1个点
    [40.00, 20.500],  # 第2个点
    [ 8.60, 50.200],  # 第3个点
])

print(coords.shape)
print(coords[0,1].item())