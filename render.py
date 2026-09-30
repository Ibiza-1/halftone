import torch

def render_one_dot(x_encoding, y_encoding):
    dot_image = torch.outer(y_encoding, x_encoding)
    return dot_image

# coords: 所有墨点的坐标，形状为 (N, 2)
#         每一行是 [x, y]，N 是墨点数量
# onet:   已加载的坐标编码网络
# 返回值: 一张形状为 (size, size) 的可求导图像
def render_dots(coords, onet, size=64, chunk_size=256):
    # 每个点有 x、y 两个坐标
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords 的形状必须是 (N, 2)")

    # 从“所有像素都未被墨点覆盖”开始。
    # new_ones 会沿用 coords 的数据类型和计算设备（CPU/GPU）。
    uncovered = coords.new_ones((size, size))

    # 一次只处理 chunk_size 个点，避免同时创建所有点的二维图
    for chunk in coords.split(chunk_size, dim=0):
        # 0:1 取所有点的 x，并保留形状 (本批点数, 1)，供 O-Net 输入
        # softmax 将交叉熵版 O-Net 输出的 logits 转成 size 个位置的概率
        x_encoding = onet(chunk[:, 0:1]).softmax(dim=1)

        # 1:2 同理取 y；它对应输出图像的“行”
        y_encoding = onet(chunk[:, 1:2]).softmax(dim=1)

        # 为本批每个点生成一张 (size, size) 的图。
        # [:, :, None] 变成 (本批点数, size, 1)
        # [:, None, :] 变成 (本批点数, 1, size)
        # 相乘后得到 (本批点数, size, size)
        dot_images = y_encoding[:, :, None] * x_encoding[:, None, :]

        # 1 - dot_images 表示各点“未覆盖”的程度。
        # prod(dim=0) 沿着点的维度连乘，结果是一张 (size, size) 的图。
        # 再乘上先前批次的结果，合并所有已经处理的点。
        uncovered = uncovered * (1 - dot_images).prod(dim=0)

    # “至少有一个点覆盖” = 1 - “所有点都未覆盖”
    # 返回的图仍连接着 coords 的求导路径。
    return 1 - uncovered
