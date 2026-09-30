import torch
import torch.nn.functional as F


def make_gaussian_kernel(kernel_size=11, sigma=1.5):
    # kernel_size 应为正奇数，sigma 应大于0

    # 生成相对于窗口中心的位置：-5, -4, ..., 0, ..., 4, 5
    positions = torch.arange(kernel_size, dtype=torch.float32)
    positions = positions - kernel_size // 2

    # 根据距离计算权重：越靠近中心，权重越大
    weights = torch.exp(-(positions ** 2) / (2 * sigma ** 2))

    # 归一化，让所有权重加起来等于1
    weights = weights / weights.sum()

    # 一维权重做外积，得到二维高斯核
    kernel = torch.outer(weights, weights)

    # 整理为卷积要求的形状：输出通道、输入通道、高、宽
    return kernel.reshape(1, 1, kernel_size, kernel_size)


def gaussian_smooth(image, kernel):
    # image 的形状约定为 (B, 1, 高, 宽)
    # 让高斯核与图像使用相同的设备和数据类型
    kernel = kernel.to(device=image.device, dtype=image.dtype)

    # 11×11的核，需要在四周各补5个位置，才能保持输出尺寸
    padding = kernel.shape[-1] // 2

    # 按边缘内容镜像填充；参数顺序是左、右、上、下
    padded_image = F.pad(
        image,
        (padding, padding, padding, padding),
        mode="reflect",
    )

    # 使用给定的固定高斯核进行卷积
    return F.conv2d(padded_image, kernel)

def tone_loss(halftone, contone, kernel):
    # 两张图必须同形状，并使用相同的黑白数值约定
    if halftone.shape != contone.shape:
        raise ValueError("半调图和目标图的形状必须相同")

    # 使用同一个高斯核，分别平滑两张图
    smooth_halftone = gaussian_smooth(halftone, kernel)
    smooth_contone = gaussian_smooth(contone, kernel)

    # 计算对应像素的差值
    difference = smooth_halftone - smooth_contone

    # 每张图展开成一个向量，但保留第0维的图像数量
    difference = difference.flatten(start_dim=1)

    # 每张图计算L2范数，再对这一批图像取平均
    per_image_loss = torch.linalg.vector_norm(difference, ord=2, dim=1)
    return per_image_loss.mean()