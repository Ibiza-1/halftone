import torch
from torch import nn
import torch.nn.functional as F



class PNet(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        # H 和 C 各占一个输入通道， 输入通道数为2
        # 将他们转换为 channels 个特征通道
        self.input_conv = nn.Conv2d(
            in_channels=2,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )

        self.feature_extractor = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.downsample = nn.Sequential(
            # 第一次 64×64 → 32×32 
            nn.Conv2d(
                channels, channels,
                kernel_size=3, stride=2, padding=1,
            ),
            nn.ReLU(),

            # 第二次 32×32 → 16×16
            nn.Conv2d(
                channels, channels,
                kernel_size=3, stride=2, padding=1,
            ),
            nn.ReLU(),
        )

        # 位移预测头：把64个特征通道转换成2个方向的位移
        self.motion_head = nn.Sequential(
            # 在上采样后的特征上继续做一次局部处理
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),

            # 输出两个通道：通道0为Δx，通道1为Δy
            nn.Conv2d(channels, 2, kernel_size=3, padding=1)
        )

    def forward(self, halftone, contone):
        # B为一次输入P-Net的图像张数
        # 两个输入目前都约定为（B， 1， 64， 64）
        # dim = 1 是通道维，拼接后成为（B， 2， 64， 64）
        combined = torch.cat((halftone, contone), dim=1)

        # 卷积后得到（B， channels， 64， 64）的中间特征
        # 提取特征
        feature = self.input_conv(combined)
        feature = self.feature_extractor(feature)

        # 下采样 （B， 64， 16， 16）
        feature = self.downsample(feature)

        # 恢复到输入图像的高和宽，通道数保持不变
        feature = F.interpolate(
            feature,
            size=halftone.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


        # 从64个特征通道生成2个方向的位移
        motion = self.motion_head(feature)
        return motion


def move_dots(coords, motion):
    # 当前版本处理一张图
    # coords：（N，2），每行是连续坐标[x, y]
    # motion：（1，2， height， width)
    if motion.ndim != 4 or motion.shape[:2] != (1,2):
        raise ValueError("motion 的形状必须是（1， 2， height， width）")

    height, width = motion.shape[-2:]

    # 连续坐标转换为查表用的整数下标
    columns = coords[:, 0].floor().long().clamp(0, width - 1)
    rows = coords[:, 1].floor().long().clamp(0, height - 1)

    # 为每个墨点取出对应位置的横向、纵向位移
    delta_x = motion[0, 0, rows, columns]
    delta_y = motion[0, 1, rows, columns]

    # 两个（N，）向量组合成（N，2），每行是[Δx, Δy]
    dot_motion = torch.stack((delta_x, delta_y), dim=1)

    # 使用原来的连续坐标加位移，保留小数部分
    moved_coords = coords + dot_motion

    # 当前采用边界截断策略，保证新坐标留在图像范围内
    new_x = moved_coords[:, 0].clamp(0, width - 1)
    new_y = moved_coords[:, 1].clamp(0, height - 1)
    return torch.stack((new_x, new_y), dim=1)