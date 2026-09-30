from pathlib import Path

import numpy as np
import torch
from PIL import Image


from onet import ONet
from render import render_dots
from losses import make_gaussian_kernel, tone_loss
from pnet import PNet, move_dots

# 参数集中在 config.py，训练和推理使用相同的模型结构。
from config import (
    IMAGE_SIZE, ONET_HIDDEN_SIZE, PREVIEW_SIZE, DEVICE,
    PNET_CHANNELS, PNET_MODE, PNET_TRAIN_STEPS as TRAIN_STEPS,
    PNET_LEARNING_RATE as LEARNING_RATE, PNET_LOG_INTERVAL as LOG_INTERVAL,
    PNET_RANDOM_SEED as RANDOM_SEED, RENDER_CHUNK_SIZE, NUM_DOTS,
    GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA, INPUT_IMAGE, OUTPUT_DIR,
    resolve_device, validate_config, onet_checkpoint_path, pnet_checkpoint_path,
)


if __name__ == "__main__":
    validate_config()
    device = resolve_device(DEVICE)
    print("使用训练设备：", device)

    # 图片路径，以当前脚本所在目录为起点
    image_path = INPUT_IMAGE

    # 打开图片，转换为单通道灰度图，再调整到训练分辨率
    with Image.open(image_path) as image:
        gray_image = image.convert("L")
        gray_image = gray_image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            resample=Image.Resampling.LANCZOS,
        )

    # 图片 → NumPy数组；把0～255转换成0～1
    gray_array = np.array(gray_image, dtype=np.float32) / 255.0

    # NumPy数组 → PyTorch张量，形状为 (IMAGE_SIZE, IMAGE_SIZE)
    brightness = torch.from_numpy(gray_array)

    # 与渲染器统一：0表示无墨，1表示有墨
    contone = 1.0 - brightness

    # 增加“图像数量”和“通道”两个维度
    contone = contone.unsqueeze(0).unsqueeze(0)

    # 把目标图搬到选定设备上
    contone = contone.to(device)

    print("目标图形状：", contone.shape)
    print("目标墨量范围：", contone.min().item(), contone.max().item())

    # 固定随机种子，方便重复实验
    torch.manual_seed(RANDOM_SEED)

    # 根据目标图总墨量，估计初始墨点数量
    total_ink = contone.sum().item()
    num_dots = max(1, round(total_ink)) if NUM_DOTS is None else NUM_DOTS

    # 每行是一对连续坐标 [x, y]
    coords = torch.rand(num_dots, 2) * (IMAGE_SIZE - 1)
    coords = coords.to(device)

    # 加载之前训练好的坐标编码网络
    checkpoint_path = onet_checkpoint_path()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"没有找到对应的 ONet 权重：{checkpoint_path}\n"
            "如果修改了 IMAGE_SIZE，请先训练相同 DIM 的 ONet。"
        )

    onet = ONet(dim=IMAGE_SIZE, hidden=ONET_HIDDEN_SIZE).to(device)
    parameters = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    onet.load_state_dict(parameters)

    # O-Net用于提供编码，后面训练时不更新它的参数
    onet.eval()
    onet.requires_grad_(False)

    # 创建固定的高斯核，后面计算损失时重复使用
    gaussian_kernel = make_gaussian_kernel(
        kernel_size=GAUSSIAN_KERNEL_SIZE,
        sigma=GAUSSIAN_SIGMA,
    ).to(device)

    # 渲染固定初始点集，并记录移动之前的损失
    with torch.no_grad():
        initial_halftone = render_dots(
            coords, onet,
            size=IMAGE_SIZE,
            chunk_size=RENDER_CHUNK_SIZE,
        )

        # (S,S) → (1,1,S,S)，匹配P-Net和损失函数的输入
        initial_halftone = initial_halftone.unsqueeze(0).unsqueeze(0)

        initial_loss = tone_loss(
            initial_halftone, contone, gaussian_kernel
        )

    print("初始半调图形状：", initial_halftone.shape)
    print("移动前的色调损失：", initial_loss.item())

    print("墨点数量：", num_dots)
    print("坐标形状：", coords.shape)
    print("前3个墨点：\n", coords[:3])

    # ===== 创建需要训练的 PNet =====
    pnet = PNet(channels=PNET_CHANNELS).to(device)
    if PNET_MODE == "infer":
        # 推理时读取已训练参数，并核对训练时的结构配置。
        saved = torch.load(pnet_checkpoint_path(), map_location=device, weights_only=True)
        if saved["image_size"] != IMAGE_SIZE or saved["channels"] != PNET_CHANNELS:
            raise ValueError("PNet 权重的尺寸或通道数与 config.py 不一致。")
        pnet.load_state_dict(saved["state_dict"])
        pnet.eval()
    else:
        pnet.train()

    # 优化器只负责更新 PNet 的参数，不更新 ONet。
    optimizer = torch.optim.Adam(
        pnet.parameters(), lr=LEARNING_RATE
    )
    # ===== 模型和优化器创建完成 =====

    # ===== 开始训练：按配置更新 PNet =====
    for step in range(1, (TRAIN_STEPS if PNET_MODE == "train" else 0) + 1):
        # 每次更新前，清空上一次计算留下的梯度。
        optimizer.zero_grad()

        # 从同一张初始半调图预测位移。
        motion = pnet(initial_halftone, contone)

        # 将预测的位移应用到原始墨点坐标。
        moved_coords = move_dots(coords, motion)

        # 渲染移动后的墨点，并补上批次和通道维度。
        predicted_halftone = render_dots(
            moved_coords, onet,
            size=IMAGE_SIZE,
            chunk_size=RENDER_CHUNK_SIZE,
        )
        predicted_halftone = predicted_halftone.unsqueeze(0).unsqueeze(0)

        # 计算损失，再求梯度、更新参数。
        loss = tone_loss(predicted_halftone, contone, gaussian_kernel)
        loss.backward()
        optimizer.step()

        # 按配置的日志间隔，重新计算一次当前模型的损失。
        if step % LOG_INTERVAL == 0 or step == TRAIN_STEPS:
            with torch.no_grad():
                updated_motion = pnet(initial_halftone, contone)
                updated_coords = move_dots(coords, updated_motion)

                updated_halftone = render_dots(
                    updated_coords, onet,
                    size=IMAGE_SIZE,
                    chunk_size=RENDER_CHUNK_SIZE,
                )
                updated_halftone = updated_halftone.unsqueeze(0).unsqueeze(0)

                updated_loss = tone_loss(
                    updated_halftone, contone, gaussian_kernel
                )

            print(
                "更新次数：", step,
                "更新后的色调损失：", updated_loss.item()
            )
    # ===== 训练结束；保存权重供之后的推理使用 =====
    if PNET_MODE == "train":
        save_path = pnet_checkpoint_path()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": pnet.state_dict(),
            "image_size": IMAGE_SIZE,
            "channels": PNET_CHANNELS,
        }, save_path)
        print("PNet 权重已保存到：", save_path)

    # 两种模式都用最终模型生成结果；推理模式无需执行训练循环。
    pnet.eval()
    with torch.no_grad():
        updated_motion = pnet(initial_halftone, contone)
        updated_coords = move_dots(coords, updated_motion)
        updated_halftone = render_dots(
            updated_coords, onet, size=IMAGE_SIZE, chunk_size=RENDER_CHUNK_SIZE,
        ).unsqueeze(0).unsqueeze(0)
        updated_loss = tone_loss(updated_halftone, contone, gaussian_kernel)
    print("最终色调损失：", updated_loss.item())


    # ===== 保存目标图、移动前和移动后的渲染图 =====

    # 在项目目录下创建 outputs 文件夹。
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存文件名及其对应的墨量张量。
    images_to_save = {
        "target.png": contone,
        "before.png": initial_halftone,
        "after_soft.png": updated_halftone,
    }

    for filename, ink_image in images_to_save.items():
        # 墨量中 1 表示黑色；保存图片时，亮度中 1 表示白色。
        brightness_image = 1.0 - ink_image.detach()

        # 去掉批次和通道维度：[1, 1, S, S] → [S, S]。
        brightness_image = brightness_image.squeeze(0).squeeze(0)

        # 将亮度限制在 [0, 1]，再转成 [0, 255] 的整数。
        pixel_array = (
            brightness_image
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

        # 放大保存，便于观察；不会增加训练时的真实细节。
        output_image = Image.fromarray(pixel_array)
        preview_image = output_image.resize(
            (PREVIEW_SIZE, PREVIEW_SIZE),
            resample=Image.Resampling.NEAREST,
        )
        preview_image.save(output_dir / filename)

    print("图片已保存到：", output_dir)
    # ===== 保存结束 =====
