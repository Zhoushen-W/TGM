import matplotlib.pyplot as plt
import cv2
import os
import torch
import numpy as np
from argparse import ArgumentParser


def read_2_images(im_A_path, im_B_path):
    """
    专用于tgm模型的match函数, 读取两张图片的地址, 转换成match需要的格式
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--im_A_path", 
        default=im_A_path, 
        type=str
    )
    parser.add_argument(
        "--im_B_path",
        default=im_B_path,
        type=str
    )
    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path 
    return im1_path, im2_path

def plot_matches(img1_path, img2_path, pts1, pts2, mask=None, save_path=None, alpha=0.6, linewidth=0.3, sample_num=None):
    """
    只绘制一张图, 可视化模型输出的匹配点, 可调线宽和透明度
    
    Args:
        img1_path, img2_path: 图片路径
        pts1, pts2: 匹配点坐标 (N,2) numpy 数组
        mask: RANSAC 内点标记 (可选)
        save_path: 保存路径 (可选)
        alpha: 线条透明度
        linewidth: 线条宽度(浮点数)
        sample_num: 抽样绘制点数，避免覆盖太多(可选)
    """
    # 读取图片并转换 RGB
    img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)
    
    # 高度统一拼接
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    new_h = max(h1, h2)
    scale1 = new_h / h1
    scale2 = new_h / h2
    img1_resized = cv2.resize(img1, (int(w1 * scale1), new_h))
    img2_resized = cv2.resize(img2, (int(w2 * scale2), new_h))
    pts1_resized = pts1 * scale1
    pts2_resized = pts2 * scale2
    
    # 拼接图
    out_img = np.hstack([img1_resized, img2_resized])
    
    # 抽样匹配点
    N = len(pts1_resized)
    if sample_num is not None and sample_num < N:
        indices = np.random.choice(N, sample_num, replace=False)
        pts1_resized = pts1_resized[indices]
        pts2_resized = pts2_resized[indices]
        if mask is not None:
            mask = mask[indices]
    
    plt.figure(figsize=(15, 8))
    plt.imshow(out_img)
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts1_resized, pts2_resized)):
        # 根据 mask 决定颜色, 红色透明度降低
        """if mask is not None:
            if mask[i]:
                color = 'green'
                plt.plot([x1, x2 + img1_resized.shape[1]], [y1, y2], color=color, linewidth=linewidth)
            else:
                color = 'red'
                plt.plot([x1, x2 + img1_resized.shape[1]], [y1, y2], color=color, linewidth=linewidth, alpha=alpha)
        else:
            color = 'green'
            plt.plot([x1, x2 + img1_resized.shape[1]], [y1, y2], color=color, linewidth=linewidth)"""

        # 只画绿色
        if mask is not None and mask[i]:
            plt.plot([x1, x2 + img1_resized.shape[1]], [y1, y2], color='green', linewidth=linewidth, alpha=alpha)
        elif mask is None:
            plt.plot([x1, x2 + img1_resized.shape[1]], [y1, y2], color='green', linewidth=linewidth, alpha=alpha)

    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def moving_average(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode="valid")

def plot_step_loss(step_losses, Avg_loss, save_path=None, smooth=False):
    plt.figure(figsize=(10, 4))
    if smooth:
        smoothed = moving_average(step_losses, window=len(step_losses) // 100)
        plt.plot(
            range(len(smoothed)),
            smoothed,
            linewidth=1,
            label="moving avg (100)"
        )
    else:
        plt.plot(step_losses, linewidth=1)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss (Avg_loss = {Avg_loss:.2f})")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def plot_epoch_loss(epoch_losses, save_path=None, smooth=False):
    plt.figure(figsize=(6, 4))
    if smooth:
        smoothed = moving_average(epoch_losses, window=100)
        plt.plot(
            range(len(smoothed)),
            smoothed,
            linewidth=1,
            label="moving avg (100)"
        )
    else:
        plt.plot(epoch_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss (per epoch)")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


def plot_eval_epe_step(step_losses, step_losses2, save_path=None, smooth=False, ylabel=None):
    """
    绘制训练损失曲线，可选添加第二条曲线对比
    
    Args:
        step_losses: 第一条损失曲线
        Avg_loss: 第一条曲线的平均损失
        step_losses2: 第二条损失曲线（可选）
        Avg_loss2: 第二条曲线的平均损失（可选）
        save_path: 保存路径（可选）
        smooth: 是否平滑曲线
    """
    plt.figure(figsize=(10, 4))
    Avg_loss = np.mean(step_losses)
    Avg_loss2 = np.mean(step_losses2)
    if smooth:
        smoothed = moving_average(step_losses, window=len(step_losses) // 100)
        plt.plot(
            range(len(smoothed)),
            smoothed,
            linewidth=1,
            label=f"with certainty (Avg = {Avg_loss:.5f})"
        )
        smoothed2 = moving_average(step_losses2, window=len(step_losses2) // 100)
        plt.plot(
            range(len(smoothed2)),
            smoothed2,
            linewidth=1,
            label=f"without certainty (Avg = {Avg_loss2:.5f})"
        )
    else:
        plt.plot(step_losses, linewidth=1, label=f"Curve 1 (Avg = {Avg_loss:.2f})")
        plt.plot(step_losses2, linewidth=1, label=f"Curve 2 (Avg = {Avg_loss2:.2f})")
    plt.xlabel("Evaluating Step")
    plt.ylabel(f"{ylabel}")
    plt.title(f"{ylabel} Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        img_path = os.path.join(save_path, f"{ylabel}.svg")
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


def visualize_flow_matches( 
        im_A, 
        im_B, 
        flow, 
        prob=None, 
        num_points=300, 
        figsize=(12, 6), 
        seed=0, 
        save_path=None,
        plot_match_points=True,
        plot_prob_A=False
    ): 
    """ 
    Args:
        im_A: torch.Tensor (3, H, W) 
        im_B: torch.Tensor (3, H, W) 
        flow: torch.Tensor (B, 2, H, W) in normalized coords (-1, 1) 
        prob: torch.Tensor (B, H, W) or None 
    """ 
    torch.manual_seed(seed) 
    im_A = im_A.permute(1, 2, 0).cpu().numpy() 
    im_B = im_B.permute(1, 2, 0).cpu().numpy() 
    flow = flow.squeeze(0).permute(1, 2, 0).cpu() 
    H, W, _ = flow.shape 
    if prob is None: 
        prob = torch.ones(H, W, device=flow.device) 
    else:
        prob = prob.cpu() 
        ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij") 
        coords = torch.stack([xs, ys], dim=-1).reshape(-1, 2) # (N, 2) 
        mask = prob.reshape(-1) > 0.99 
        if plot_prob_A:
            mask_img = mask.reshape(H, W).float() 
            plt.imshow(mask_img.cpu(), cmap="gray") 
            plt.title("mask in A coords") 
            plt.show() 

        coords = coords[mask] 
        flow_flat = flow.reshape(-1, 2)[mask] 
        if coords.shape[0] == 0: 
            print("No valid points to visualize.") 
            return 
        idx = torch.randperm(coords.shape[0])[:num_points] 
        coords = coords[idx] 
        flow_flat = flow_flat[idx] 
        xA = coords[:, 0].float() 
        yA = coords[:, 1].float() 
        xB = (flow_flat[:, 0] + 1) * 0.5 * W - 0.5 
        yB = (flow_flat[:, 1] + 1) * 0.5 * H - 0.5 
        fig, ax = plt.subplots(1, 1, figsize=figsize) 
        canvas = np.concatenate([im_A, im_B], axis=1) 
        ax.imshow(canvas) 
        ax.axis("off") 
        offset = W # B image x-offset 
        for i in range(len(xA)): 
            ax.plot( 
                [xA[i], xB[i] + offset], 
                [yA[i], yB[i]], 
                color="lime", 
                linewidth=0.6, 
                alpha=0.7, 
            )
            if plot_match_points:
                ax.scatter(xA[i], yA[i], s=3, color="red") 
                ax.scatter(xB[i] + offset, yB[i], s=3, color="cyan") 
        # plt.title(f"Match Result (Sample Points: {num_points})")
        plt.tight_layout() 
        if save_path: 
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0) 
            plt.close() 
        else: 
            plt.show()



if __name__ == "__main__":
    a = [np.random.uniform(0.2, 5) for a in range(5000)]
    plot_step_loss(a, 0, smooth=True)