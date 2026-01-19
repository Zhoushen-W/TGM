import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from TGMatcher.data.generate_dataset import TGM_Dataset
from TGMatcher.utils.prob import compute_HM

"""
这个文件用于绘制构建数据集时的变换和prob可视化
"""

def plot_full_warp(img_full, M_full):
    h, w = img_full.shape
    warped_img_full = cv2.warpAffine(
        img_full,
        M_full,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_full, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].set_xlabel('W (pixel)')
    axes[0].set_ylabel('H (pixel)')

    center_x, center_y = w // 2, h // 2
    half_size = 128  # 256/2
    rect_original = plt.Rectangle(
        (center_x - half_size, center_y - half_size), 
        256, 256, 
        linewidth=2, 
        edgecolor='red', 
        facecolor='none'
    )
    axes[0].add_patch(rect_original)
    # 在原始图像中心点添加标记
    axes[0].plot(center_x, center_y, 'r+', markersize=10)
    axes[1].imshow(warped_img_full, cmap='gray')
    axes[1].set_title('Warped Image')
    axes[1].set_xlabel('W (pixel)')
    axes[1].set_ylabel('H (pixel)')
    rect_warped = plt.Rectangle(
        (center_x - half_size, center_y - half_size), 
        256, 256, 
        linewidth=2, 
        edgecolor='red', 
        facecolor='none'
    )
    axes[1].add_patch(rect_warped)
    # 在变换后的图像中心点添加标记
    axes[1].plot(center_x, center_y, 'r+', markersize=10)
    plt.tight_layout()
    plt.show()

def plot_prob(im_A_np, im_B_np, prob_A_np, prob_B_np, save_path=None, name=None):
    h, w = prob_B_np.shape
    overlay_A = np.zeros((h, w, 4), dtype=np.float32)
    overlay_A[prob_A_np == 0] = [1.0, 0.2, 0.2, 0.15]
    overlay_A[prob_A_np == 1] = [0.2, 1.0, 0.2, 0.15]
    overlay_B = np.zeros((h, w, 4), dtype=np.float32)
    overlay_B[prob_B_np == 0] = [1.0, 0.2, 0.2, 0.15]  # 红色
    overlay_B[prob_B_np == 1] = [0.2, 1.0, 0.2, 0.15]  # 绿色
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(im_A_np)
    axs[0].imshow(overlay_A)
    axs[0].set_title("Image A with probability Mask")
    axs[0].set_xlabel('W (pixel)')
    axs[0].set_ylabel('H (pixel)')
    axs[1].imshow(im_B_np)
    axs[1].imshow(overlay_B)
    axs[1].set_title("Image B with Probability Mask")
    axs[1].set_xlabel('W (pixel)')
    axs[1].set_ylabel('H (pixel)')

    # 添加图例
    red_patch = mpatches.Patch(color='red', alpha=0.3, label='prob = 0')
    green_patch = mpatches.Patch(color='green', alpha=0.3, label='prob = 1')
    axs[1].legend(handles=[red_patch, green_patch], loc='upper right')
    plt.tight_layout()
    if save_path is not None and name is not None:
        os.makedirs(save_path, exist_ok=True)
        img_path = os.path.join(save_path, name)
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_hight_prob_of_imgA(im_A_tensor, prob_A_tensor, im_A_np, save_path=None, name=None):
    h, w, c = im_A_np.shape
    prob_A_tensor_new = compute_HM(
        img=im_A_tensor,
        prob=prob_A_tensor,
        num_bins=32,
        norm_mode="sigmoid",
        a=16
    ).squeeze(0)
    prob_A_np = prob_A_tensor_new.cpu().detach().numpy()
    overlay_A = np.zeros((h, w, 4),dtype=np.float32)
    overlay_A[prob_A_np == 0] = [1.0, 0.2, 0.2, 0.15]
    overlay_A[prob_A_np > 0.99] = [0.2, 1.0, 0.2, 0.15]
    overlay_A[np.logical_and(prob_A_np > 0, prob_A_np <= 0.99)] = [1.0, 0.8, 0.0, 0.15]
    plt.imshow(im_A_np)
    plt.imshow(overlay_A)
    plt.title("Image A with probability Mask")
    plt.xlabel('W (pixel)')
    plt.ylabel('H (pixel)')
    red_patch = mpatches.Patch(color='red', alpha=0.3, label='prob = 0')
    green_patch = mpatches.Patch(color='green', alpha=0.3, label='prob ∈ (0.99, 1)')
    orange_patch = mpatches.Patch(color='orange', alpha=0.3, label='prob ∈ (0, 0.99]')
    plt.legend(handles=[red_patch, orange_patch, green_patch], loc='upper right')
    plt.tight_layout()
    if save_path is not None and name is not None:
        os.makedirs(save_path, exist_ok=True)
        img_path = os.path.join(save_path, name)
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data_path = r"/home/wenzhoushen/datasets/tgm_eval_dataset"
    data_path = "./TGMatcher/data/test/test_pair"
    dataset = TGM_Dataset(data_path)
    sample = 2
    data = dataset[sample]
    im_A = data["im_A"]
    im_B = data["im_B"]
    prob_B = data["prob_B"]
    prob_A = data["prob_A"]
    im_A = im_A.cpu().detach().permute(1, 2, 0) # c, h, w -> h, w, c
    im_B = im_B.cpu().detach().permute(1, 2, 0)
    prob_B_np = prob_B.cpu().detach().numpy() # h, w
    prob_A_np = prob_A.cpu().detach().numpy()
    im_B_np = im_B.numpy()
    im_A_np = im_A.numpy()
    # M_full = data["M_full"].cpu().numpy()
    # plot_full_warp(full_img, M_full[:2, :])
    plot_prob(
        im_A_np, 
        im_B_np, 
        prob_A_np, 
        prob_B_np,
        save_path="./Experiment/plot/figs",
        name=f"{dataset.names[sample]}_prob_A_B.svg"
    )
    plot_hight_prob_of_imgA(
        im_A_tensor=data['im_A'].unsqueeze(0).to(device), 
        prob_A_tensor=data['prob_A'].unsqueeze(0).to(device), 
        im_A_np=im_A_np,
        save_path="./Experiment/plot/figs",
        name=f"{dataset.names[sample]}_prob_A.svg"
    )