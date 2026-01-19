from TGMatcher.data.generate_dataset import BuildDataset
import cv2
import numpy as np
import torch
import os

if __name__ == "__main__":
    # d = BuildDataset(
    #     origin_img_path="./TGMatcher/data/test",
    #     dataset_path="./TGMatcher/data/test/test_pair",
    #     transform_num=2,
    #     img_channel=1,
    #     mode="terrain",
    #     swath_factor=1.4
    # )
    # d.make_pair()
    data = torch.load(
        os.path.join("./TGMatcher/data/test/test_pair/meta", "patch_00108.pt"),
        map_location="cpu",
        weights_only=True,
    )
    print(data["prob_A"].shape)
    print(data["prob_B"].shape)
    print(data["M_AtoB"].shape)
    print(data["mode"])


    # im_A = cv2.imread("./Data/terrain/p1.png", 0)
    # im_B = im_A.copy()
    # h, w = im_A.shape
    # scale = 2
    # M = np.float32([
    #     [scale, 0, (1 - scale) * w / 2], 
    #     [0, scale, (1 - scale) * h / 2]
    # ])
    # im_B = cv2.warpAffine(im_B, M, (w, h), flags=cv2.INTER_LINEAR)
    # prob = cv2.warpAffine(
    #     np.ones_like(im_A, dtype=np.uint8), 
    #     M, 
    #     (w, h), 
    #     flags=cv2.INTER_NEAREST, 
    #     borderValue=0
    # ).astype(np.float32)
    # im_A_tensor = torch.tensor(im_A / 255.0).float()
    # im_B_tensor = torch.tensor(im_B / 255.0).float()
    # im_A_tensor = im_A_tensor.repeat(3, 1, 1)
    # im_B_tensor = im_B_tensor.repeat(3, 1, 1)
    # prob_tensor = torch.tensor(prob).float()
    # M_AtoB_tensor = torch.tensor(M).float()
    # data = {
    #     "im_A": im_A_tensor,
    #     "im_B": im_B_tensor,
    #     "M_AtoB": M_AtoB_tensor,
    #     "prob": prob_tensor,
    # }
    # torch.save(data, "./Data/terrain/p1.pt")