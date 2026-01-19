import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from TGMatcher.model.model import tgm_model
from TGMatcher.utils.exp_utils import visualize_flow_matches, plot_eval_epe_step
from TGMatcher.utils.warp import get_warp, scale_affine
from TGMatcher.utils.prob import compute_HM
from TGMatcher.data.generate_dataset import TGM_Dataset
from TGMatcher.train.train_tgm import collate_fn


def eval_setting(device):
    test_time = "2026-01-19 16-57-44"
    weights = torch.load(
        f=f"./TGMatcher/checkpoints/{test_time}/tgm.pth", 
        map_location="cpu", 
        weights_only=True
    )
    model = tgm_model(
        resolution=256,
        upsample_preds=True,
        device=device,
        encoder_pretrained=False,
        tgm_weight=weights,
        use_custom_corr=True,
        upsample_res=256,
    )
    return model

def eval_img(data_path, model, device, save_path=None, with_certainty=True):
    model.eval()
    dataset = TGM_Dataset(data_path)
    sample = 5218
    batch = dataset[sample]
    # batch2 = TGM_Dataset("./Experiment/localization/beam_strip_patch")[4]
    batch_device = {
        "im_A": batch["im_A"].unsqueeze(0).to(device),
        "im_B": batch["im_B"].unsqueeze(0).to(device),
        "prob": batch["prob_A"].unsqueeze(0).to(device),
        "M_AtoB": batch["M_AtoB"].unsqueeze(0).to(device)
    }
    with torch.no_grad():
        scale = 1
        corresps = model(batch_device)
        flow = corresps[scale]["flow"] # [b, c, h, w]
        warp = get_warp(
            M_AtoB=batch_device["M_AtoB"],
            H=256,
            W=256,
        )
        if with_certainty:
            certainty = corresps[scale]["certainty"][:, 0, :, :] # [b, h, w]
            if scale == 1:
                ws, cs = warp, certainty
            else:
                ws, cs = scale_affine(warp, certainty, scale)
            epe = (flow.permute(0, 2, 3, 1) - ws).norm(dim=-1).detach()[cs > 0.99]
        else:
            prob = batch_device["prob"]
            if scale == 1:
                ws, ps = warp, prob
            else:
                ws, ps = scale_affine(warp, prob, scale)
            epe = (flow.permute(0, 2, 3, 1) - ws).norm(dim=-1).detach()[ps > 0.99]

    count_current = ((certainty > 0.99)).sum().item()
    print(count_current)
    assert save_path is not None, "save_path should not be None on linux platform"
    visualize_flow_matches(
        im_A=batch["im_A"],
        im_B=batch["im_B"],
        prob=certainty.squeeze(0) if with_certainty else batch["prob_A"],
        flow=flow,
        num_points=50,
        # plot_match_points=False,
        plot_prob_A=False,
        save_path=os.path.join(save_path, f"{dataset.names[sample]}_{with_certainty}_eval.svg")
    )
    print(f"EPE_{scale} stats:")
    print("mean:", epe.mean().item())
    print("median:", epe.median().item())
    print("min:", epe.min().item())
    print("max:", epe.max().item())


def eval_batch(data_path, model, device, batch_size, num_workers):
    finest_scale = 1
    dataset = TGM_Dataset(data_path)
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    epe_mean_T, epe_min_T, epe_max_T = [], [], []
    epe_mean_F, epe_min_F, epe_max_F = [], [], []
    loop = tqdm(dataloader, desc="Epoch", leave=False)
    for batch in loop:
        model.eval()
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.no_grad():
            corresps = model(batch)
            flow = corresps[finest_scale]["flow"] # [b, c, h, w]
            certainty = corresps[finest_scale]["certainty"] # [b, c, h, w]
            warp = get_warp(batch["M_AtoB"], 256, 256)
            prob = batch["prob_A"]
            epe_T = (flow.permute(0, 2, 3, 1) - warp).norm(dim=-1).detach()[certainty[:, 0] > 0.99]
            epe_F = (flow.permute(0, 2, 3, 1) - warp).norm(dim=-1).detach()[prob > 0.99]
        epe_mean_T.append(epe_T.mean().item())
        epe_min_T.append(epe_T.min().item())
        epe_max_T.append(epe_T.max().item())
        epe_mean_F.append(epe_F.mean().item())
        epe_min_F.append(epe_F.min().item())
        epe_max_F.append(epe_F.max().item())
    
    save_path = "./TGMatcher/train/eval_result"
    plot_eval_epe_step(epe_mean_T, epe_mean_F, save_path, smooth=True, ylabel="EPE_mean")
    plot_eval_epe_step(epe_min_T, epe_min_F, save_path, smooth=True, ylabel="EPE_min")
    plot_eval_epe_step(epe_max_T, epe_max_F, save_path, smooth=True, ylabel="EPE_max")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with_certainty = True
    # eval_img(
    #     data_path=r"/home/wenzhoushen/datasets/tgm_eval_dataset",
    #     model=eval_setting(device),
    #     device=device,
    #     save_path="./TGMatcher/train/eval_result",
    #     with_certainty=with_certainty
    # )
    eval_batch(
        data_path=r"/home/wenzhoushen/datasets/tgm_eval_dataset",
        model=eval_setting(device),
        device=device,
        batch_size=4,
        num_workers=2
    )