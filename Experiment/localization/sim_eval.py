import torch
import os 
import yaml
from tqdm import tqdm

from TGMatcher.train.evaluate_tgm import eval_setting
from Experiment.localization.coordinate import estimate_transform
from Experiment.localization.sim_sample import sample_multibeam_and_dem_patches, visualize_matching_trajectory
from TGMatcher.utils.exp_utils import visualize_flow_matches
from Experiment.localization.sim_data import sim_dataset


def beam_dem_match(patch_save_path, match_result_save_path, model, device, if_visual=False):
    beam_dataset = sim_dataset(patch_save_path, "beam")
    beam_data_lenth = beam_dataset.__len__()
    match_result = {}
    for n in range(beam_data_lenth):
        beam_data = beam_dataset[n]
        valid_mask = beam_data["prob"].unsqueeze(0).to(device)
        dem_dataset = sim_dataset(patch_save_path, "dem", f"{n:04}")
        dem_data_lenth = dem_dataset.__len__()
        count_max, flow_max, cert_max, idx = (0, None, None, None) # init
        for i in tqdm(range(dem_data_lenth)):
            dem_data = dem_dataset[i]
            batch_device = {
                "im_A": dem_data["img"].unsqueeze(0).to(device),
                "im_B": beam_data["img"].unsqueeze(0).to(device)
            }
            model.eval()
            with torch.no_grad():
                corresps = model(batch_device)
                flow = corresps[1]["flow"]
                certainty = corresps[1]["certainty"]
                count_current = ((certainty[:, 0][valid_mask > 0] > 0.99)).sum().item()
                if count_current > count_max:
                    flow_max = flow
                    cert_max = certainty
                    count_max = count_current
                    idx = i
                    coord = dem_data["coord"]
        print(f"The matching result of beam_{n} is {dem_dataset.names[idx]}, with {count_max} high certainty matches")
        tfm = estimate_transform(flow_max, cert_max, device, 1000)
        match_result[n] = {"tfm": tfm, "coord": coord}
        if if_visual:
            os.makedirs(match_result_save_path, exist_ok=True)
            visualize_flow_matches(
                im_A=dem_dataset[idx]["img"], 
                im_B=beam_data["img"],
                flow=flow_max,
                prob=cert_max[:, 0].squeeze(0),
                num_points=100,
                save_path=f"{match_result_save_path}/beam_{n}_matches.svg"
            )
    return match_result, beam_data_lenth


if __name__ == "__main__":
    config_path = "./Experiment/localization/sim_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    path = config["path"]
    sim_para = config["sim_para"]
    dem_path = path.get("dem_path")
    patch_save_path = path.get("patch_save_path")
    match_result_save_path = path.get("match_save_path")
    strip_save_path = path.get("strip_save_path")
    trajectory_save_path = path.get("trajectory_save_path")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strip_shape = sim_para.get("strip_shape")
    new_simulation = sim_para.get("new_simulation")
    is_visual = sim_para.get("is_visual")
    start_coord = sim_para.get("start_coord")
    end_coord = sim_para.get("end_coord")
    delta_x = sim_para.get("delta_x")
    mb_patch_size = sim_para.get("mb_patch_size")
    mb_patch_stride = sim_para.get("mb_patch_stride")
    dem_patch_size = sim_para.get("dem_patch_size")
    dem_subpatch_size = sim_para.get("dem_subpatch_size")
    dem_subpatch_stride = sim_para.get("dem_subpatch_stride")
    N_beams = sim_para.get("N_beams")
    resolution = sim_para.get("resolution")
    swath_factor = sim_para.get("swath_factor")
    noise_ratio = sim_para.get("noise_ratio")

    if new_simulation:
        sample_multibeam_and_dem_patches(
            dem_path=dem_path,
            start_coord=start_coord,
            end_coord=end_coord,
            delta_x=delta_x,
            mb_patch_size=mb_patch_size,
            mb_patch_stride=mb_patch_stride,
            dem_patch_size=dem_patch_size,
            dem_subpatch_size=dem_subpatch_size,
            dem_subpatch_stride=dem_subpatch_stride,
            strip_save_dir=strip_save_path,
            save_dir=patch_save_path,
            N_beams=N_beams,
            resolution=resolution,
            swath_factor=swath_factor,
            noise_ratio=noise_ratio
        )

    match_result, beam_num = beam_dem_match(
        patch_save_path=patch_save_path,
        model=eval_setting(device),
        device=device,
        if_visual=is_visual,
        match_result_save_path=match_result_save_path
    )
    visualize_matching_trajectory(
        dem_path=dem_path,
        start_coord=start_coord,
        end_coord=end_coord,
        match_result=match_result,
        beam_num=beam_num,
        strip_shape=strip_shape,
        save_path=trajectory_save_path,
    )