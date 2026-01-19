import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from TGMatcher.loss.robust_loss import Robust_Loss
from TGMatcher.model.model import tgm_model
from TGMatcher.data.generate_dataset import TGM_Dataset
from TGMatcher.utils.exp_utils import plot_epoch_loss, plot_step_loss


def set_optim(model):
    parameters = [
        {
            "params": model.encoder.parameters(), 
            "lr": 5e-6,
        },
        {
            "params": model.decoder.parameters(),
            "lr": 1e-4,
        },
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=0.01)
    return optimizer

def set_loader_loss(dataset, batch_size, num_workers):
    loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    loss = Robust_Loss(
        ce_weight=0.01,
        local_dist={1:4, 2:4, 4:8}, # 8:8
        local_largest_scale=4,
        alpha=0.5,
        c = 1e-4,
    )
    return loader, loss

def collate_fn(batch_list):
    batch = {}
    for key in batch_list[0]:
        if isinstance(batch_list[0][key], str):
            batch[key] = [d[key] for d in batch_list]
            continue
        elif isinstance(batch_list[0][key], torch.Tensor):
            batch[key] = torch.stack([d[key] for d in batch_list])
            continue
        else:
            raise TypeError(f"Unsupported type for key {key}: {type(batch_list[0][key])}")
    return batch


def train(
    device, 
    model, 
    dataset, 
    batch_size, 
    num_works, 
    epoches, 
    save_dir=None, 
    decay_factor_1=0.85, 
    decay_factor_2=0.9
):
    optimizer = set_optim(model)
    dataloader, robust_loss = set_loader_loss(dataset, batch_size, num_works)
    scaler = torch.amp.GradScaler("cuda")
    assert save_dir is not None, "ckpt save_dir should not be None"
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        current_time = datetime.now()
        time_str = current_time.strftime("%Y-%m-%d %H-%M-%S")
        current_path = os.path.join(save_dir, time_str)
        os.makedirs(current_path, exist_ok=True)

    min_loss = 1e9
    epoch_avg_losses = []
    for epoch in range(epoches):
        all_step_losses = []
        epoch_loss = 0.0
        if 3 < epoch < 8:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= decay_factor_1
        if 8<= epoch:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= decay_factor_2
        loop = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        for batch in loop:
            model.train(True)
            loss = train_step(
                device=device,
                optimizer=optimizer,
                model=model,
                scaler=scaler,
                batch=batch,
                loss=robust_loss,
            )
            epoch_loss += loss
            all_step_losses.append(loss)
            loop.set_postfix(loss=loss)
        avg_loss = epoch_loss / len(dataloader)
        epoch_avg_losses.append(avg_loss)
        print(f"Epoch {epoch} finished. Avg loss = {avg_loss:.4f}")
        if avg_loss < min_loss:
            min_loss = avg_loss
            save_path = os.path.join(current_path, "tgm.pth")
            torch.save(model.state_dict(), save_path)
        plot_step_loss(
            all_step_losses, 
            avg_loss, 
            save_path=os.path.join(current_path, f"epoch{epoch}_step_loss.svg"), 
            smooth=True
        )
    print(f"Saved checkpoint to {current_path}\n")
    if epoches > 4:
        plot_epoch_loss(
            epoch_avg_losses, 
            save_path=os.path.join(current_path, "epoch_avg_loss.svg")
        )


def train_step(device, optimizer, model, scaler, batch, loss, grad_clip_norm=1.):
    optimizer.zero_grad()
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    out = model(batch)
    l = loss(out, batch)
    scaler.scale(l).backward()
    # 训练不稳定时裁剪梯度
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    scaler.step(optimizer)
    scaler.update()
    return l.item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "./TGMatcher/checkpoints"
    # data_path = "./Data/terrain/overfit"
    data_path = r"/home/wenzhoushen/datasets/tgm_train_dataset"
    dataset = TGM_Dataset(data_path)
    model = tgm_model(
        resolution=256,
        upsample_preds=True,
        device=device,
        encoder_pretrained=True,
        use_custom_corr=True,
        upsample_res=256,
    )
    train(
        device=device,
        model=model,
        dataset=dataset,
        batch_size=4,
        num_works=2,
        epoches=12,
        decay_factor_1=0.88,
        decay_factor_2=0.95,
        save_dir=save_dir
    )
