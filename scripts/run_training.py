#!/usr/bin/env python3
"""Full training script for 3D gravity-magnetic joint inversion."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np, yaml
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader

from src.utils import set_seed, save_checkpoint, count_parameters

def make_loader(data, batch_size, shuffle=True):
    g = torch.from_numpy(data['gravity']).float()
    m = torch.from_numpy(data['magnetic']).float()
    g3 = g.unsqueeze(-1).repeat(1, 1, 1, 20)
    m3 = m.unsqueeze(-1).repeat(1, 1, 1, 20)
    inp = torch.stack([g3, m3], dim=1)
    r = torch.from_numpy(data['rho']).float()
    k = torch.from_numpy(data['kappa']).float()
    s = torch.from_numpy(data['sim']).float()
    return DataLoader(TensorDataset(inp, r, k, s), batch_size=batch_size,
                       shuffle=shuffle, drop_last=True, num_workers=0)

def main():
    with open('configs/full.yaml') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['training']['seed'])
    device = torch.device('cuda')
    print(f'=== FULL TRAINING START ===')
    print(f'{datetime.now().isoformat()}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

    train_data = np.load('data/train_dataset.npz')
    val_data = np.load('data/val_dataset.npz')
    train_loader = make_loader(train_data, cfg['data']['batch_size'], shuffle=True)
    val_loader = make_loader(val_data, cfg['data']['batch_size'], shuffle=False)
    n_epochs = cfg['training']['epochs']
    print(f'Train: {len(train_data["rho"])} samples, {len(train_loader)} batches')
    print(f'Val: {len(val_data["rho"])} samples, {len(val_loader)} batches')
    print(f'Epochs: {n_epochs}')

    from src.model.joint_inversion_net import JointInversionNet
    from src.model.loss_functions import MultiTaskLoss
    model = JointInversionNet().to(device)
    nparams = count_parameters(model)
    print(f'Params: {nparams["total"]:,} (trainable: {nparams["trainable"]:,})')
    criterion = MultiTaskLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'],
                                  weight_decay=cfg['training'].get('weight_decay', 0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    t_start = time.time()

    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()
        tloss, nb = 0.0, 0
        for xb, rb, kb, sb in train_loader:
            xb, rb, kb, sb = [t.to(device) for t in (xb, rb, kb, sb)]
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(xb, return_all=True)
                total_loss, _ = criterion(out, {'rho': rb, 'kappa': kb, 'sim': sb})
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            tloss += total_loss.item()
            nb += 1
        train_loss = tloss / max(nb, 1)
        scheduler.step()

        # Validate
        model.train()
        vloss, vb = 0.0, 0
        with torch.no_grad():
            for xb, rb, kb, sb in val_loader:
                xb, rb, kb, sb = [t.to(device) for t in (xb, rb, kb, sb)]
                with torch.amp.autocast('cuda'):
                    out = model(xb, return_all=True)
                    vl, _ = criterion(out, {'rho': rb, 'kappa': kb, 'sim': sb})
                vloss += vl.item()
                vb += 1
        val_loss = vloss / max(vb, 1)

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['lr'].append(float(scheduler.get_last_lr()[0]))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, float(val_loss), 'checkpoints/best_model.pt')

        epoch_time = time.time() - t0
        lr_val = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch:3d}/{n_epochs} | train={train_loss:.4f} | '
              f'val={val_loss:.4f} | lr={lr_val:.6f} | '
              f'{epoch_time:.1f}s | best={best_val_loss:.4f}', flush=True)

    save_checkpoint(model, optimizer, n_epochs - 1, float(val_loss), 'checkpoints/final_model.pt')
    history['config'] = str(cfg)
    history['timestamp'] = datetime.now().isoformat()
    history['total_time_s'] = time.time() - t_start
    with open('results/training_history.json', 'w') as f:
        json.dump(history, f, indent=2, default=float)
    print(f'\n=== TRAINING COMPLETE ({time.time()-t_start:.0f}s) ===')
    print(f'Best val loss: {best_val_loss:.4f}')
    print(f'Checkpoints saved.')

if __name__ == '__main__':
    main()
