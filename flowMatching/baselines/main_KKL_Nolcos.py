import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import seaborn as sns
import argparse, os, time

from dataset import datasets
from utils import set_seed, Train_Stats
from models import KKL_Latent_Dynamics, create_mlp


# %% --- config ---

parser = argparse.ArgumentParser(description="Baseline: Set-Valued KKL (Bernard's approach)")
# data
parser.add_argument('--dataset', type=str, default='Test')
parser.add_argument('--noise_std', type=float, default=.0)
parser.add_argument('--process_std', type=float, default=.0)
parser.add_argument('--traj_len', type=int, default=1000*2)
# model
parser.add_argument('--name', type=str, default='baselineNolcos')
parser.add_argument('--z_dim', type=int, default=6)
# transient
parser.add_argument('--transient_len', type=int, default=400)
# train
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3*2)
parser.add_argument('--batchsize', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)


# %% --- init datasets ---

KKLDataset = datasets[args.dataset]
x_dim, y_dim = KKLDataset.x_dim, KKLDataset.y_dim
dt = KKLDataset.dt

train_dataset = KKLDataset(n_trajs=1000*2, traj_len=args.traj_len, noise_std=args.noise_std)
train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)

valid_dataset = KKLDataset(n_trajs=train_dataset.n_trajs//10, traj_len=train_dataset.traj_len, noise_std=args.noise_std)
valid_loader = DataLoader(valid_dataset, batch_size=args.batchsize)


# %% --- init model T: x -> z ---

model = create_mlp(input_dim=x_dim, output_dim=args.z_dim, net_arch=[128, 128, 128]).to(device)
latent_dyn = KKL_Latent_Dynamics(z_dim=args.z_dim, dt=dt, device=device)

def trained_folder():
    return os.path.join("trained_models", args.dataset)

def path(name):
    folder = trained_folder()
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name + ".pth")
    
def save(model):
    print("\t save model", path(args.name))
    checkpoint = {'model': model.to('cpu').state_dict()}
    torch.save(checkpoint, path(args.name))
    model.to(device)

def load(model):
    print("\t load model", path(args.name))
    checkpoint = torch.load(path(args.name))
    model.load_state_dict(checkpoint['model'])
    model.to(device)

if args.epochs == 0:
    load(model)


# %% --- training ---

if args.epochs > 0:
    print("==== training T: x -> z ====")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs//4), gamma=.5)
    stats = Train_Stats({'T_mse'}, trained_folder(), args.name)

    def get_loss(model, ts, xs, ys):
        # skip transient
        xs, ys = xs[:, args.transient_len:], ys[:, args.transient_len:]
        
        xs, ys = xs.to(device), ys.to(device)
        zs_target = latent_dyn.compute_z_fast(ys)
        zs_pred = model(xs)
        return torch.nn.functional.mse_loss(zs_pred, zs_target)

    tic = time.time()
    for epoch in range(args.epochs):
        model.train()
        for ts, xs, ys in train_loader:
            loss = get_loss(model, ts, xs, ys)        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            stats.batch(T_mse=loss)
        stats.epoch(epoch, 'train', time=time.time() - tic, lr=scheduler.get_last_lr()[0])
        
        model.eval()
        with torch.no_grad():
            for ts, xs, ys in valid_loader:
                loss = get_loss(model, ts, xs, ys)
                stats.batch(T_mse=loss)
        stats.epoch(epoch, 'valid')
        
        if (epoch > 0 and epoch % 20 == 0) or (epoch == args.epochs - 1):
            stats.render({'T_mse'}, save=True)
        
        if np.argmin(stats.valid['T_mse']) == epoch:
            save(model)
        
        stats.save()
        scheduler.step()


# %% --- test & inference ([Nolcos, Eq 10] with Warm Start) ---
# ---- REMARKS : 
#               - very sensitive to X_radius

traj_len = 400
n_opt = 100

load(model)
model.eval()

# data
test_dataset = KKLDataset(n_trajs=10, traj_len=traj_len, noise_std=args.noise_std)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
ts, xs, ys = (_.to(device) for _ in next(iter(test_loader)))
zs = latent_dyn.compute_z_fast(ys)



from scipy.optimize import minimize

def warm_start_observer(z_traj, x_init, n_opt=20, X_radius=2.0):
    """
    Inversion locale via scipy.optimize.minimize (L-BFGS-B).
    Utilise le gradient exact fourni par PyTorch (autograd).
    """
    T_len = z_traj.shape[0]
    x_dim = x_init.shape[0]
    xs_obs = torch.zeros(T_len, x_dim).to(device)
    
    current_x_np = x_init.detach().cpu().numpy().astype(np.float64)
    
    bnds = [(-X_radius, X_radius)] * x_dim
    
    for t in range(T_len):
        target_z = z_traj[t:t+1] # shape (1, z_dim) sur device
        
        def cost_and_grad(x_np):
            # 1. NumPy -> PyTorch Tensor
            x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
            x_tensor.requires_grad_(True)
            
            # 2. PyTorch model: forward
            z_pred = model(x_tensor)
            loss = torch.sum((z_pred - target_z)**2)
            
            # 3. PyTorch model: backward for Jacobian
            loss.backward()
            
            # 4. PyTorch Tensor -> NumPy
            loss_val = loss.item()
            grad_val = x_tensor.grad.squeeze(0).cpu().numpy().astype(np.float64)
            
            return loss_val, grad_val

        res = minimize(
            fun=cost_and_grad, 
            x0=current_x_np, 
            method='L-BFGS-B',
            jac=True,
            bounds=bnds,
            options={'maxiter': n_opt, 'ftol': 1e-14, 'disp': False}
        )
        
        # warm start update
        current_x_np = res.x
        xs_obs[t] = torch.tensor(current_x_np, dtype=torch.float32, device=device)
        
        if t % 100 == 0:
            print(f"Step {t}/{T_len} | SciPy Msg: {res.message} | Iters: {res.nit}")
            
    return xs_obs

# Inference (one trajectory)
batch_idx = 0
# Initialisation aux vrais modes (pour tester le suivi de branche) =======
xs_obs_0 = warm_start_observer(zs[batch_idx], xs[batch_idx, 0], n_opt=n_opt)
xs_obs_1 = warm_start_observer(zs[batch_idx], -xs[batch_idx, 0]) # branche symétrique

# Plot
# plt.figure(figsize=(10, 6))
# plt.plot(ts[batch_idx].cpu(), xs[batch_idx, :, 0].cpu(), 'k', label='True $x_1$')
# plt.plot(ts[batch_idx].cpu(), xs_obs_0[:, 0].cpu(), 'r--', label='Bernard (Branch 1)')
# plt.plot(ts[batch_idx].cpu(), xs_obs_1[:, 0].cpu(), 'b--', label='Bernard (Branch 2)')
# plt.legend()
# plt.title(f"Set-Valued KKL Baseline: {args.dataset}")
# plt.show()


KKLDataset.render_obs(ts.cpu(), xs.cpu(), xs_obs_0.unsqueeze(0).cpu(), ys.cpu())
KKLDataset.render_obs(ts.cpu(), xs.cpu(), xs_obs_1.unsqueeze(0).cpu(), ys.cpu())