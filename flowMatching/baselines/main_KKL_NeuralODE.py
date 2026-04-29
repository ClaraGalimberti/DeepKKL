import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import argparse, os
import time

# --- Vos imports ---
from dataset import datasets, RK4
from utils import set_seed, Train_Stats, Normalizer
from models import KKL_Latent_Dynamics

# %% --- config ---

parser = argparse.ArgumentParser(description="Neural ODE KKL Baseline")
# data
parser.add_argument('--dataset', type=str, default='VDP')
parser.add_argument('--noise_std', type=float, default=.0)
parser.add_argument('--process_std', type=float, default=.0)
parser.add_argument('--traj_len', type=int, default=2000)
# model
parser.add_argument('--name', type=str, default='node_temp')
parser.add_argument('--z_dim', type=int, default=6)
parser.add_argument('--hidden_dim', type=int, default=128, help="Neural ODE hidden state dim")
parser.add_argument('--n_steps', type=int, default=2, help="Number of RK4 integration steps")
# transient
parser.add_argument('--transient_len', type=int, default=400, help="Transient length during training")
parser.add_argument('--transient_skip', action="store_true", help="Transient not in the loss")
# train
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3*2)
parser.add_argument('--batchsize', type=int, default=5, help="/!\ Number of trajectories in a batch.")
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
print(args)

CRITERION = torch.nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(args.seed)

# %% --- init datasets ---

KKLDataset = datasets[args.dataset]
x_dim, y_dim = KKLDataset.x_dim, KKLDataset.y_dim
dt = KKLDataset.dt

train_dataset = KKLDataset(n_trajs=1000*2,
                           traj_len=args.traj_len,
                           noise_std=args.noise_std,
                           process_std=args.process_std)
train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)

valid_dataset = KKLDataset(n_trajs=train_dataset.n_trajs//10,
                           traj_len=train_dataset.traj_len,
                           noise_std=args.noise_std,
                           process_std=args.process_std)
valid_loader = DataLoader(valid_dataset, batch_size=args.batchsize)

# %% --- init model ---

class ODEFunc(nn.Module):
    """ Champ de vecteurs pour la Neural ODE """
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, hidden_dim)
        )

    def forward(self, h, u=None):
        return self.net(h)

class NeuralODE_KKL(nn.Module):
    """ Modèle d'inversion déterministe via Neural ODE """
    def __init__(self, z_dim, x_dim, hidden_dim=128, n_steps=10):
        super().__init__()
        self.proj_in = nn.Linear(z_dim, hidden_dim)
        self.odefunc = ODEFunc(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, x_dim)
        self.n_steps = n_steps

    def forward(self, z):
        # Project : z -> h
        h = self.proj_in(z)
        
        # Integrate (RK4) from tau=0 à tau=1
        dt_ode = 1.0 / self.n_steps
        for _ in range(self.n_steps):
            h = RK4(self.odefunc, dt_ode, h)
            
        # Project : h -> x
        x_hat = self.proj_out(h)
        return x_hat

model = NeuralODE_KKL(z_dim=args.z_dim, x_dim=x_dim, 
                      hidden_dim=args.hidden_dim, n_steps=args.n_steps).to(device)

latent_dyn = KKL_Latent_Dynamics(z_dim=args.z_dim, dt=dt,
                                 noise_std=args.noise_std, device=device)

normalizer = Normalizer(train_dataset.xs, time_period=0)

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
    print("==== training ====")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs//4), gamma=.5)
    stats = Train_Stats({'node_mse'}, trained_folder(), args.name + "_node")
else:
    print("==== evaluate ====")

def get_loss(model, ts, xs, ys):
    if args.transient_skip:
        ts, xs, ys = ts[:, args.transient_len:], xs[:, args.transient_len:], ys[:, args.transient_len:]
    
    xs, ys = xs.to(device), ys.to(device)
    
    # -- Normalisation de la cible
    x_norm_true = normalizer.normalize(xs)
    
    # -- Filtre KKL
    zs = latent_dyn.compute_z_fast(ys)
    
    # -- Prédiction via Neural ODE
    x_norm_pred = model(zs)
    
    # -- Loss MSE (Déterministe)
    loss = CRITERION(x_norm_pred, x_norm_true)
    return loss

tic = time.time()
for epoch in range(args.epochs):
    #-- train
    model.train()
    for ts, xs, ys in train_loader:
        loss = get_loss(model, ts, xs, ys)        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        stats.batch(**{'node_mse': loss})
    stats.epoch(epoch, 'train', time=time.time() - tic, lr=scheduler.get_last_lr()[0])
    
    #-- valid
    model.eval()
    with torch.no_grad():
        for ts, xs, ys in valid_loader:
            loss = get_loss(model, ts, xs, ys)
            stats.batch(**{'node_mse': loss})
    stats.epoch(epoch, 'valid')
    
    #-- verbose / plot evolution
    if (epoch > 0 and epoch % 20 == 0) or (epoch == args.epochs - 1):
        stats.render({'node_mse'}, save=True)
    
    # -- best loss ?
    best_epoch = np.argmin(stats.valid['node_mse'])
    if best_epoch == epoch:
        save(model)

    stats.save()
    scheduler.step()


# %% --- test ---
load(model)

# -- config data
traj_len = 5000#args.traj_len
n_traj = 100

# -- data
set_seed(args.seed)
test_dataset = KKLDataset(n_trajs=n_traj,
                           traj_len=traj_len,
                           noise_std=args.noise_std,
                           process_std=args.process_std)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
ts, xs, ys = (_.to(device) for _ in next(iter(test_loader)))

# -- compute z
zs = latent_dyn.compute_z_fast(ys)
plt.figure()
plt.plot(ts[0].cpu(), zs[0].cpu())
plt.title('Latent state z(t)')
plt.show()


def obs_unimodal():
    model.eval()
    with torch.no_grad():
        # inference (deterministic)
        x_norm_pred = model(zs)
        xs_obs = normalizer.unnormalize(x_norm_pred)
    
    MSE = ((xs_obs - xs)**2).mean().cpu()
    print('MSE %.3f RMSE %.3f' % (MSE, np.sqrt(MSE)))
    
    # -- plot
    fig = KKLDataset.render_obs(ts.cpu(), xs.cpu(), xs_obs.cpu(), ys.cpu())
    return fig

batch = 0
fig = obs_unimodal()