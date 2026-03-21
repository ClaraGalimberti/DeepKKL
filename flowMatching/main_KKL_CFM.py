import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import argparse, os
import time
from scipy.optimize import linear_sum_assignment

from dataset import datasets
from plot_densities import plot_transport_and_density
from utils import set_seed, Train_Stats, Normalizer
from models import (Vector_Field_MLP, Conditional_Flow_Matching,
                    KKL_Latent_Dynamics, get_multimodal_estimates)


# %% --- config ---

parser = argparse.ArgumentParser()
# data
parser.add_argument('--dataset', type=str, default='VDP') # VDP, Test, Duffing
parser.add_argument('--noise_std', type=float, default=0.)
parser.add_argument('--process_std', type=float, default=.0)
parser.add_argument('--traj_len', type=int, default=2000)
# model
parser.add_argument('--name', type=str, default='temp')
parser.add_argument('--z_dim', type=int, default=6)
parser.add_argument('--use_t', action="store_true", help="learn p(x|z,t) if True, p(x|z) else")
# transient
parser.add_argument('--transient_len', type=int, default=400)
parser.add_argument('--transient_skip', action="store_true", help="during training, discard transient (let z converge)", default=True)
# train
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3*2)
parser.add_argument('--batchsize', type=int, default=50)
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

train_dataset = KKLDataset(n_trajs=1000*10,
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

model = Vector_Field_MLP(x_dim=x_dim,
                         z_dim=args.z_dim,
                         net_arch=[128*2]*3).to(device)
latent_dyn = KKL_Latent_Dynamics(z_dim=args.z_dim, dt=dt,
                                 noise_std=args.noise_std, device=device)

normalizer = Normalizer(train_dataset.xs,
                        time_period=args.transient_len * dt if args.use_t else 0)
# normalizer.render(t_max=args.traj_len * dt)

def trained_folder():
    return os.path.join("trained_models", args.dataset)

def path(name):
    folder = trained_folder()
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name + ".pth")
    
def save(model):
    print("\t save model", path(args.name))
    checkpoint = {'model': model.to('cpu').state_dict(),
                  }
    torch.save(checkpoint, path(args.name))
    model = model.to(device)

def load(model):
    print("\t load model", path(args.name))
    checkpoint = torch.load(path(args.name))
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)

if args.epochs == 0:
    load(model)


# %% --- training ---

if args.epochs > 0:
    print("==== training ====")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs//4), gamma=.5)
    stats = Train_Stats({'flow_match'}, trained_folder(), args.name)
else:
    print("==== evaluate ====")


def get_loss(model, ts, xs, ys):
    if args.transient_skip:
        ts, xs, ys = ts[:, args.transient_len:], xs[:, args.transient_len:], ys[:, args.transient_len:]

    # -- normalize
    # xs: (batchsize, traj_len, nx)
    # ts: (batchsize, traj_len, 1)
    ts, xs, ys = ts.to(device), xs.to(device), ys.to(device)
    x1 = normalizer.normalize(xs)
    ts = normalizer.normalize_t(ts)

    # --- FLOW MATCHING LOSS (Optimal Transport) ---
    x0 = torch.randn_like(x1).to(x1.device)
    tau = torch.rand(*x1.shape[:-1], 1).to(x1.device)
    # -- Minibatch Optimal Transport (OT-CFM)
    # Appariement optimal (Algorithme Hongrois).
    # En triant les paires au sein du batch pour minimiser la distance totale, on évite des trajectoires qui se croisent inutilement.
    # Le champ de vecteurs devient plus lisse et simple.
    # dists = torch.cdist(x0, x1) ** 2
    # row_idx, col_idx = linear_sum_assignment(dists.cpu().detach().numpy())
    
    # -- Target: Velocity
    x_tau = (1 - tau) * x0 + tau * x1
    u_tau = x1 - x0
    # -- Predict
    zs = latent_dyn.compute_z_fast(ys)
    v_tau = model(x_tau, tau, zs, ts)
    # -- Loss
    loss = CRITERION(v_tau, u_tau)
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
        
        stats.batch(**{'flow_match': loss})
    stats.epoch(epoch, 'train', time=time.time() - tic, lr=scheduler.get_last_lr()[0])
    
    #-- valid
    model.eval()
    with torch.no_grad():
        for ts, xs, ys in valid_loader:
            loss = get_loss(model, ts, xs, ys)
            stats.batch(**{'flow_match': loss})
    stats.epoch(epoch, 'valid')
    
    #-- verbose / plot evolution
    if (epoch > 0 and epoch % 20 == 0) or (epoch == args.epochs - 1):
        stats.render({'flow_match'}, save=True)
    
    # -- best loss ?
    best_epoch = np.argmin(stats.valid['flow_match'])
    if best_epoch == epoch:
        save(model)

    stats.save()
    scheduler.step()


# %% --- test ---

# -- config data
traj_len = 500 #args.traj_len
if args.dataset == 'VDP':
    traj_len = 5000
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
zs = latent_dyn.compute_z_fast(ys) # (n_traj, traj_len, z_dim)
plt.figure(figsize=(10, 6))
plt.plot(ts[0].cpu(), zs[0].cpu())
plt.title("Latent dynamics")
plt.show()

load(model)

def obs_unimodal(batch_to_plot=10, n_steps=20, median=False, mse=True):
    # -- compute x_obs
    ts_norm = normalizer.normalize_t(ts)
    if median:
        xs_obs = Conditional_Flow_Matching.solve_ode_median(model, zs, ts_norm, n_steps=20, n_candidates=50)
    else:
        xs_obs = Conditional_Flow_Matching.solve_ode(model, zs, ts_norm, n_steps=n_steps, sample_every_t=False)
    xs_obs = normalizer.unnormalize(xs_obs)
    
    MSE = ((xs_obs - xs)**2).mean().cpu()
    print('MSE %.3f RMSE %.3f' % (MSE, np.sqrt(MSE)))
    
    # -- plot
    fig = KKLDataset.render_obs(ts.cpu(), xs.cpu(), xs_obs.cpu(), ys.cpu(), batch_number=batch_to_plot, mse=mse)
    filename = "%s_noise%.2f" % (args.dataset, args.noise_std)
    plt.savefig(filename + ".pdf", format='pdf')
    plt.savefig(filename + ".png")
    plt.show()
    return fig


def obs_multimodal(batch=0, n_modes=2, with_y=False):
    # TODO : fusionner solve_ode / solve_ode_median
    N_particles = 100
    
    # 2. Générer des particules avec le Flow
    ts_norm = normalizer.normalize_t(ts)
    z_particles = zs[batch].unsqueeze(0).repeat(N_particles, *([1]*zs[batch].ndim)) # (N_particles, traj_len, z_dim)
    x_particles = Conditional_Flow_Matching.solve_ode(model, z_particles, ts_norm) # (N_particles, traj_len,  x_dim)
    
    # -- compute x_obs
    xs_obs = torch.zeros(n_modes, *xs.shape[1:]).to(device) # (n_modes, traj_len, x_dim)
    for step in range(traj_len):
        previous = xs_obs[:, step - 1] if step > 0 else None # (n_modes, x_dim)
        xs_obs[:, step] = get_multimodal_estimates(x_particles[:, step],
                                                   n_modes=2,
                                                   previous_estimates=previous)
    xs_obs = normalizer.unnormalize(xs_obs)
    
    # -- plot
    y_size = 8 if with_y else 6
    fig = plt.figure(figsize=(10, y_size))
    subplots = x_dim
    if with_y:
        subplots = subplots + 1
    colors = sns.color_palette("Paired", n_colors=10); #sns.palplot(colors)
    # -- plot x and x_obs
    for i in range(x_dim):
        plt.subplot(subplots, 1, i + 1)
        plt.plot(ts[batch].cpu(), xs[batch, :, i].cpu(), label='$x_%i$' % (i + 1), color=colors[3])
        for k in range(n_modes):
            plt.plot(ts[batch].cpu(), xs_obs[k, :, i].cpu(), '--',
                     label='$\hat x_%i^{(%i)}$' % (i + 1, k), color=colors[5+k])
        if i < subplots - 1:
            plt.xticks([]) 
        else: 
            plt.xlabel('time (s)')
        plt.grid(which='both')
        plt.legend(loc='right')
    if with_y:
        plt.subplot(subplots, 1, subplots)
        plt.plot(ts[batch].cpu(), ys[batch, :, 0].cpu(), label='$y$', color=colors[4])
        #plt.xlabel('time (s)')
        plt.legend()
        plt.grid(which='both')
    plt.tight_layout()
    filename = "%s_noise%.2f" % (args.dataset, args.noise_std)
    if not with_y:
        filename = filename + "_noy"
    plt.savefig(filename + ".pdf", format='pdf')
    plt.savefig(filename + ".png")
    plt.show()
    return fig


def exact_density(batch=0, steps=[0, -1]):

    # -- density
    for step in steps:
        # zs: (n_traj, traj_len, z_dim)
        z = zs[batch, step] # (z_dim,)
        x = xs[batch, step] # (x_dim,)
        t = ts[batch, step] # (1,)
        t_norm = normalizer.normalize_t(t)

        # -- exact density
        x_min, x_max = KKLDataset.x0_low, KKLDataset.x0_high # [-2, -2], [2, 2]
        # x_min, x_max = -x.cpu() - .05, -x.cpu() + .05

        X_grid, Y_grid, density = Conditional_Flow_Matching.compute_exact_density_map(
            model,
            z, # to_torch = lambda a: torch.tensor(a, dtype=torch.float32).to(device)
            t_norm,
            x_min=x_min,
            x_max=x_max,
            normalizer=normalizer,
            grid_size=500, # critical parameter to obtain nice plot (e.g. 500) !
            steps=20
        )
        density_norm = density/(density.max())
        fig = plt.figure(figsize=(10, 8))
        plt.contourf(X_grid, Y_grid, density_norm, levels=50, cmap='viridis')
        cbar = plt.colorbar()
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        # plt.plot(x[0].cpu(), x[1].cpu(), 'r*', markersize=10, label='$x$', alpha=0.3)
        # plt.plot(xs_obs[batch, step, 0].cpu(), xs_obs[batch, step, 1].cpu(), 'w*', markersize=5, label='$\hat x$')

        ticks = [-2, -1, 0, 1, 2]
        plt.xticks(ticks)
        plt.yticks(ticks)

        # plt.ylim([0.8,1.1])
        # plt.xlim([-0.5, -0.2])

        # plt.legend()
        #plt.title('$p(x|z)$ (t=%.2f)' % ts[batch, step])
        plt.tight_layout()
        filename = "%s_contour_b%i" % (args.dataset, batch)
        plt.savefig(filename + ".pdf", format='pdf')
        plt.savefig(filename + ".png")
        print("Save as %s" % filename)
        plt.show()

    return

def exact_density_evolution(batch=0, steps=[0, -1], taus=[0, 0.25, 0.5, 1.]):

    # -- density
    tau_len = len(taus)
    for step in steps:
        fig = plt.figure(figsize=(8 * tau_len, 8))
        for i, tau in enumerate(taus):
            # zs: (n_traj, traj_len, z_dim)
            z = zs[batch, step] # (z_dim,)
            x = xs[batch, step] # (x_dim,)
            t = ts[batch, step] # (1,)
            t_norm = normalizer.normalize_t(t)

            # -- exact density
            x_min, x_max = KKLDataset.x0_low, KKLDataset.x0_high # [-2, -2], [2, 2]
            # x_min, x_max = -x.cpu() - .05, -x.cpu() + .05

            X_grid, Y_grid, density = Conditional_Flow_Matching.compute_exact_density_map(
                model,
                z, # to_torch = lambda a: torch.tensor(a, dtype=torch.float32).to(device)
                t_norm,
                x_min=x_min,
                x_max=x_max,
                normalizer=normalizer,
                grid_size=250, # critical parameter to obtain nice plot (e.g. 500) !
                steps=20,
                tau_end=tau
            )

            plt.subplot(1, tau_len, i + 1)
            density_norm = density/(density.max())
            plt.contourf(X_grid, Y_grid, density_norm, levels=50, cmap='viridis', vmin=0, vmax=1.01)
            # if i+1 == tau_len:
            #     cbar = plt.colorbar()
            #     cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            #plt.plot(x[0].cpu(), x[1].cpu(), 'r*', markersize=10, label='$x$', alpha=0.3)
            # plt.plot(xs_obs[batch, step, 0].cpu(), xs_obs[batch, step, 1].cpu(), 'w*', markersize=5, label='$\hat x$')
            #plt.legend()
            # ticks = [-2, -1, 0, 1, 2]
            ticks=[]
            plt.xticks(ticks)
            plt.yticks(ticks)
            #plt.title(r'$p(x|z)$ (t=%.2f - $\tau$=%.3f)' % (ts[batch, step], tau))
        plt.tight_layout()
        filename = "%s_density_t%.2f_noise%.1f" % (args.dataset, ts[batch, step], args.noise_std)
        plt.savefig(filename + ".pdf", format='pdf')
        plt.savefig(filename + ".png")
        plt.show()

    return

def obs_unimodal_multisampled(batch_to_plot=0, n_candidates=150, n_steps=40, confidence=0.90):
    """
    For a single trajectory, runs the ODE n_candidates times with different xi0
    and plots a confidence tube (default 90%) around the mean estimate.
    """
    ts_norm = normalizer.normalize_t(ts)
    b = batch_to_plot
    # z and t for the selected trajectory, repeated n_candidates times
    z_single = zs[b].unsqueeze(0).expand(n_candidates, -1, -1)  # (C, T, z_dim)
    t_single = ts_norm[b].unsqueeze(0).expand(n_candidates, -1, -1)  # (C, T, 1)
    # solve ODE with n_candidates different xi0
    xs_candidates = Conditional_Flow_Matching.solve_ode(
        model, z_single, t_single, n_steps=n_steps, sample_every_t=False
    )  # (C, T, x_dim)
    xs_candidates = normalizer.unnormalize(xs_candidates).cpu().numpy()  # (C, T, x_dim)
    # percentile bounds
    alpha = (1 - confidence) / 2 * 100  # e.g. 5
    low = np.percentile(xs_candidates, alpha, axis=0)  # (T, x_dim)
    high = np.percentile(xs_candidates, 100 - alpha, axis=0)  # (T, x_dim)
    mean = xs_candidates.mean(axis=0)  # (T, x_dim)
    # -- plot
    sns.set(style="ticks", context="talk", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"
    colors = sns.color_palette("Paired", n_colors=10)
    t_cpu = ts[b].cpu().numpy().squeeze()  # (T,) en vez de (T, 1)
    fig, axes = plt.subplots(x_dim + 1, 1, figsize=(10, 8), sharex=True)
    # y(t)
    axes[0].plot(t_cpu, ys[b, :, 0].cpu(), '--', color=colors[3], label='$y$')
    axes[0].legend(loc='right')
    axes[0].grid(which='both')
    # x(t) with confidence tube
    for i in range(x_dim):
        # confidence tube
        axes[i + 1].fill_between(t_cpu, low[:, i], high[:, i],
                                 color=colors[5], alpha=0.3,
                                 linewidth=0,
                                 )
        # single candidate trajectory
        axes[i + 1].plot(t_cpu, xs_candidates[4, :, i],
                         color=colors[5], linewidth=1.5, label=r'$\hat{x}_%i$' % (i + 1))
        # ground truth
        axes[i + 1].plot(t_cpu, xs[b, :, i].cpu(), '--',
                         color=colors[3], linewidth=1.5, label='$x_%i$' % (i + 1))
        axes[i + 1].legend(loc='right')
        axes[i + 1].grid(which='both')
    axes[-1].set_xlabel('time (s)')
    plt.tight_layout()
    filename = "%s_noise%.2f_confidence%i" % (args.dataset, args.noise_std, int(confidence * 100))
    plt.savefig(filename + ".pdf", format='pdf')
    plt.savefig(filename + ".png")
    print("Save as %s" % filename)
    plt.show()
    return fig

def obs_multimodal2(batch=0, n_modes=2, with_y=True, n_candidates=100, confidence=0.90, n_steps=40):
    ts_norm     = normalizer.normalize_t(ts)
    colors      = sns.color_palette("Paired", n_colors=10)
    mode_colors = [colors[5], colors[7], colors[9], colors[1]]
    alpha       = (1 - confidence) / 2 * 100

    # -- generate particles
    z_particles = zs[batch].unsqueeze(0).repeat(n_candidates, *([1]*zs[batch].ndim))
    x_particles = Conditional_Flow_Matching.solve_ode(model, z_particles, ts_norm, n_steps=n_steps)

    # -- track modes
    xs_obs = torch.zeros(n_modes, *xs.shape[1:]).to(device)
    for step in range(traj_len):
        previous = xs_obs[:, step - 1] if step > 0 else None
        xs_obs[:, step] = get_multimodal_estimates(x_particles[:, step],
                                                    n_modes=n_modes,
                                                    previous_estimates=previous)
    xs_obs = normalizer.unnormalize(xs_obs)

    # -- assign particles to modes
    x_particles_np = normalizer.unnormalize(x_particles).cpu().numpy()
    xs_obs_np      = xs_obs.cpu().numpy()
    t_cpu          = ts[batch].cpu().numpy().squeeze()

    dists      = np.linalg.norm(
        x_particles_np[:, :, None, :] - xs_obs_np[None, :, :, :].transpose(0, 2, 1, 3),
        axis=-1
    )
    assignment = dists.argmin(axis=-1)  # (C, T)

    # -- plot
    sns.set(style="ticks", context="talk", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"

    subplots = x_dim + (1 if with_y else 0)
    fig, axes = plt.subplots(subplots, 1, figsize=(10, 6 + (2 if with_y else 0)), sharex=True)
    if subplots == 1: axes = [axes]

    offset = 0
    # -- y(t) first
    if with_y:
        axes[0].plot(t_cpu, ys[batch, :, 0].cpu(), '--', color=colors[3], label='$y$')
        axes[0].legend(loc='right'); axes[0].grid(which='both')
        offset = 1

    # -- x(t) with tube per mode
    for i in range(x_dim):
        ax = axes[i + offset]
        # ground truth
        ax.plot(t_cpu, xs[batch, :, i].cpu(), '--',
                color=colors[3], linewidth=1.5, label='$x_%i$' % (i+1))
        # tube per mode
        for k in range(n_modes):
            mask = (assignment == k)
            low  = np.array([np.percentile(x_particles_np[mask[:, t], t, i], alpha)
                             for t in range(traj_len)])
            high = np.array([np.percentile(x_particles_np[mask[:, t], t, i], 100 - alpha)
                             for t in range(traj_len)])
            ax.fill_between(t_cpu, low, high, color=mode_colors[k], alpha=0.3,
                            linewidth=0)
            ax.plot(t_cpu, xs_obs_np[k, :, i], '--',
                    color=mode_colors[k], linewidth=1.5,
                    label=r'$\hat x_%i^{(%i)}$' % (i+1, k))
        if i < x_dim - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('time (s)')
        ax.grid(which='both'); ax.legend(loc='right')

    plt.tight_layout()
    filename = "%s_noise%.2f_confidence%.2f%s" % (args.dataset, args.noise_std, confidence, "_noy" if not with_y else "")
    plt.savefig(filename + ".pdf", format='pdf')
    print("Saved as %s" % filename)
    plt.savefig(filename + ".png")
    plt.show()
    return fig


def exact_density2(batch=0, steps=[0, -1]):
    for step in steps:
        z = zs[batch, step]
        x = xs[batch, step]
        t = ts[batch, step]
        t_norm = normalizer.normalize_t(t)

        x_min, x_max = KKLDataset.x0_low, KKLDataset.x0_high

        X_grid, Y_grid, density = Conditional_Flow_Matching.compute_exact_density_map(
            model, z, t_norm,
            x_min=x_min, x_max=x_max,
            normalizer=normalizer,
            grid_size=500, steps=20
        )
        density_norm = density / density.max()

        fig, ax_main = plt.subplots(figsize=(10, 8))
        cf = ax_main.contourf(X_grid, Y_grid, density_norm, levels=50, cmap='viridis')
        cbar = plt.colorbar(cf, ax=ax_main)
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax_main.set_xticks([-2, -1, 0, 1, 2])
        ax_main.set_yticks([-2, -1, 0, 1, 2])

        # rectangle zoom 1
        ax_main.add_patch(Rectangle((-0.45, 0.85), 0.20, 0.20,
                                    fill=False, edgecolor='red', linewidth=1))

        # rectangle zoom 2
        ax_main.add_patch(Rectangle((0.25, -1.05), 0.20, 0.20,
                                    fill=False, edgecolor='red', linewidth=1))

        # -- zoom 1: top-left mode
        ax_z1 = fig.add_axes([0.59, 0.62, 0.22, 0.22])
        ax_z1.contourf(X_grid, Y_grid, density_norm, levels=50, cmap='viridis')
        ax_z1.set_xlim([-0.45, -0.25])
        ax_z1.set_ylim([0.85, 1.05])
        ax_z1.set_xticks([-0.45, -0.25])
        ax_z1.set_yticks([0.85, 1.05])
        ax_z1.tick_params(labelsize=12, labelcolor='red')
        for spine in ax_z1.spines.values():
            spine.set_edgecolor('red');
            spine.set_linewidth(1.5)

        # -- zoom 2: bottom-right mode
        ax_z2 = fig.add_axes([0.59, 0.20, 0.22, 0.22])
        ax_z2.contourf(X_grid, Y_grid, density_norm, levels=50, cmap='viridis')
        ax_z2.set_xlim([0.25, 0.45])
        ax_z2.set_ylim([-1.05, -0.85])
        ax_z2.set_xticks([0.25, 0.45])
        ax_z2.set_yticks([-1.05, -0.85])
        ax_z2.tick_params(labelsize=12, labelcolor='red')
        for spine in ax_z2.spines.values():
            spine.set_edgecolor('red');
            spine.set_linewidth(1.5)

        filename = "%s_contour_b%i" % (args.dataset, batch)
        plt.savefig(filename + ".pdf", format='pdf')
        plt.savefig(filename + ".png")
        print("Save as %s" % filename)
        plt.show()


batch = 0
# Time trajectories of states and MSE
# fig = obs_multimodal(batch=batch)
if args.dataset == "VDP":
    obs_unimodal(median=False, mse=False)
    obs_unimodal_multisampled()
if args.dataset == "Test":
    obs_multimodal2()
    plot_transport_and_density(
        model, latent_dyn, normalizer, valid_loader, args, device, KKLDataset,
        traj_idx=0,
        taus=[0.0, 0.5, 0.7, 0.9, 1.0],
        t_indices=[-1, -20],
    )
    exact_density2(batch=3, steps=[-1])
if args.dataset == "Duffing":
    exact_density(batch=0, steps=[-1])
    exact_density(batch=1, steps=[-1])

# Density
exact_density_evolution(batch=batch, steps=[400], taus=[0, 0.02, 0.05, 0.1, 0.5, 1.])

