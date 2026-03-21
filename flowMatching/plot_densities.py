import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_transport_and_density(model, latent_dyn, normalizer, data_loader, args, device,
                                KKLDataset, traj_idx=0, n_samples=800, n_steps=50,
                                n_clusters=2, taus=[0.0, 0.5, 0.75, 1.0],
                                t_indices=[-1, -5, -10, -15, -20, -25], save_plots=True):
    from sklearn.cluster import KMeans
    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:pink",
              "tab:purple", "tab:brown", "tab:red", "tab:cyan"]

    # -- data
    tss, xss, ys = next(iter(data_loader))
    xs = normalizer.normalize(xss).to(device)
    ts = normalizer.normalize_t(tss).to(device)
    ys = ys.to(device)
    with torch.no_grad():
        z_all = latent_dyn.compute_z_fast(ys)

    z_list = [z_all[traj_idx, t_idx, :] for t_idx in t_indices]
    t_list = [ts[traj_idx, t_idx, 0]    for t_idx in t_indices]

    # -- sample xi0 once for all z
    torch.manual_seed(0)
    xi0 = torch.randn(n_samples, 1, model.x_dim, device=device)

    # -- transport
    def transport(z_vec, t_val):
        N, xi = n_samples, xi0.clone()
        z_e   = z_vec.view(1, 1, -1).expand(N, 1, -1)
        t_e   = torch.full((N, 1, 1), t_val, device=device)
        snaps, dt_tau = {}, 1.0 / n_steps
        tau_grid = torch.linspace(0, 1, n_steps + 1, device=device)
        model.eval()
        with torch.no_grad():
            for i in range(n_steps):
                tv = tau_grid[i].item()
                for st in sorted(taus):
                    if st not in snaps and tv >= st - dt_tau / 2:
                        snaps[st] = xi.squeeze(1).cpu().numpy()
                xi = xi + model(xi, torch.full((N, 1, 1), tv, device=device), z_e, t_e) * dt_tau
        snaps[1.0] = xi.squeeze(1).cpu().numpy()
        return [snaps[t] for t in sorted(taus)]

    torch.manual_seed(0)
    results = []
    for z_vec, t_val in zip(z_list, t_list):
        snaps  = transport(z_vec, t_val)
        labels = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit_predict(snaps[-1])
        results.append((snaps, labels))

    # -- plot 1: transport
    fig1, axes = plt.subplots(len(t_indices), len(taus), figsize=(4 * len(taus), 4 * len(t_indices)))
    xlim = ylim = None
    for row, ((snaps, labels), t_idx) in enumerate(zip(results, t_indices)):
        for col, (snap, tau_val) in enumerate(zip(snaps, taus)):
            ax = axes[row, col]
            for k in range(n_clusters):
                ax.scatter(snap[labels == k, 0], snap[labels == k, 1],
                           c=COLORS[k], alpha=0.4, s=12)
            # ax.set_title(rf"$\tau={tau_val}$ | $z[{t_idx}]$")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(alpha=0.3)
            if col == 0 and row == 0:
                xlim, ylim = ax.get_xlim(), ax.get_ylim()
            else:
                ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_aspect("equal")  # ← al final, después de set_xlim/ylim

    plt.tight_layout()
    if save_plots:
        filename = f"{args.dataset}_noise{args.noise_std}_transport_clusters"
        plt.savefig(filename + ".pdf", format='pdf')
        plt.savefig(filename + ".png")
    plt.show()

    # -- plot 2: temporal evolution
    fig2, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    T    = z_all.shape[1]
    time = np.arange(T)
    for dim, (data, label) in enumerate([(z_all[traj_idx].cpu().numpy(), "z(t)"),
                                          (xs[traj_idx].cpu().numpy(),    "x(t)"),
                                          (ys[traj_idx].cpu().numpy(),    "y(t)")]):
        for d in range(data.shape[1]):
            axes[dim].plot(tss[traj_idx, :, 0], data[:, d], alpha=0.7, label=f"{label[0]}[{d}]")
        axes[dim].set_ylabel(label)
        axes[dim].legend(loc="upper left", fontsize=8, ncol=data.shape[1])
    axes[2].set_xlabel("time (s)")
    for t_idx in t_indices:
        actual = T + t_idx if t_idx < 0 else t_idx
        for ax in axes: ax.axvline(tss[traj_idx, actual, 0], color="red", linestyle="--", linewidth=1.0, alpha=0.8)
        axes[0].annotate(f"t={t_idx}", xy=(tss[traj_idx, actual, 0], axes[0].get_ylim()[1]),
                         fontsize=7, color="red", ha="center", va="bottom")
    plt.suptitle(f"Trajectory {traj_idx} — temporal evolution", fontsize=12)
    plt.tight_layout()
    if save_plots:
        filename = f"{args.dataset}_noise{args.noise_std}_temporal_evolution"
        plt.savefig(filename + ".pdf", format='pdf')
        plt.savefig(filename + ".png")
    plt.show()
    return fig1, fig2
