"""
Particle Filter Observer for the 'Test' system.

Implements a Bootstrap Particle Filter (SIR) to estimate the state x from
noisy measurements y, for the same system used in the KKL-CFM project.

Noise parameters are kept consistent with main_KKL_CFM.py defaults:
    noise_std   = 0.0  (measurement noise std)
    process_std = 0.0  (process noise std)

You can override them via --noise_std and --process_std CLI args.

Visualization: temporal "probability tube" (percentile bands) per state dimension.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy.optimize import linear_sum_assignment

from dataset import datasets, RK4


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',     type=str,   default='Test')
parser.add_argument('--noise_std',   type=float, default=0.0,
                    help='Measurement noise std (matches KKL-CFM default)')
parser.add_argument('--process_std', type=float, default=0.0,
                    help='Process noise std (matches KKL-CFM default)')
parser.add_argument('--traj_len',    type=int,   default=500,
                    help='Trajectory length (time-steps)')
parser.add_argument('--n_particles', type=int,   default=2000,
                    help='Number of particles')
parser.add_argument('--batch',       type=int,   default=0,
                    help='Trajectory index to plot in detail')
parser.add_argument('--n_trajs',     type=int,   default=1,
                    help='Number of test trajectories')
parser.add_argument('--resample_method', type=str, default='systematic',
                    choices=['systematic', 'multinomial'],
                    help='Resampling strategy: systematic (lower variance) or multinomial')
parser.add_argument('--n_modes', type=int, default=2,
                    help='Number of modes for k-means clustering of particles. '
                         '1 = disabled (default behaviour: weighted median).')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
print(args)

np.random.seed(args.seed)

# ---------------------------------------------------------------------------
# Dataset / system
# ---------------------------------------------------------------------------
KKLDataset = datasets[args.dataset]
x_dim = KKLDataset.x_dim
y_dim = KKLDataset.y_dim
dt    = KKLDataset.dt

dataset = KKLDataset(
    n_trajs=args.n_trajs,
    traj_len=args.traj_len,
    noise_std=args.noise_std,
    process_std=args.process_std,
)

# ts: (n_trajs, traj_len, 1)  ->  squeeze to (n_trajs, traj_len)
ts = dataset.ts[:, :, 0]        # (n_trajs, traj_len)
xs = dataset.xs                 # (n_trajs, traj_len, x_dim)
ys = dataset.ys                 # (n_trajs, traj_len, y_dim)


# ---------------------------------------------------------------------------
# K-Means clustering of particles  (numpy port of get_multimodal_estimates)
# ---------------------------------------------------------------------------

def kmeans_particles(particles, n_modes, previous_centroids=None, n_iter=10):
    """
    Simple k-means on a set of particles (numpy).

    Parameters
    ----------
    particles          : (N, x_dim)
    n_modes            : int
    previous_centroids : (n_modes, x_dim) or None
                         If provided, centroids are matched to previous ones via
                         the Hungarian algorithm to avoid label-switching.
    n_iter             : int  — k-means iterations (5-10 is enough for well-separated modes)

    Returns
    -------
    centroids : (n_modes, x_dim)  — ordered to match previous_centroids
    labels    : (N,)              — cluster assignment for each particle
    """
    N, x_dim = particles.shape

    # --- 1. Initialise centroids (random subset of particles) ---
    idx = np.random.choice(N, size=n_modes, replace=False)
    centroids = particles[idx].copy()

    labels = np.zeros(N, dtype=int)
    for _ in range(n_iter):
        diffs = particles[:, None, :] - centroids[None, :, :]  # (N, K, x_dim)
        dists = np.linalg.norm(diffs, axis=-1)                 # (N, K)
        labels = np.argmin(dists, axis=1)                      # (N,)
        for k in range(n_modes):
            mask = labels == k
            if mask.any():
                centroids[k] = particles[mask].mean(axis=0)

    # --- 2. Tracking: match centroids to previous via Hungarian ---
    if previous_centroids is not None:
        cost = np.linalg.norm(
            previous_centroids[:, None, :] - centroids[None, :, :], axis=-1
        )  # (n_modes, n_modes)
        _, col_ind = linear_sum_assignment(cost)
        centroids = centroids[col_ind]
        remap = np.zeros(n_modes, dtype=int)
        remap[col_ind] = np.arange(n_modes)
        labels = remap[labels]
    else:
        sort_idx = np.argsort(centroids[:, 0])
        centroids = centroids[sort_idx]
        remap = np.zeros(n_modes, dtype=int)
        remap[sort_idx] = np.arange(n_modes)
        labels = remap[labels]

    return centroids, labels


def _weighted_median_per_cluster(p_k, w_k, x_dim):
    """Weighted median of cluster particles, independently per dimension."""
    w_k = w_k / w_k.sum() if w_k.sum() > 0 else np.ones(len(w_k)) / len(w_k)
    median = np.zeros(x_dim)
    for d in range(x_dim):
        sort_idx = np.argsort(p_k[:, d])
        cumw = np.cumsum(w_k[sort_idx])
        cumw /= cumw[-1]
        idx = np.clip(np.searchsorted(cumw, 0.5), 0, len(p_k) - 1)
        median[d] = p_k[sort_idx[idx], d]
    return median


# ---------------------------------------------------------------------------
# Bootstrap Particle Filter
# ---------------------------------------------------------------------------

def particle_filter(ys_traj, n_particles, noise_std, process_std,
                    x0_low, x0_high, dt, get_derivs, get_y,
                    resample_method='systematic', n_modes=1):
    """
    Bootstrap SIR Particle Filter.

    Parameters
    ----------
    ys_traj          : (traj_len, y_dim)
    n_particles      : int
    noise_std        : float
    process_std      : float
    x0_low, x0_high  : (x_dim,)
    dt               : float
    get_derivs       : callable(x, u=None) -> dx/dt
    get_y            : callable(x) -> y
    resample_method  : 'systematic' or 'multinomial'
    n_modes          : int — number of k-means modes.
                       1 = single weighted median (no clustering).
                       >1 = one weighted median per cluster.

    Returns
    -------
    particles_history  : (traj_len, n_particles, x_dim)
    weights_history    : (traj_len, n_particles)
    modal_estimates    : (n_modes, traj_len, x_dim)
                         If n_modes=1, shape is (1, traj_len, x_dim).
    resample_steps     : list of int
    """
    traj_len, y_dim = ys_traj.shape
    x_dim = len(x0_low)
    meas_std = noise_std if noise_std > 1e-8 else 1e-2

    particles = np.random.uniform(x0_low, x0_high,
                                  size=(n_particles, x_dim)).astype(np.float32)
    weights = np.ones(n_particles, dtype=np.float32) / n_particles

    particles_history = np.zeros((traj_len, n_particles, x_dim), dtype=np.float32)
    weights_history   = np.zeros((traj_len, n_particles), dtype=np.float32)
    labels_history    = np.zeros((traj_len, n_particles), dtype=int)  # cluster label per particle
    modal_estimates   = np.zeros((n_modes, traj_len, x_dim))
    resample_steps    = []
    previous_centroids = None

    resample_fn = _systematic_resample if resample_method == 'systematic' \
                  else _multinomial_resample

    for k in range(traj_len):
        y_k = ys_traj[k]

        # ----- 1. Propagate -----
        if k > 0:
            noise = np.random.normal(0, process_std, size=particles.shape).astype(np.float32) \
                    if process_std > 1e-8 else 0.0
            particles = (RK4(get_derivs, dt, particles) + dt * noise).astype(np.float32)

        # ----- 2. Weighting -----
        y_pred = get_y(particles)
        diff   = y_pred - y_k[np.newaxis, :]
        log_w  = -0.5 * np.sum(diff**2, axis=-1) / meas_std**2
        log_w -= log_w.max()
        weights = np.exp(log_w)
        weights /= weights.sum()

        # ----- 3. Store -----
        particles_history[k] = particles
        weights_history[k]   = weights

        # ----- 4. Modal estimate -----
        if n_modes == 1:
            modal_estimates[0, k] = _weighted_median_per_cluster(particles, weights, x_dim)
            labels_history[k] = 0  # all particles belong to mode 0
        else:
            centroids, labels = kmeans_particles(particles, n_modes,
                                                 previous_centroids=previous_centroids)
            previous_centroids = centroids
            labels_history[k] = labels
            for m in range(n_modes):
                mask = labels == m
                if mask.any():
                    modal_estimates[m, k] = _weighted_median_per_cluster(
                        particles[mask], weights[mask], x_dim)
                else:
                    modal_estimates[m, k] = modal_estimates[m, k-1] if k > 0 else 0.0

        # ----- 5. Resample if N_eff < 50% -----
        neff = 1.0 / np.sum(weights**2)
        if neff < n_particles / 2:
            particles = resample_fn(particles, weights)
            weights = np.ones(n_particles, dtype=np.float32) / n_particles
            resample_steps.append(k)

    return particles_history, weights_history, modal_estimates, resample_steps, labels_history


def _systematic_resample(particles, weights):
    """
    Systematic resampling — O(N), single random draw.
    Positions: (u + 0, u+1, ..., u+N-1) / N  with u ~ U[0,1).
    Lower variance than multinomial; standard choice in PF literature.
    """
    N = len(weights)
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # guard against float rounding
    positions = (np.arange(N) + np.random.uniform()) / N
    indices = np.searchsorted(cumsum, positions)
    indices = np.clip(indices, 0, N - 1)
    return particles[indices]


def _multinomial_resample(particles, weights):
    """
    Multinomial resampling — N independent draws from Categorical(weights).
    Simple but higher variance: a particle can be missed even if its weight is high.
    """
    N = len(weights)
    indices = np.random.choice(N, size=N, p=weights)
    return particles[indices]


# ---------------------------------------------------------------------------
# Run filter on every trajectory
# ---------------------------------------------------------------------------
print(f"Running Bootstrap Particle Filter  [N={args.n_particles}, "
      f"noise_std={args.noise_std}, process_std={args.process_std}]")

all_particles       = []  # list of (traj_len, n_particles, x_dim)
all_weights         = []  # list of (traj_len, n_particles)
all_modal_estimates = []  # list of (n_modes, traj_len, x_dim)
all_resample_steps  = []  # list of lists of int
all_labels          = []  # list of (traj_len, n_particles)

for i in range(args.n_trajs):
    print(f"  trajectory {i+1}/{args.n_trajs} ...", end=" ", flush=True)
    ph, wh, me, rs, lh = particle_filter(
        ys_traj         = ys[i],
        n_particles     = args.n_particles,
        noise_std       = args.noise_std,
        process_std     = args.process_std,
        x0_low          = KKLDataset.x0_low,
        x0_high         = KKLDataset.x0_high,
        dt              = dt,
        get_derivs      = KKLDataset.get_derivs,
        get_y           = KKLDataset.get_y,
        resample_method = args.resample_method,
        n_modes         = args.n_modes,
    )
    all_particles.append(ph)
    all_weights.append(wh)
    all_modal_estimates.append(me)
    all_resample_steps.append(rs)
    all_labels.append(lh)

    # MSE using mode 0 as reference estimate
    mse = np.mean((me[0] - xs[i])**2)
    print(f"MSE={mse:.4f}  RMSE={np.sqrt(mse):.4f}")

print("Done.")


# ---------------------------------------------------------------------------
# Plot  — probability tube for one trajectory
# ---------------------------------------------------------------------------

def compute_weighted_percentiles(particles, weights, percentiles):
    """
    particles : (traj_len, n_particles, x_dim)
    weights   : (traj_len, n_particles)
    percentiles: list of floats in [0, 100]

    Returns dict {pct: (traj_len, x_dim)}
    Each dimension is sorted independently before computing the weighted CDF.
    """
    traj_len, n_p, x_d = particles.shape
    result = {p: np.zeros((traj_len, x_d)) for p in percentiles}
    for t in range(traj_len):
        for d in range(x_d):
            sorted_idx = np.argsort(particles[t, :, d])
            w_sorted   = weights[t, sorted_idx]
            p_sorted   = particles[t, sorted_idx, d]
            cumw       = np.cumsum(w_sorted)
            cumw      /= cumw[-1]  # normalise to exactly 1
            for p in percentiles:
                idx = np.searchsorted(cumw, p / 100.0)
                idx = np.clip(idx, 0, n_p - 1)
                result[p][t, d] = p_sorted[idx]
    return result


def plot_probability_tube(b=0):
    sns.set(style="ticks", context="talk", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"
    colors = sns.color_palette("Paired", n_colors=12)
    mode_colors = sns.color_palette("Set1", n_colors=max(args.n_modes, 2))

    t   = ts[b]
    xb  = xs[b]
    yb  = ys[b]
    ph  = all_particles[b]
    wh  = all_weights[b]
    me  = all_modal_estimates[b]   # (n_modes, traj_len, x_dim)
    lh  = all_labels[b]            # (traj_len, n_particles)
    resample_times = t[all_resample_steps[b]] if all_resample_steps[b] else []
    multimodal = args.n_modes > 1

    # percentiles per mode: dict {m: {pct: (traj_len, x_dim)}}
    pcts_per_mode = {}
    for m in range(args.n_modes):
        # mask particles belonging to mode m at each timestep
        ph_m = np.where(
            (lh == m)[:, :, np.newaxis],   # (traj_len, N, 1)
            ph,
            np.nan
        )  # (traj_len, N, x_dim)  — non-cluster particles set to nan
        # build weights zeroed out for other clusters, then renormalize
        wh_m = np.where(lh == m, wh, 0.0)  # (traj_len, N)
        wh_m_sum = wh_m.sum(axis=1, keepdims=True)
        wh_m_sum = np.where(wh_m_sum > 0, wh_m_sum, 1.0)
        wh_m = wh_m / wh_m_sum
        pcts_per_mode[m] = compute_weighted_percentiles(ph, wh_m, percentiles=[5, 95])

    fig, axes = plt.subplots(x_dim + 1, 1,
                             figsize=(12, 3 * (x_dim + 1)),
                             sharex=True)

    # --- top row: measurement y ---
    ax = axes[0]
    for j in range(y_dim):
        ax.plot(t, yb[:, j], color=colors[3+j], linewidth=1.2,
                label=f'$y_{j+1}$' + (' (noisy)' if args.noise_std > 0 else ''))
    # resample instants
    for k, rt in enumerate(resample_times):
        ax.axvline(rt, color='gray', linewidth=0.5, alpha=0.4,
                   label='resampling times' if k == 0 else None)
    ax.set_ylabel('Output $y$')
    ax.legend(loc='right', fontsize=10)
    ax.grid(True, which='both', alpha=0.4)
    mode_str = f', k={args.n_modes} modes' if multimodal else ''
    noise_str = f'noise={args.noise_std}' if args.noise_std > 0 else 'noiseless'
    ax.set_title(f"Particle Filter  —  {args.dataset}  "
                 f"(N={args.n_particles}, {noise_str}, "
                 f"{len(resample_times)} resamplings{mode_str})")

    # --- one row per state ---
    for i in range(x_dim):
        ax = axes[i + 1]

        # # scatter: particle cloud
        # t_expanded = np.repeat(t, args.n_particles)
        # p_expanded = ph[:, :, i].flatten()
        # ax.scatter(t_expanded, p_expanded, s=0.1, color='skyblue', alpha=0.05,
        #            label='Particles')

        # 90% tube per mode
        for m in range(args.n_modes):
            ax.fill_between(t,
                            pcts_per_mode[m][5][:, i],
                            pcts_per_mode[m][95][:, i],
                            alpha=0.20, color=mode_colors[m],
                            label='90%% interval' if not multimodal
                                  else '90%% interval (mode %i)' % m)

        # modal estimates  — me has shape (n_modes, traj_len, x_dim)
        for m in range(args.n_modes):
            label = r'$\hat x_{%i}^{(%i)}$' % (i+1, m) if multimodal \
                    else r'$\hat x_{%i}$ (median)' % (i+1)
            ax.plot(t, me[m, :, i],
                    color=mode_colors[m], linewidth=1.8, linestyle='--',
                    label=label)

        # ground truth
        ax.plot(t, xb[:, i], color=colors[3], linewidth=1.2,
                label='$x_{%i}$ (true)' % (i + 1))

        # resample instants
        for k, rt in enumerate(resample_times):
            ax.axvline(rt, color='gray', linewidth=0.5, alpha=0.4,)

        ax.set_ylabel(f'$x_{i+1}$')
        ax.legend(loc='right', fontsize=9)
        ax.grid(True, which='both', alpha=0.4)

    axes[-1].set_xlabel('time (s)')
    plt.tight_layout()
    plt.savefig('particle_filter_tube.pdf', bbox_inches='tight', format='pdf')
    plt.savefig('particle_filter_tube.png', dpi=150, bbox_inches='tight')
    print("Saved: particle_filter_tube.png / .pdf")
    plt.show()

    # --- per-mode MSE for this trajectory (avg over time only) ---
    print(f"\n===== Modal MSE  —  trajectory {b} =====")
    for m in range(args.n_modes):
        mse_m = ((me[m] - xb)**2).mean()
        print(f"  Mode {m}: MSE = {mse_m:.5f}   RMSE = {np.sqrt(mse_m):.5f}")
    print("=" * 40)

    return fig


# ---------------------------------------------------------------------------
# Plot  — Modal MSE metrics
# ---------------------------------------------------------------------------

def plot_mse_over_time():
    """
    Two metrics:

    1) Best-mode curve (per time step):
       At each t, pick the mode k with the smallest squared error to the true state.
       Average over trajectories → one curve showing the oracle lower bound.

    2) Per-mode scalar MSE:
       For each mode k, average squared error over time AND trajectories.
       Printed as a table and plotted as individual curves.
    """
    sns.set(style="ticks", context="talk", font_scale=1.2)
    plt.rcParams["font.family"] = "serif"
    colors     = sns.color_palette("Paired",  n_colors=10)
    mode_colors = sns.color_palette("Set1",   n_colors=max(args.n_modes, 2))

    t = ts[0]

    # --- build error arrays ---
    # err_modal[i] : (n_modes, traj_len)  — MSE over x_dim per mode per step
    # err_best[i]  : (traj_len,)          — min over modes at each step
    err_modal_all = []  # (n_trajs, n_modes, traj_len)
    err_best_all  = []  # (n_trajs, traj_len)

    for i in range(args.n_trajs):
        me = all_modal_estimates[i]   # (n_modes, traj_len, x_dim)
        x_true = xs[i]               # (traj_len, x_dim)

        # squared error per mode per step, averaged over x_dim
        err = ((me - x_true[np.newaxis])**2).mean(axis=-1)  # (n_modes, traj_len)
        err_modal_all.append(err)

        # at each t: best mode = argmin over k
        err_best_all.append(err.min(axis=0))                 # (traj_len,)

    err_modal_all = np.array(err_modal_all)  # (n_trajs, n_modes, traj_len)
    err_best_all  = np.array(err_best_all)   # (n_trajs, traj_len)

    # 1) best-mode oracle: for each traj and each t, pick the closest mode
    best_curve = err_best_all.mean(axis=0)       # (traj_len,)

    # 2) per-mode scalar MSE (avg over time and trajectories) — printed only
    per_mode_mse = err_modal_all.mean(axis=(0, 2))  # (n_modes,)

    # --- print oracle scalar MSE ---
    print("\n===== Oracle MSE (best mode, avg over time & trajectories) =====")
    print(f"  MSE = {best_curve.mean():.5f}   RMSE = {np.sqrt(best_curve.mean()):.5f}")
    print("=" * 55)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 4))

    # best-mode oracle curve
    ax.semilogy(t, best_curve, color=colors[9], linewidth=2.0,
                label=r'Best mode $\min_k$ (MSE=%.3e)' % best_curve.mean())

    ax.set_xlabel('time (s)')
    ax.set_ylabel('MSE')
    ax.set_title(f'Particle Filter MSE  —  {args.dataset}  '
                 f'(k={args.n_modes}, N={args.n_particles})')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, which='both', alpha=0.4)
    plt.tight_layout()
    plt.savefig('particle_filter_mse.pdf', bbox_inches='tight', format='pdf')
    plt.savefig('particle_filter_mse.png', dpi=150, bbox_inches='tight')
    print("Saved: particle_filter_mse.png / .pdf")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Run plots
# ---------------------------------------------------------------------------
plot_probability_tube(b=args.batch)
plot_mse_over_time()