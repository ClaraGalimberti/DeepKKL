import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy import signal
from scipy.optimize import linear_sum_assignment


# ==========================================
# Neural Networks
# ==========================================


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def create_mlp(input_dim=1, output_dim=1, net_arch=[128]*2, activation_fn=Swish):
    assert len(net_arch) > 0
    modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(nn.LayerNorm(net_arch[idx]))
        modules.append(activation_fn())
    modules.append(nn.Linear(net_arch[-1], output_dim))
    return nn.Sequential(*modules)


class Vector_Field_MLP(nn.Module):
    def __init__(self, x_dim=2, z_dim=4, net_arch=[128]*2, activation_fn=nn.GELU):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.input_dim = x_dim + 1 + z_dim + 1 # (x, tau, z, t) 
        
        self.net = create_mlp(input_dim=self.input_dim,
                              output_dim=x_dim, # ouput : velocity v
                              net_arch=net_arch,
                              activation_fn=activation_fn)

    def forward(self, x, tau, z, t):
        inp = torch.cat([x, tau, z, t], dim=-1)
        return self.net(inp)
    

# ==========================================
# Conditional Flow Matching
# ==========================================

def get_multimodal_estimates(particles, n_modes, previous_estimates=None):
    """
    Extrait et suit plusieurs modes d'une distribution de particules.
    
    Args:
        particles: Tensor (N_particles, x_dim)
        n_modes: int, nombre de modes attendus
        previous_estimates: Tensor (n_modes, x_dim) ou None. 
                            Sert à maintenir l'ordre des modes (tracking).
    
    Returns:
        sorted_centroids: Tensor (n_modes, x_dim)
                          Les estimations ordonnées pour correspondre à previous_estimates.
    """
    N, x_dim = particles.shape
    
    # --- 1. K-Means simple pour trouver les centroïdes actuels ---
    
    # Initialisation aléatoire
    indices = torch.randperm(N)[:n_modes]
    centroids = particles[indices].clone()
    
    # Quelques itérations suffisent pour converger sur des modes bien séparés
    for _ in range(5):
        # Calcul des distances (N, n_modes)
        dists = torch.cdist(particles, centroids)
        # Assignation au cluster le plus proche
        labels = torch.argmin(dists, dim=1)
        
        # Mise à jour des centroïdes
        for k in range(n_modes):
            mask = (labels == k)
            if mask.any():
                centroids[k] = particles[mask].mean(dim=0)
            # Si un cluster est vide (rare), on garde l'ancienne position
            
    # --- 2. Tracking / Matching (Association de données) ---
    
    if previous_estimates is None:
        # Pas d'historique : on retourne les centroïdes tels quels (ou triés par x1 pour être propre)
        # On peut trier par la première coordonnée pour avoir une consistance initiale
        sort_idx = torch.argsort(centroids[:, 0])
        return centroids[sort_idx]
    
    else:
        # On doit associer chaque nouveau centroïde à son correspondant historique
        # pour éviter que les courbes ne s'interchangent (chattering d'indices).
        
        # Calcul de la matrice de coût (Distance entre Ancien[i] et Nouveau[j])
        # shape : (n_modes, n_modes)
        cost_matrix = torch.cdist(previous_estimates, centroids).cpu().numpy()
        
        # Algorithme Hongrois pour trouver l'assignation optimale (minimise la somme des distances)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # row_ind correspond aux indices de previous_estimates (0, 1, 2...)
        # col_ind correspond aux indices de centroids qui matchent
        
        # On réordonne les centroids actuels pour qu'ils s'alignent avec previous_estimates
        sorted_centroids = centroids[col_ind]
        
        return sorted_centroids


class Conditional_Flow_Matching:

    @staticmethod
    def solve_ode(model, z, t, n_steps=20, sample_every_t=True):
        """ Solve the ODE dx/dtau = v(xt, tau, z) """
        model.eval()
        device = z.device
        
        # -- Init: x(0) ~ N(0, I)
        if sample_every_t:
            x = torch.randn(*z.shape[:-1], model.x_dim).to(device)
        else:
            # Clara: for the whole trajectory we pick the same random \xi_0
            x = torch.randn(*z.shape[:-2], 1,  model.x_dim).repeat(1, z.shape[1], 1).to(device)
        
        # -- Integrate: Euler from tau=0 to tau=1
        dt = 1.0 / n_steps
        t_grid = torch.linspace(0, 1, n_steps+1).to(device)
        with torch.no_grad():
            for i in range(n_steps):
                tau = torch.ones(*z.shape[:-1], 1).to(device) * t_grid[i]
                v = model(x, tau, z, t) # velocity
                x = x + v * dt
                
        return x

    @staticmethod
    def solve_ode_with_prior_on_centers(model, z, t, n_steps=20, sample_every_t=True, sigma=0.5, x_prev=None):
        """
        Solve the ODE dx/dtau = v(xt, tau, z)
        We use the information of the previous x(t) for initalizing the centers of the Gaussians
        representing the priors (i.e. xi_0)
        """
        model.eval()
        device = z.device

        if x_prev is None:
            # First step: we start from pure noise
            x = torch.randn(*z.shape[:-1], model.x_dim).to(device)
        else:
            # next steps: we start from x_prev + noise
            x = x_prev + torch.randn_like(x_prev) * sigma

        # -- Integrate: Euler from tau=0 to tau=1
        dt = 1.0 / n_steps
        t_grid = torch.linspace(0, 1, n_steps + 1).to(device)
        with torch.no_grad():
            for i in range(n_steps):
                tau = torch.ones(*z.shape[:-1], 1).to(device) * t_grid[i]
                v = model(x, tau, z, t)  # velocity
                x = x + v * dt

        return x

    @staticmethod
    def solve_ode_median(model, z, t, n_steps=20, n_candidates=50):
        """ 
        méthode "Batch Geometric Median"
            Génère un nuage de particules et retourne la médiane géométrique 
            (le point du nuage qui minimise la somme des distances aux autres).
            Robuste : tombe forcément dans la distribution (sur un des modes).
        """
        model.eval()
        device = z.device
        broad_shape = z.shape[:-1] # z is of shape (..., z_dim)
        z = z.view(-1, z.shape[-1]) # (batch, z_dim)
        t = t.view(-1, 1) # (batch, 1)
        x = torch.zeros(z.shape[0], model.x_dim).to(device) # (batch, x_dim)
        
        # -- Memory parameters
        N = 1000 # Memory chunk size (to reduce if memory issue)
        
        # -- Integrate: Euler from tau=0 to tau=1
        dt = 1.0 / n_steps
        t_grid = torch.linspace(0, 1, n_steps+1).to(device)
        with torch.no_grad():
            for k in range(0, z.size(0), N):
                z_chunk = z[k : k + N] # (N, z_dim)
                t_chunk = t[k : k + N] # (N, 1)
                z_expanded = z_chunk.unsqueeze(0).expand(n_candidates, -1, -1) # (n_candidates, N, z_dim)
                t_expanded = t_chunk.unsqueeze(0).expand(n_candidates, -1, -1) # (n_candidates, N, 1)
                
                actual_N = z_chunk.size(0)
                x_batch = torch.randn(n_candidates, actual_N, model.x_dim).to(device)
                for i in range(n_steps):
                    tau = torch.ones(n_candidates, actual_N, 1).to(device) * t_grid[i]
                    v = model(x_batch, tau, z_expanded, t_expanded)
                    x_batch = x_batch + v * dt
                
                x_permuted = x_batch.permute(1, 0, 2)  # (N, Candidates, x_dim)
                dists = torch.cdist(x_permuted, x_permuted) # (N, Candidates, Candidates)
                sum_dists = dists.sum(dim=2) # (N, Candidates)
                best_idx = torch.argmin(sum_dists, dim=1) # (N,)
                x_selected = x_permuted[torch.arange(actual_N), best_idx]
                x[k : k + N] = x_selected
                
        return x.view(*broad_shape, model.x_dim) # (..., x_dim)

    
    def compute_exact_density_map(model, z, t_norm, x_min, x_max, normalizer, sigma=1.0, grid_size=100, steps=20, tau_end=1.0):
        # TODO : option RK4/Euler
        """
        Calcule la log-probabilité exacte avec intégrateur RK4.
        if x prev norm is none, then we are in the original case, where gaussians for sampling xi0
        are always centered at zero
        """
        model.eval()
        device = z.device
        
        # 1. Création de la grille
        x_linspace = np.linspace(x_min, x_max, grid_size)        
        X1, X2 = np.meshgrid(x_linspace[:, 0], x_linspace[:, 1])
        x_flat = np.vstack([X1.ravel(), X2.ravel()]).T
        x = torch.FloatTensor(x_flat).to(device)
        x = normalizer.normalize(x)
        
        # Condition z répétée
        z_batch = z.repeat(x.shape[0], 1)
        t_batch = t_norm.repeat(x.shape[0], 1)

        # Fonction dynamique conjointe (Vitesse + Divergence)
        def f(tau_scalar, x_in):
            """ Retourne dx/dt et d(log_prob)/dt """
            x_in = x_in.detach().requires_grad_(True)
            tau_batch = torch.ones(x_in.shape[0], 1).to(device) * tau_scalar
            
            # Vitesse
            v = model(x_in, tau_batch, z_batch, t_batch)
            
            # Divergence (Trace du Jacobien) - Optimisé 2D
            # (Attention: pour d>2, utiliser l'estimateur de Hutchinson)
            grad_v1 = torch.autograd.grad(v[:, 0].sum(), x_in, create_graph=False, retain_graph=True)[0]
            grad_v2 = torch.autograd.grad(v[:, 1].sum(), x_in, create_graph=False, retain_graph=False)[0]
            div = grad_v1[:, 0] + grad_v2[:, 1]
            
            return v, div

        # 3. Intégration RK4 Inverse (tau=1 -> tau=0) 
        dtau = tau_end / steps
        
        current_x = x.clone()
        log_prob_change = torch.zeros(x.shape[0]).to(device)
        
        for i in range(steps):
            # Temps courant (on recule)
            tau = 1.0 - i * dtau
            
            # RK4 Step (Backward: on soustrait les termes)
            v1, div1 = f(tau, current_x)
            x2 = current_x - v1 * (dtau / 2)
            v2, div2 = f(tau - dtau/2, x2)
            x3 = current_x - v2 * (dtau / 2)
            v3, div3 = f(tau - dtau/2, x3)
            x4 = current_x - v3 * dtau
            v4, div4 = f(tau - dtau, x4)
            
            # Mise à jour état
            current_x = current_x - (dtau / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
            
            # Mise à jour log-prob (Accumulation divergence)
            # On intègre div*dtau. (Signe + car on recule dans le temps et formule CNF)
            # Formellement : Integrale_1^0 div dtau = - Integrale_0^1 div dtau
            # Or log p1 = log p0 - Integrale_0^1 div.
            # Donc log p1 = log p0 + Integrale_1^0 div.
            # Ici on "accumule" l'intégrale backward (qui sera négative si div>0).
            current_delta = (dtau / 6.0) * (div1 + 2*div2 + 2*div3 + div4)
            log_prob_change += current_delta.detach()

        # 4. Calcul Final
        # current_x est maintenant à tau=0
        log_p0 = -0.5 * torch.sum(current_x ** 2, dim=1) - 0.5 * 2 * np.log(2 * np.pi)
        
        # log p1 = log p0 + Terme_Integral_Backward
        # Dans le code précédent, vous aviez log_px = log_p0 - log_change
        # Si log_change est la somme positive des divergences (comme dans Euler avant),
        # il faut faire attention aux signes.
        # Avec ma formulation RK4 ci-dessus, j'ai "ajouté" (dtau positif) les divergences calculées en backward.
        # Si le flux contracte en forward (div < 0), il expand en backward.
        # Mais div est calculée au point x. La divergence du champ ne change pas de signe, c'est le sens du temps qui change.
        #
        # Règle simple CNF : log p(x(1)) = log p(x(0)) - Integrale_0^1 div(v) dtau
        # Mon calcul `log_prob_change` approxime Integrale_0^1 div(v) dtau
        # (car j'ai sommé div * dtau positif, ce qui revient à intégrer sur le chemin).
        
        log_px = log_p0 - log_prob_change
        log_px = log_px - torch.sum(torch.log(normalizer.std)).item()
        density_map = torch.exp(log_px).reshape(grid_size, grid_size).detach().cpu().numpy()
        return X1, X2, density_map


# ==========================================
# Linear dynamics (latent state z)
# ==========================================
def get_bessel_dynamics(z_dim, dt, noise_std=0.01, base_bandwidth=5.0, device='cpu'):
    '''
    Compute A and B matrices, to obtain the Bessel filter
        z+ = A z + B u      (discrete-time)
    '''
    
    alpha = 10.0
    cutoff_freq = base_bandwidth / (1.0 + alpha * noise_std)
    _, poles, _ = signal.bessel(N=z_dim, Wn=cutoff_freq, analog=True,
                                output='zpk', norm='phase')
    
    # -- Continuous time
    real_poles = poles[np.abs(poles.imag) < 1e-6].real
    complex_poles = poles[np.abs(poles.imag) >= 1e-6]
    print("Bessel filter, real poles", real_poles)
    print("Bessel filter, complex poles", complex_poles)
    
    A_c = np.zeros((z_dim, z_dim))
    current_idx = 0
    for p in real_poles:
        A_c[current_idx, current_idx] = p
        current_idx += 1
    for p in complex_poles[complex_poles.imag > 0]:
        sigma = p.real
        omega = p.imag
        # Bloc [sigma  omega]
        #      [-omega sigma]
        A_c[current_idx, current_idx]     = sigma
        A_c[current_idx, current_idx+1]   = omega
        A_c[current_idx+1, current_idx]   = -omega
        A_c[current_idx+1, current_idx+1] = sigma
        current_idx += 2

    B_c = np.ones((z_dim, 1))

    # -- Discretize (ZOH)
    sys_c = (A_c, B_c, np.eye(z_dim), np.zeros((z_dim, 1)))
    sys_d = signal.cont2discrete(sys_c, dt=dt, method='zoh')
    A_d = torch.tensor(sys_d[0], dtype=torch.float32, device=device)
    B_d = torch.tensor(sys_d[1], dtype=torch.float32, device=device)
    
    return A_d, B_d


class KKL_Latent_Dynamics:
    
    def __init__(self, z_dim, dt, noise_std=0.01, device='cpu'):
        self.z_dim = z_dim
        self.device = device
        
        #  -- Compute A and B matrices (the Bessel filter)
        self.A, self.B = get_bessel_dynamics(z_dim=z_dim,
                                             dt=dt,
                                             device=device,
                                             noise_std=noise_std)
        # -- Diagonalize (A_diag = complex values)
        self.eigenvals, self.P = torch.linalg.eig(self.A)
        self.A_diag = torch.linalg.inv(self.P)
    
    
    def compute_z(self, ys): # ys shape(batch, traj, y_dim)
        batch_size, traj_len, _ = ys.shape
        zs = torch.zeros((batch_size, traj_len, self.z_dim)).to(self.device) # z0 = 0
        for k in range(1, traj_len):
            z, y = zs[:, k-1, :], ys[:, k-1, :]
            z = (self.A @ z.unsqueeze(-1)).squeeze(-1) + (self.B @ y.unsqueeze(-1)).squeeze(-1)
            zs[:, k, :] = z
        return zs
    
    def compute_z_fast(self, ys):  # ys shape(batch, traj, y_dim)
        batch_size, traj_len, _ = ys.shape

        # Complex
        ys_complex = ys.to(dtype=torch.cfloat)
        B_complex = self.B.to(dtype=torch.cfloat)
        
        # 2. Projection de l'entrée dans la base propre (u_tilde)
        input_projector = self.A_diag @ B_complex
        # Einsum: batch(b), traj(t), y_dim(y), z_dim(z) -> b t z
        u_tilde = torch.einsum('zy, bty -> btz', input_projector, ys_complex)

        # 3. Création du noyau de convolution (Kernel)
        # Kernel = [1, lambda, lambda^2, ...] pour chaque dimension propre
        # Forme : (Traj, z_dim)
        range_vec = torch.arange(traj_len, device=self.device)
        # Broadcasting: (1, z_dim) ** (traj, 1) -> (traj, z_dim)
        kernel = self.eigenvals.unsqueeze(0) ** range_vec.unsqueeze(-1)

        # 4. Convolution via FFT
        # Théorème de convolution : conv(u, k) = IFFT( FFT(u) * FFT(k) )
        # On double la taille (2*L) pour éviter l'aliasing circulaire (padding implicite)
        n_fft = 2 * traj_len
        
        u_f = torch.fft.fft(u_tilde, n=n_fft, dim=1)
        k_f = torch.fft.fft(kernel, n=n_fft, dim=0)
        
        # Multiplication élément par élément dans le domaine fréquentiel
        # k_f est broadcasté sur la dimension batch
        y_f = u_f * k_f.unsqueeze(0)
        
        # Retour au domaine temporel
        z_tilde_conv = torch.fft.ifft(y_f, n=n_fft, dim=1)
        
        # On garde seulement la partie valide (longueur traj_len)
        z_tilde_conv = z_tilde_conv[:, :traj_len, :]

        # 5. Gestion du décalage (Causalité)
        # La boucle originale fait z[k] = A*z[k-1] + B*y[k-1].
        # Cela signifie que z[k] dépend de y[0]...y[k-1].
        # Notre convolution aligne y[0] avec kernel[0] au temps t=0.
        # Il faut donc décaler le résultat vers la droite de 1 pas.
        
        z_tilde_shifted = torch.zeros_like(z_tilde_conv)
        z_tilde_shifted[:, 1:, :] = z_tilde_conv[:, :-1, :]
        # z_tilde_shifted[:, 0, :] reste à 0 (condition initiale z0=0)

        # 6. Projection inverse vers l'espace d'origine (Reconstruction)
        # z = P @ z_tilde
        zs_complex = torch.einsum('zq, btq -> btz', self.P, z_tilde_shifted)

        # Retourner la partie réelle (si le système physique est réel)
        return zs_complex.real
