"""
Created: November 2023
Modified: January 2026
@author: Johan Peralez
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import copy
import seaborn as sns


# -- Runge-Kutta (order 4)
def RK4(f, dt, x, u=None):
    k1 = f(x, u)
    k2 = f(x + dt / 2 * k1, u)
    k3 = f(x + dt / 2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class KKL_Dataset(Dataset):
    """
    if trajectories
        data (numpy array)
            x: shape (n_traj, traj_len, x_dim)
            y: shape (n_traj, traj_len, y_dim)
        __getitem__
            xs, ys
    else
        data (numpy array)
            x, x_next: shape = (n_samples, x_dim)
            y, y_next: shape = (n_samples, y_dim)
        __getitem__
            x, y, x_next, y_next
    """
    dt = 0 # step time
    x_dim = -1 # state dimension
    y_dim = -1 # output (measurement) dimension
    u_dim = -1 # control dimension
    x0_high = np.float32([0] * x_dim) # for initial rnd values (trajectories)
    x0_low = -x0_high
    name = "KKL_Dataset"
    
    def __init__(self,
                 n_trajs: int, # dataset size
                 traj_len: int=1, # trajectories length
                 noise_std: float=.0, # measurement noise
                 process_std: float=.0, # process noise
                 ):
        self.n_trajs = n_trajs
        self.traj_len = traj_len
        self.noise_std = noise_std
        ts, xs, ys, _ = self.generate_trajectories(n_trajs, traj_len,
               autonomous=True, noise_std=noise_std, process_std=process_std)
        self.ts, self.xs, self.ys = ts[..., None], xs, ys

    def __len__(self):
        return self.n_trajs

    def __getitem__(self, index):
        return self.ts[index], self.xs[index], self.ys[index]
        
    @staticmethod
    def get_derivs(x, u=None):
        raise NotImplementedError
        
    def get_x_next(self, x, u=None, process_std=.0):
        '''
        Parameters
            x: shape = (brodcast_dim, x_dim)
            u: shape = (brodcast_dim, u_dim)
        Returns
            x_next: shape = (brodcast_dim, x_dim)
        '''
        # x_next = x + self.dt * self.get_derivs(x, u)
        
        # def get_derivs(x, u):
        #     return self.get_derivs(x, u) + np.float32(np.random.normal(0, process_std, size=x.shape))
        # x_next = RK4(get_derivs, self.dt, x, u)
        
        def get_derivs(x, u):
            return self.get_derivs(x, u)
        x_next = RK4(get_derivs, self.dt, x, u) + self.dt * np.float32(np.random.normal(0, process_std, size=x.shape))
        
        return x_next
    
    @staticmethod
    def get_y(x):
        '''
        Get the outputs (measurements) corresponding to the states x.
        Parameters
            x: shape = (..., x_dim)
        Returns
            y: shape = (..., y_dim)
        '''
        raise NotImplementedError
    
    def get_u(self, t, x=None):
        '''
        Controller.
        Parameters
            t: float
            x: shape = (brodcast_dim, x_dim)
        Returns
            u: shape = (brodcast_dim, u_dim) or None
        '''
        raise NotImplementedError
        
    def generate_trajectories(self, n_traj, traj_len, autonomous=True,
                              noise_std=.0, process_std=.0):
        """
        returns t, x, y, u where
            t shape: (n_traj, traj_len)
            x shape: (n_traj, traj_len, x_dim)
            y shape: (n_traj, traj_len, y_dim)
            u shape: (n_traj, traj_len, u_dim)
        """
        ts = np.zeros((n_traj, traj_len), dtype=np.float32)
        xs = np.zeros((n_traj, traj_len, self.x_dim), dtype=np.float32)
        us = np.zeros((n_traj, traj_len, self.u_dim), dtype=np.float32)
        #-- init x at random values
        xs[:, 0, :] = np.random.uniform(self.x0_low,
                                        self.x0_high,
                                        xs[:, 0, :].shape)
        #-- make time-steps
        t = 0
        for k in range(traj_len - 1):
            x = xs[:, k, :]
            if autonomous:
                xs[:, k + 1, :] = self.get_x_next(x, process_std=process_std)
            else:
                us[:, k, 0] = self.get_u(t, x)
                xs[:, k + 1, :] = self.get_x_next(x, us[:, k, :], process_std=process_std)            
            t += self.dt
            ts[:, k+1] = t
        ys = copy.deepcopy(self.get_y(xs))
        if noise_std > 0:
            ys += np.float32(np.random.normal(0, noise_std, size=ys.shape))

        return ts, xs, ys, us    
        
    def render(self, n_traj=3, plot_phase=True):
        traj_idx = np.random.choice(list(range(len(self))), n_traj)
        ts = self.ts[traj_idx, :]
        xs = self.xs[traj_idx, :, :]
        ys = self.ys[traj_idx, :, :]
        
        #-- plot a trajectory
        plt.figure()
        if plot_phase and self.x_dim == 2:
            for traj in range(n_traj):
                plt.plot(xs[traj, :, 0], xs[traj, :, 1], linewidth=1)
                plt.plot(xs[traj, 0, 0], xs[traj, 0, 1], '.k') # t=t0
                plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
        else:
            n_subplots = self.x_dim + self.y_dim
            for i in range(self.x_dim):
                #-- plot states
                plt.subplot(n_subplots, 1, i + 1)
                for traj in range(n_traj):
                    plt.plot(ts[traj], xs[traj, :, i])
                plt.legend(['$x_%i$' % (i + 1,)] * n_traj)
                if i == 0:
                    plt.title(self.name)
            for i in range(self.y_dim):
                #-- plot outputs
                plt.subplot(n_subplots, 1, self.x_dim + i + 1)
                for traj in range(n_traj):
                    plt.plot(ts[traj], ys[traj, :, i])
                plt.legend(['$y_%i$' % (i + 1,)] * n_traj)
            plt.xlabel('t')
        
    @staticmethod
    def render_obs(ts, xs, xs_obs, ys, batch_number=0, mse=True):
        x_dim = xs.shape[-1]
        sns.set(style="ticks", context="talk", font_scale=1.2)
        plt.rcParams["font.family"] = "serif" #"Times New Roman" # "Palatino" # "Georgia"
        colors = sns.color_palette("Paired", n_colors=10); #sns.palplot(colors)
        
        n_traj = xs.shape[0]
        b = batch_number # the batch number to plot in detail (y, x, x_obs)
        if ts.ndim > 1:
            ts = ts[b] # assume uniform time-steps
        
        fig = plt.figure(figsize=(10, 6))
        subplots = x_dim + 2 if mse else x_dim + 1
        # -- plot y
        plt.subplot(subplots, 1, 1)
        plt.plot(ts, ys[b], '--', label='y', color=colors[3])
        plt.xticks([])
        plt.grid(which='both')
        plt.legend(loc='right')
        # -- plot x and x_obs
        for i in range(x_dim):
            plt.subplot(subplots, 1, i + 2)
            plt.plot(ts, xs[b, :, i], '--', label='$x_%i$' % (i + 1), color=colors[3])
            plt.plot(ts, xs_obs[b, :, i], label='$\hat x_%i$' % (i + 1), color=colors[5])
            if i < subplots - 2:
                plt.xticks([])
            plt.grid(which='both')
            plt.legend(loc='right')
        # -- plot error
        if mse:
            plt.subplot(subplots, 1, subplots)
            mse = ((xs - xs_obs)**2).mean(axis=0).mean(axis=-1)
            plt.semilogy(ts, mse, label='MSE (on %i trajs)' % n_traj, color=colors[9])
            plt.xlim((ts[0], ts[-1])) # handle nan values
            plt.legend(loc='right')
            plt.grid()
        plt.tight_layout()
        return fig


class OscillatorWithParameter(KKL_Dataset):
    """
    System:
        \dot x_1 = x_2
        \dot x_2 = -x_1 * x_3 
        \dot x_3 = 0
        y = x_1
    """
    x_dim, y_dim, u_dim = 3, 1, 0
    x0_high = np.float32([1.5, 1.5, 1.2])
    x0_low = np.float32([-1.5, -1.5, .8])
    dt = .01
    name = "Oscillator_With_Parameter"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def get_derivs(x, u=None):
        dxdt = 0 * x
        x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
        dxdt[..., 0] = x2
        dxdt[..., 1] = -x1 * x3
        dxdt[..., 2] = 0
        return dxdt
    
    @staticmethod
    def get_y(x):
        return x[..., :1]
    

class VanDerPol(KKL_Dataset):
    x_dim, y_dim, u_dim = 2, 1, 0
    x0_high = np.float32([2, 2])
    x0_low = -x0_high
    dt = .01
    name = "Van der Pol"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_derivs(x, u=None):
        dxdt = 0 * x
        x1, x2 = x[..., 0], x[..., 1]
        dxdt[..., 0] = x2
        dxdt[..., 1] = (1 - x1**2) * x2 - x1
        return dxdt
    
    @staticmethod
    def get_y(x):
        return x[..., :1] # y = x1
    
        
class Rossler(KKL_Dataset):
    x_dim, y_dim, u_dim = 3, 1, 0
    x0_high = np.float32([1] * 3)
    x0_low = -x0_high
    dt = .05
    name="Rossler"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def get_derivs(x, u=None):
        dxdt = 0 * x
        x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
        dxdt[..., 0] = -x2 - x3
        dxdt[..., 1] = x1 + .2 * x2
        dxdt[..., 2] = .2 + x3 * (x1 - 5.7)
        return dxdt
    
    @staticmethod
    def get_y(x):
        return x[..., 1:2] # y = x2


class LinearDyn_PolynomOut(KKL_Dataset):
    x_dim, y_dim, u_dim = 2, 1, 0
    x0_high = np.float32([.5, .5])
    x0_low = -x0_high
    dt = .01
    name = "Linear Dynamics With Nonlinear Output [Brivadis et al., CDC2019]"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def get_derivs(x, u=None):
        dxdt = 0 * x
        x1, x2 = x[..., 0], x[..., 1]
        dxdt[..., 0] = x2
        dxdt[..., 1] = -x1
        return dxdt
    
    @staticmethod
    def get_y(x):
        x1, x2 = x[..., :1], x[..., 1:]
        return x1**2 - x2**2 + x1 + x2


class LotkaVolterra(KKL_Dataset):  # "Predator prey"
    x_dim, y_dim, u_dim = 2, 1, 0
    x0_high = np.float32([1] * 2) * 6
    x0_low = x0_high*0
    dt = 0.02
    name = "Lotka Volterra"
    
    @staticmethod
    def get_derivs(x, u=None):
        c1 = 3.
        c2 = 2.
        dxdt = 0 * x
        x1, x2 = x[..., 0], x[..., 1]
        dxdt[..., 0] = x1 - x1 * x2 / c2
        dxdt[..., 1] = -x2 + x2 * x1 / c1
        return dxdt
    
    @staticmethod
    def get_y(x):
        return x[..., 0:1] + x[..., 1:2]  # y = x1 + x2
    
    def get_x_next(self, x, u=None, process_std=.0):
        '''
        Mantain x1, x2 > 0
        '''        
        def get_derivs(x, u):
            return self.get_derivs(x, u)
        x_next = RK4(get_derivs, self.dt, x, u) + self.dt * np.float32(np.random.normal(0, process_std, size=x.shape))
        lib = torch if type(x) == torch.Tensor else np
        # if lib.any(x_next <= 0):
        #     print("ok")
        x_next = lib.where(x_next > 1e-4, x_next, 1e-4)
        return x_next


class Duffing_Indistinguishable(KKL_Dataset):
    ''' Example from 
            [Data-Driven Observability Analysis for Nonlinear Stochastic Systems]
            [Uniting Observers] (Astolfi et al., BUT with a different y ????)
    '''
    x_dim, y_dim, u_dim = 2, 1, 0
    x0_high = np.float32([2, 2])
    x0_low = -x0_high
    dt = 0.02
    name = "Duffing Indistinguishable"
    
    @staticmethod
    def get_derivs(x, u=None):
        dxdt = 0 * x
        x1, x2 = x[..., 0], x[..., 1]
        dxdt[..., 0] = x2
        dxdt[..., 1] = x1 - x1**3
        return dxdt
    
    @staticmethod
    def get_y(x):
        x1, x2 = x[..., 0:1], x[..., 1:2]
        return -x1**2 / 2 + x2**2 / 2 + x1**4 / 4 # /!\ CONSTANT ALONG A TRAJECTORY


class Test(KKL_Dataset): ############## EXEMPLE AUTOMATICA de Bernard
    x_dim, y_dim, u_dim = 2, 2, 0
    x0_high = np.float32([2, 2])
    x0_low = -x0_high
    dt = .05
    name="Test"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def get_derivs(x, u=None):
        dxdt = 0 * x
        x1, x2 = x[..., 0], x[..., 1]
        dxdt[..., 0] = x2 + x1 * (1 - (x1**2 + x2**2))
        dxdt[..., 1] = -x1 + x2 * (1 - (x1**2 + x2**2))
        return dxdt
    
    @staticmethod
    def get_y(x):
        x1, x2 = x[..., 0], x[..., 1]
        y = 0 * x
        y[..., 0] = x1**2 - x2**2
        y[..., 1] = 2 * x1 * x2
        return y
    

datasets = {'VDP': VanDerPol,
           "Param": OscillatorWithParameter,
           "Rossler": Rossler,
           "LV": LotkaVolterra,
           "CDC19": LinearDyn_PolynomOut,
           "Duffing": Duffing_Indistinguishable,
           "Test": Test,
           }
    
if __name__ == "__main__":
    dataset = VanDerPol(n_trajs=40,
                   traj_len=1500,
                   noise_std=.0,
                   process_std=.0)
    dataset.render(n_traj=5, plot_phase=True)# dataset.render()
