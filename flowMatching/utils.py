import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pickle
import copy
import os


to_torch = lambda a: torch.tensor(a, dtype=torch.float32)
to_numpy = lambda a: a.detach().cpu().numpy()

def set_seed(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Normalizer:
    def __init__(self, x_data, time_period=0):
        x_data = torch.tensor(x_data, dtype=torch.float32)
        flat_x = x_data.view(-1, x_data.shape[-1])
        self.mean = flat_x.mean(dim=0)
        self.std = flat_x.std(dim=0)
        self.std[self.std < 1e-6] = 1e-6
        self.time_period = time_period
        
    def normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
    
    def normalize_t(self, t):
        if self.time_period > 0:
            # return torch.clamp(1 - t / self.time_period, 0, 1)
            return torch.exp(- 3 * t / self.time_period)
        else:
            return 0 * t
        
    def unnormalize(self, x_norm):
        return x_norm * self.std.to(x_norm.device) + self.mean.to(x_norm.device)

    def log_prob_correction(self):
        return -torch.sum(torch.log(self.std))
    
    def render(self, t_max=10.0):
        print("Normalizer: std", self.std, "mean", self.mean)
        if t_max > 0: 
            t = torch.arange(0, t_max, t_max/250)
            plt.plot(t, self.normalize_t(t))
            plt.title('Time Normalizer')
            plt.xlabel('t'); plt.ylabel('t norm')
            plt.show()


class Train_Stats:
    def __init__(self, loss_names={'forecast', 'continuous'}, save_folder='trained_models', model_name='test'):
        self.current = {name: [] for name in loss_names}
        self.train = copy.deepcopy(self.current)
        self.valid = copy.deepcopy(self.current)
        self.train['epochs'] = []
        self.valid['epochs'] = []
        self.times = []
        self.lrs = []
        os.makedirs(save_folder, exist_ok=True)
        self.filepath = os.path.join(save_folder, model_name + ".stats")
        self.filepath_png = os.path.join(save_folder, model_name + ".png")
        
    def batch(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == torch.Tensor:
                v = v.item()
            self.current[k].append(v)
            
    def epoch(self, epoch=0, mode='train', time=None, lr=None):
        if mode == "train":
            print("ep %i (%s)" % (epoch, mode), end='')
        else:
            print("\t (%s)" % mode, end='')
        c = self.current
        if mode == 'train':
            d = self.train
        else:
            d = self.valid
        d['epochs'].append(epoch)
        for k in c.keys():
            mean = np.mean(c[k])
            d[k].append(mean)
            c[k] = []
            print(" %s %.2e" %(k, mean), end='')
        if time:
            self.times.append(time)
            print(" | t %.1f" % time, end="")
        if lr:
            self.lrs.append(lr)
            print("| lr %.1e" % lr, end="")
        print()        
    
    def save(self):
        with open(self.filepath, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self):
        with open(self.filepath, 'rb') as f:
            return pickle.load(f)
    
    def render(self, loss_names={'forecast', 'continuous'}, save=True):
        sns.set(style="ticks", context="talk", font_scale=1.2)
        plt.rcParams["font.family"] = "serif" #"Times New Roman" # "Palatino" # "Georgia"
        colors = sns.color_palette("Paired", n_colors=10); #sns.palplot(colors)
        plt.figure(figsize=(10, 6))
        title = "train/valid:"
        for i, name in enumerate(sorted(loss_names)):
            plt.semilogy(self.train['epochs'], self.train[name],
                         color=colors[2*i], label=name + " train")
            title += " " + name + " %.2e" % np.min(self.train[name])
            plt.semilogy(self.valid['epochs'], self.valid[name],
                         color=colors[2*i+1], label=name + " valid")
            title += "/%.2e" % np.min(self.valid[name])
        plt.legend()
        plt.title(title)
        plt.xlabel('epochs')
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        if save:
            plt.savefig(self.filepath_png)
        plt.show()
