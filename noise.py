import torch
import numpy as np
import numpy as np
from scipy import stats

def set_noise(self, dev_var, write_var, N, m):
    # N: number of bits per weight, m: number of bits per device
    # Dev_var: device variation before write and verify
    # write_var: device variation after write and verity
    scale = self.op.weight.abs().max().item()
    noise_dev = torch.zeros_like(self.noise).to(self.op.weight.device)
    noise_write = torch.zeros_like(self.noise).to(self.op.weight.device)
    new_sigma = 0
    for i in range(1, N//m + 1):
        new_sigma += pow(2, - i*m) ** 2
    new_sigma = np.sqrt(new_sigma)
    noise_dev = torch.randn_like(self.noise) * new_sigma * dev_var
    noise_write = torch.randn_like(self.noise) * new_sigma * write_var
    noise_dev = noise_dev.to(self.op.weight.device) * scale
    noise_write = noise_write.to(self.op.weight.device) * scale

    self.noise = noise_dev * self.mask + noise_write * (1 - self.mask)

def set_noise_multiple(self, noise_type, dev_var, rate_max=0, rate_zero=0, write_var=0, **kwargs):
    if   noise_type == "Gaussian":
        set_noise(self, dev_var, write_var, kwargs["N"], kwargs["m"])
    elif noise_type == "pepper":
        set_pepper(self, dev_var, rate_max)
    elif noise_type == "uni":
        set_uni(self, dev_var)
    elif noise_type == "SPU":
        set_SPU(self, rate_max, rate_zero, dev_var)
    elif noise_type == "SU":
        set_SU(self, rate_max, dev_var)
    elif noise_type == "SG":
        set_SG(self, rate_max, dev_var)
    elif noise_type == "LSG":
        set_LSG(self, rate_max, dev_var)
    elif noise_type == "FSG":
        set_FSG(self, rate_max, rate_zero, dev_var)
    elif noise_type == "CSG":
        set_CSG(self, rate_max, dev_var)
    elif noise_type == "BSG":
        set_BSG(self, rate_max, dev_var)
    elif noise_type == "BG":
        set_BG(self, rate_max, dev_var)
    elif noise_type == "FG":
        set_BG(self, rate_max, dev_var)
    elif noise_type == "TG":
        set_TG(self, rate_max, dev_var)
    elif noise_type == "LTG":
        set_TG(self, rate_max, dev_var)
    elif noise_type == "ATG":
        set_ATG(self, rate_max, dev_var)
    elif noise_type == "powerlaw":
        set_powerlaw(self, dev_var, rate_max)
    elif noise_type == "SL":
        set_SL(self, dev_var, rate_max, rate_zero)
    elif noise_type == "Four":
        set_four(self, dev_var, rate_max, kwargs["N"], kwargs["m"])
    else:
        raise NotImplementedError(f"Noise type: {noise_type} is not supported")

def set_four(self, dev_var, s_rate, N, m):
    dev_var = dev_var / np.sqrt((s_rate**2 * 0.4 + 0.6))
    dev_var_list = [1., s_rate, s_rate, 1.]
    scale = self.op.weight.abs().max().item()
    mask = ((1/6 < (self.op.weight.abs() / scale)) * ((self.op.weight.abs() / scale) < 5/6)).float()
    new_sigma = 0
    for i in range(1, N//m + 1):
        new_sigma += pow(2, - i*m) ** 2
    new_sigma = np.sqrt(new_sigma)
    noise_dev = torch.randn_like(self.noise) * new_sigma * dev_var
    noise_dev = noise_dev.to(self.op.weight.device) * scale
    self.noise = noise_dev * mask * dev_var_list[1] + noise_dev * (1-mask) * dev_var_list[0]

def set_pepper(self, dev_var, rate):

    scale = self.op.weight.abs().max().item()
    rate_mat = torch.ones_like(self.noise).to(self.op.weight.device) * rate
    sign_bit = torch.randn_like(self.noise).sign().to(self.op.weight.device)
    noise_dev = torch.bernoulli(rate_mat).to(self.op.weight.device) * sign_bit * dev_var * scale

    self.noise = noise_dev

def set_uni(self, dev_var):
    scale = self.op.weight.abs().max().item()
    self.noise = (torch.rand_like(self.noise) - 0.5) * 2 * dev_var * scale

def set_SPU(self, s_rate, p_rate, dev_var):
    assert s_rate + p_rate < 1
    scale = self.op.weight.abs().max().item()
    self.noise = (torch.rand_like(self.noise) - 0.5) * 2
    rate_mat = torch.rand_like(self.noise)
    zero_mat = rate_mat < p_rate
    th_mat = rate_mat > (1 - s_rate)
    self.noise[zero_mat] = 0
    self.noise[th_mat] = self.noise[th_mat].data.sign()
    self.noise = self.noise * scale * dev_var

def set_SU(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    self.noise = (torch.rand_like(self.noise) - 0.5) * 2
    rate_mat = torch.rand_like(self.noise)
    th_mat = rate_mat > (1 - s_rate)
    self.noise[th_mat] = self.noise[th_mat].data.sign()
    self.noise = self.noise * scale * dev_var

def set_powerlaw(self, dev_var, s_rate, p_rate=0.1 ):
    # here s_rate means alpha of lognormal distribution
    scale = self.op.weight.abs().max().item()
    lognorm_scale = p_rate
    np_noise = np.random.power(s_rate, self.noise.shape)
    self.noise = torch.Tensor(np_noise).to(torch.float32).to(self.noise.device) / lognorm_scale * create_sign_map(self)
    self.noise = self.noise * scale * dev_var

def set_SG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    self.noise = torch.randn_like(self.noise)
    self.noise[self.noise > s_rate] = s_rate
    # self.noise[self.noise < -s_rate] = -s_rate
    self.noise = self.noise * scale * dev_var

def set_LSG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    self.noise = torch.randn_like(self.noise)
    # self.noise[self.noise > s_rate] = s_rate
    self.noise[self.noise < -s_rate] = -s_rate
    self.noise = self.noise * scale * dev_var

def set_FSG(self, s_rate, f_rate, dev_var):
    dev_var = dev_var / np.sqrt((f_rate**2 * 0.4 + 0.6))
    scale = self.op.weight.abs().max().item()
    self.noise = torch.randn_like(self.noise)
    self.noise[self.noise > s_rate] = s_rate
    self.noise = self.noise * scale * dev_var

    dev_var_list = [1., f_rate, f_rate, 1.]
    mask = ((0.25 < (self.op.weight.abs() / scale)) * ((self.op.weight.abs() / scale) < 0.75)).float()
    self.noise = self.noise * mask * dev_var_list[1] + self.noise * (1-mask) * dev_var_list[0]

def set_CSG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    self.noise = torch.randn_like(self.noise)
    self.noise[self.noise > s_rate] = s_rate
    self.noise[self.noise < -s_rate] = -s_rate
    self.noise = self.noise * scale * dev_var

def set_BSG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    self.noise = torch.randn_like(self.noise)
    self.noise[self.noise > s_rate] = s_rate
    cdf = stats.norm.cdf(s_rate)
    bias = (1 - cdf) * s_rate - 1 / np.sqrt(2 * np.pi) * np.exp(0-(s_rate**2 / 2))
    self.noise = self.noise - bias
    self.noise = self.noise * scale * dev_var

def set_BG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    self.noise = torch.randn_like(self.noise)
    cdf = stats.norm.cdf(s_rate)
    bias = (1 - cdf) * s_rate - 1 / np.sqrt(2 * np.pi) * np.exp(0-(s_rate**2 / 2))
    # bias = 0
    # bias = -0.19779655740130608 # 0.5
    # bias = -0.142879 # 0.7
    # bias = -0.083 # 1.0
    # bias = -0.0085 # 2.0
    self.noise = self.noise + bias
    self.noise = self.noise * scale * dev_var

def set_FG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    noise = (stats.foldnorm.rvs(s_rate, size=self.op.weight.shape) - s_rate) * -1
    self.noise.data = torch.Tensor(noise).to(self.noise.device).data
    self.noise = self.noise * scale * dev_var


def set_SL(self, dev_var, s_rate, p_rate=0.1):
    # here s_rate means alpha of lognormal distribution
    scale = self.op.weight.abs().max().item()
    lognorm_scale = p_rate
    np_noise = np.random.power(s_rate, self.noise.shape)
    self.noise = torch.Tensor(np_noise).to(torch.float32).to(self.noise.device) / lognorm_scale * create_sign_map(self)
    self.noise[self.noise > 1] = 1
    self.noise = self.noise * scale * dev_var

def set_TG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    noise = stats.truncnorm.rvs(-np.inf, s_rate, size = self.op.weight.shape)
    self.noise.data = torch.Tensor(noise).to(self.noise.device).data
    self.noise = self.noise * scale * dev_var

def set_LTG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    noise = stats.truncnorm.rvs(-s_rate, np.inf, size = self.op.weight.shape)
    self.noise.data = torch.Tensor(noise).to(self.noise.device).data
    self.noise = self.noise * scale * dev_var

# def set_TG(self, s_rate, dev_var):
#     scale = self.op.weight.abs().max().item()
#     def oversample_Gaussian(target_size, th):
#         tmp = np.random.normal(size=int(target_size*1/th*2))
#         index = np.abs(tmp) < 1*th
#         tmp = tmp[index][:target_size]
#         return tmp
#     target_size = self.noise.shape.numel()
#     for _ in range(10):
#         sampled_Gaussian = oversample_Gaussian(target_size, s_rate)
#         if len(sampled_Gaussian) == target_size:
#             break
#         else:
#             sampled_Gaussian = oversample_Gaussian(target_size, s_rate)
#     assert len(sampled_Gaussian) == target_size
#     self.noise = torch.Tensor(sampled_Gaussian).view(self.noise.size()).to(device=self.op.weight.device)
#     self.noise = self.noise * scale * dev_var

def set_ATG(self, s_rate, dev_var):
    scale = self.op.weight.abs().max().item()
    def oversample_Gaussian(target_size, th):
        tmp = np.random.normal(size=int(target_size*1/th*2))
        index = tmp < (1*th)
        tmp = tmp[index][:target_size]
        return tmp
    target_size = self.noise.shape.numel()
    for _ in range(10):
        sampled_Gaussian = oversample_Gaussian(target_size, s_rate)
        if len(sampled_Gaussian) == target_size:
            break
        else:
            sampled_Gaussian = oversample_Gaussian(target_size, s_rate)
    assert len(sampled_Gaussian) == target_size
    self.noise = torch.Tensor(sampled_Gaussian).view(self.noise.size()).to(device=self.op.weight.device)
    self.noise = self.noise * scale * dev_var