import argparse
import torch
from utils import str2bool
from utils import get_dataset, get_logger, get_model, prepare_model
from utils import MEval, MEachEval, CEval
from configs import noise_config
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_epoch', action='store', type=int, default=100,
            help='# of epochs of noise validations')
    parser.add_argument('--dev_var', action='store', type=float, default=0.3,
            help='device variation [std] before write and verify')
    parser.add_argument('--device_type', action='store', type=str, default="RRAM1",
            help='type of device')
    parser.add_argument('--compute_device', action='store', default="cuda:0",
            help='device used')
    parser.add_argument('--verbose', action='store', type=str2bool, default=False,
            help='see training process')
    parser.add_argument('--model', action='store', default="MLP4", choices=["MLP3", "MLP3_2", "MLP4", "LeNet", "CIFAR", "Res18", "TIN", "QLeNet", "QCIFAR", "QRes18", "QDENSE", "QTIN", "QVGG", "Adv", "QVGGIN", "QResIN"],
            help='model to use')
    parser.add_argument('--header', action='store',type=int, default=1,
            help='use which saved state dict')
    parser.add_argument('--model_path', action='store', default="./pretrained",
            help='where you put the pretrained model')
    parser.add_argument('--use_tqdm', action='store',type=str2bool, default=False,
            help='whether to use tqdm')
    args = parser.parse_args()

    print(args)

    BS = 128
    NW = 4
    trainloader, testloader = get_dataset(args, BS, NW)
    model = get_model(args)
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
    compute_device = torch.device(args.compute_device)
    model, optimizer, w_optimizer, scheduler = prepare_model(model, compute_device, args)
    
    criteria = torch.nn.CrossEntropyLoss()
    model_group = model, criteria, optimizer, scheduler, compute_device, trainloader, testloader
    
    n_cfg = noise_config[args.device_type]
    
    clean_acc = CEval(model_group)
    M_acc = MEachEval(model_group, n_cfg.noise_type, args.dev_var, n_cfg.rate_max, n_cfg.rate_zero, 0, verbose=True, N=model.N_weight, m=n_cfg.m)
    print(f"clean acc: {clean_acc:.4f}, fast noise acc: {M_acc:.4f}")
    
    noise_acc = []
    if args.use_tqdm:
        loader = tqdm(range(args.noise_epoch))
    else:
        loader = range(args.noise_epoch)
    for _ in loader:
        model.clear_noise()
        this_noise_acc = MEval(model_group, n_cfg.noise_type, args.dev_var, n_cfg.rate_max, n_cfg.rate_zero, 0, verbose=True, N=model.N_weight, m=n_cfg.m)
        noise_acc.append(this_noise_acc)
    print(f"noise acc, mean: {np.mean(noise_acc):4f}, std: {np.std(noise_acc):4f}, max: {max(noise_acc):4f}")