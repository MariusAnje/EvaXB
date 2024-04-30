import argparse
import torch
from utils import str2bool
from utils import get_dataset, get_logger, get_model, prepare_model
from utils import MTrain, UpRange, CEval, MEachEval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epoch', action='store', type=int, default=20,
            help='# of epochs of training')
    parser.add_argument('--noise_epoch', action='store', type=int, default=100,
            help='# of epochs of noise validations')
    parser.add_argument('--train_var', action='store', type=float, default=0.1,
            help='device variation [std] when training')
    parser.add_argument('--dev_var', action='store', type=float, default=0.3,
            help='device variation [std] before write and verify')
    parser.add_argument('--write_var', action='store', type=float, default=0.03,
            help='device variation [std] after write and verify')
    parser.add_argument('--device_type', action='store', default="RRAM1",
            help='type of device, e.g., RRAM1')
    parser.add_argument('--compute_device', action='store', default="cuda:0",
            help='device used')
    parser.add_argument('--verbose', action='store', type=str2bool, default=False,
            help='see training process')
    parser.add_argument('--model', action='store', default="MLP3", choices=["MLP3", "MLP3_2", "MLP4", "LeNet", "CIFAR", "Res18", "TIN", "QLeNet", "QCIFAR", "QRes18", "QDENSE", "QTIN", "QVGG", "Adv", "QVGGIN", "QResIN"],
            help='model to use')
    parser.add_argument('--alpha', action='store', type=float, default=1e6,
            help='weight used in saliency - substract')
    parser.add_argument('--header', action='store',type=int, default=1,
            help='use which saved state dict')
    parser.add_argument('--pretrained', action='store',type=str2bool, default=True,
            help='if to use pretrained model')
    parser.add_argument('--model_path', action='store', default="./pretrained",
            help='where you put the pretrained model')
    parser.add_argument('--save_file', action='store',type=str2bool, default=True,
            help='if to save the files')
    parser.add_argument('--use_tqdm', action='store',type=str2bool, default=False,
            help='whether to use tqdm')
    args = parser.parse_args()

    print(args)

    header = args.header

    BS = 128
    NW = 4
    trainloader, testloader = get_dataset(args, BS, NW)
    model = get_model(args)
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    compute_device = torch.device(args.compute_device)
    model, optimizer, w_optimizer, scheduler = prepare_model(model, compute_device, args)
    criteria = torch.nn.CrossEntropyLoss()
    model_group = model, criteria, optimizer, scheduler, compute_device, trainloader, testloader

    MTrain(model_group, args.train_epoch, header, "Gaussian", args.train_var, 1, 1, 0, verbose=True, N=1, m=1)

    state_dict = torch.load(f"tmp_best_{header}.pt")
    model.load_state_dict(state_dict)
    model.clear_noise()
    print(f"Fast: No mask no noise: {CEval(model_group):.4f}")
    performance = MEachEval(model_group, "Gaussian", args.train_var, 1, 1, 0, N=1, m=1)
    print(f"Fast: No mask noise acc: {performance:.4f}")
    
    model.make_slow()

    UpRange(model_group)
    torch.save(model.state_dict(), f"saved_B_{header}_noise_{args.train_var}.pt")

    print(f"Slow: No mask no noise: {CEval(model_group):.4f}")