from VQ_CVAE import VQ_CVAE
from log2 import setup_logging_and_results
import sys
import time
import argparse
import torch.utils.data
from torchvision import transforms
import ycb_loader
import torch
from torch.nn import functional as F

def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path, args):
    for batch_idx, (data, label_img) in enumerate(train_loader):
        print('Batch idx:  {}'.format(batch_idx))
        optimizer.zero_grad()
        if cuda:
            data = data.cuda()
            label_img = label_img.cuda()

        outputs = model(data)

        loss = F.mse_loss(outputs, torch.zeros((1, 128, 64, 64)).cuda())
        #loss = model.loss_function(label_img, *outputs)
        loss.backward()

        optimizer.step()


def main(args):
    print('PyTorch version: {}'.format(torch.__version__))
    print('Cuda version: {}'.format(torch.version.cuda))
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')

    parser.add_argument('--train-dir', default='/home/paulo/datasets/ycb/train/',
                                 help='directory containing the dataset')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                              help='input batch size for training (default: 128)')
    parser.add_argument('--max-mem-batch_size', type=int, default=1, metavar='N',
                              help='input max memory batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                                     help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                                     help='enables CUDA training')
    args = parser.parse_args(args)
    args.seed = 1
    args.img_res = (256, 256)
    args.num_channels_in = 4
    args.num_channels_out = 3
    args.log_interval = 10
    args.hidden = 128
    args.k = 256
    args.lr = 2e-4
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        print('Using CUDA')
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
    else:
        print('Not using CUDA')



    torch.manual_seed(args.seed)

    transform = transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])


    train_loader = ycb_loader.DataLoader(args.train_dir,
                                                 noise_channel=True,
                                                 batch_size=args.max_mem_batch_size,
                                                 img_res=args.img_res,
                                                 num_channels=args.num_channels_in,
                                                 transform=transform)

    results, save_path = setup_logging_and_results(args)

    model = VQ_CVAE(d=args.hidden, k=args.k, num_channels_in=args.num_channels_in, num_channels_out=args.num_channels_out)
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.5)

    for epoch in range(1, args.epochs + 1):
        print('Epoch: {}'.format(epoch))
        train_losses = train(epoch, model, train_loader, optimizer, args.cuda, args.log_interval, save_path, args)

        scheduler.step()





if __name__ == "__main__":
    main(sys.argv[1:])