from VQ_CVAE import VQ_CVAE
from log2 import setup_logging_and_results
import sys
import time
import argparse
import torch.utils.data
from torchvision import transforms
from auto_encoder2 import *
import ycb_loader
import torch


def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path, args):
    start_time = time.time()
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    for batch_idx, (data, label_img) in enumerate(train_loader):
        outputs = model(data)

        loss = model.loss_function(label_img, *outputs)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % log_interval == 0:
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key + '_train'] += float(latest_losses[key])
                epoch_losses[key + '_train'] += float(latest_losses[key])
            for key in latest_losses:
                losses[key + '_train'] /= log_interval
            loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
            logging.info('Train Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time:'
                         ' {time:3.2f}   {loss}'
                         .format(epoch=epoch, batch=(batch_idx + 1) * len(data),
                                 total_batch=len(train_loader) * len(data),
                                 percent=int(100. * (batch_idx + 1) / len(train_loader)),
                                 time=time.time() - start_time,
                                 loss=loss_string))
            start_time = time.time()


def main(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')

    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                              help='input batch size for training (default: 128)')
    parser.add_argument('--max-mem-batch_size', type=int, default=1, metavar='N',
                              help='input max memory batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                                     help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                                     help='enables CUDA training')
    args = parser.parse_args(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.img_res = (256, 256)
    args.num_channels_in = 4
    args.num_channels_out = 3
    args.log_interval = 10
    args.hidden = 128
    args.k = 256
    args.lr = 2e-4
    transform = transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])
    args.dataset_root_folder = "/home/paulo/datasets/ycb/train/"


    train_loader = ycb_loader.DataLoader(args.dataset_root_folder,
                                                 noise_channel=True,
                                                 batch_size=args.max_mem_batch_size,
                                                 img_res=args.img_res,
                                                 num_channels=args.num_channels_in,
                                                 transform=transform)

    results, save_path = setup_logging_and_results(args)

    model = VQ_CVAE(d=args.hidden, k=args.k, num_channels_in=args.num_channels_in, num_channels_out=args.num_channels_out)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, 0.5)

    for epoch in range(1, args.epochs + 1):
        print('Epoch: {}'.format(epoch))
        train_losses = train(epoch, model, train_loader, optimizer, args.cuda, args.log_interval, save_path, args)

        scheduler.step()





if __name__ == "__main__":
    main(sys.argv[1:])