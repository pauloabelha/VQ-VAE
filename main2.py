import os
import sys
import time
import argparse
import torch.utils.data
from torch import optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torchvision.utils import save_image
from log2 import setup_logging_and_results
from auto_encoder2 import *
import ycb_loader
import visualize as vis
import util

models = {'ycb': {'vqvae': VQ_CVAE},
          'imagenet': {'vqvae': VQ_CVAE},
          'cifar10': {'vae': CVAE,
                      'vqvae': VQ_CVAE},
          'mnist': {'vae': VAE,
                    'vqvae': VQ_CVAE}}
datasets_classes = {'imagenet': datasets.ImageFolder,
                    'cifar10': datasets.CIFAR10,
                    'mnist': datasets.MNIST}
dataset_train_args = {'imagenet': {},
                      'cifar10': {'train': True, 'download': True},
                      'mnist': {'train': True, 'download': True}}
dataset_test_args = {'imagenet': {},
                     'cifar10': {'train': False, 'download': True},
                     'mnist': {'train': False, 'download': True},
}

ycb_train = 'train_small/'
ycb_test = 'test_small/'

dataset_sizes = {'ycb': (4, 3, 640, 480),
                 'imagenet': (3, 3, 256, 224),
                 'cifar10': (3, 3, 32, 32),
                 'mnist': (1, 1, 28, 28)}



dataset_transforms = {'ycb': transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]),
                      'imagenet': transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                      'cifar10': transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                      'mnist': transforms.ToTensor()}
default_hyperparams = {'ycb': {'lr': 2e-4, 'k': 256, 'hidden': 128},
                       'imagenet': {'lr': 2e-4, 'k': 512, 'hidden': 128},
                       'cifar10': {'lr': 2e-4, 'k': 10, 'hidden': 256},
                       'mnist': {'lr': 1e-4, 'k': 10, 'hidden': 64}}


def main(args):
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')

    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model', default='vae', choices=['vae', 'vqvae'],
                              help='autoencoder variant to use: vae | vqvae')
    model_parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                              help='input batch size for training (default: 128)')
    model_parser.add_argument('--max-mem-batch_size', type=int, default=1, metavar='N',
                              help='input max memory batch size for training (default: 1)')
    model_parser.add_argument('--hidden', type=int, metavar='N',
                              help='number of hidden channels')
    model_parser.add_argument('-k', '--dict-size', type=int, dest='k', metavar='K',
                              help='number of atoms in dictionary')
    model_parser.add_argument('--lr', type=float, default=None,
                              help='learning rate')
    model_parser.add_argument('--vq_coef', type=float, default=None,
                              help='vq coefficient in loss')
    model_parser.add_argument('--commit_coef', type=float, default=None,
                              help='commitment coefficient in loss')
    model_parser.add_argument('--kl_coef', type=float, default=None,
                              help='kl-divergence coefficient in loss')

    training_parser = parser.add_argument_group('Training Parameters')
    training_parser.add_argument('--dataset', default='cifar10', choices=['ycb', 'mnist', 'cifar10', 'imagenet'],
                                 help='dataset to use: mnist | cifar10')
    training_parser.add_argument('--data-dir', default='/home/paulo/datasets/',
                                 help='directory containing the dataset')
    training_parser.add_argument('--epochs', type=int, default=20, metavar='N',
                                 help='number of epochs to train (default: 10)')
    training_parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='enables CUDA training')
    training_parser.add_argument('--seed', type=int, default=1, metavar='S',
                                 help='random seed (default: 1)')
    training_parser.add_argument('--gpus', default='0',
                                 help='gpus used for training - e.g 0,1,3')

    logging_parser = parser.add_argument_group('Logging Parameters')
    logging_parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                help='how many batches to wait before logging training status')
    logging_parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results',
                                help='results dir')
    logging_parser.add_argument('--save-name', default='',
                                help='saved folder')
    logging_parser.add_argument('--data-format', default='json',
                                help='in which format to save the data')
    args = parser.parse_args(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    lr = args.lr or default_hyperparams[args.dataset]['lr']
    k = args.k or default_hyperparams[args.dataset]['k']
    hidden = args.hidden or default_hyperparams[args.dataset]['hidden']
    num_channels_in = dataset_sizes[args.dataset][0]
    num_channels_out = dataset_sizes[args.dataset][1]

    args.log_num_images_to_save = 8

    results, save_path = setup_logging_and_results(args)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)

    models_dict = models[args.dataset]
    if args.dataset == 'ycb':
        model = VQ_CVAE(d=hidden, k=k, num_channels_in=num_channels_in, num_channels_out=num_channels_out)
    elif args.dataset == 'imagenet':
        model = models_dict['vqvae'](d=hidden, k=k, num_channels_in=num_channels_in, num_channels_out=num_channels_out)
    else:
        model = models_dict[args.model](d=hidden, k=k, num_channels_in=num_channels_in, num_channels_out=num_channels_out)
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10 if args.dataset == 'imagenet' else 30, 0.5,)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    dataset_train_dir = os.path.join(args.data_dir, args.dataset)
    dataset_test_dir = os.path.join(args.data_dir, args.dataset)
    if 'imagenet' in args.dataset:
        dataset_train_dir = os.path.join(dataset_train_dir, 'train')
        dataset_test_dir = os.path.join(dataset_test_dir, 'val')
    if args.dataset == 'ycb':
        train_loader = ycb_loader.DataLoader(args.data_dir + ycb_train,
                                             noise_channel=True,
                                             batch_size=args.max_mem_batch_size,
                                             img_res=(dataset_sizes[args.dataset][2], dataset_sizes[args.dataset][3]),
                                             num_channels=dataset_sizes[args.dataset][0],
                                             transform=dataset_transforms[args.dataset])
        test_loader = ycb_loader.DataLoader(args.data_dir + ycb_test,
                                            noise_channel=True,
                                            batch_size=args.max_mem_batch_size,
                                            img_res=(dataset_sizes[args.dataset][2], dataset_sizes[args.dataset][3]),
                                            num_channels=dataset_sizes[args.dataset][0],
                                            transform=dataset_transforms[args.dataset],)
        print('Length of training dataset: {}'.format(len(train_loader.dataset)))
        print('Length of test dataset: {}'.format(len(test_loader.dataset)))
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets_classes[args.dataset](dataset_train_dir,
                                           transform=dataset_transforms[args.dataset],
                                           **dataset_train_args[args.dataset]),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets_classes[args.dataset](dataset_test_dir,
                                           transform=dataset_transforms[args.dataset],
                                           **dataset_test_args[args.dataset]),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    for epoch in range(1, args.epochs + 1):
        if args.dataset == 'ycb':
            train_losses = train_ycb(epoch, model, train_loader, optimizer, args.cuda, args.log_interval, save_path, args)
            test_losses = test_net_ycb(epoch, model, test_loader, args.cuda, save_path, args, args.log_interval)
        else:
            train_losses = train(epoch, model, train_loader, optimizer, args.cuda, args.log_interval, save_path, args)
            test_losses = test_net(epoch, model, test_loader, args.cuda, save_path, args, args.log_interval)
        results.add(epoch=epoch, **train_losses, **test_losses)
        for k in train_losses:
            key = k[:-6]
            #results.plot(x='epoch', y=[key + '_train', key + '_test'])
        results.save()
        scheduler.step()


def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path, args):
    model.train()
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    start_time = time.time()
    batch_idx, data = None, None
    for batch_idx, (data, _) in enumerate(train_loader):
        #print(data[0])
        if cuda:
            data = data.cuda()
        optimizer.zero_grad()
        outputs = model(data)
        #image = ((data[0].cpu().detach().numpy() + 0.5) * 255.).astype(int)
        #vis.plot_image(image)
        #vis.show()

        #image_output = ((outputs[0][0].cpu().detach().numpy() + 0.5) * 255.).astype(int)
        #vis.plot_image(image_output)
        #vis.show()
        loss = model.loss_function(data, *outputs)
        loss.backward()
        optimizer.step()
        latest_losses = model.latest_losses()
        for key in latest_losses:
            losses[key + '_train'] += float(latest_losses[key])
            epoch_losses[key + '_train'] += float(latest_losses[key])
        if batch_idx % log_interval == 0:
            for key in latest_losses:
                losses[key + '_train'] /= log_interval
            loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
            logging.info('Train Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time:'
                         ' {time:3.2f}   {loss}'
                         .format(epoch=epoch, batch=batch_idx * len(data), total_batch=len(train_loader) * len(data),
                                 percent=int(100. * batch_idx / len(train_loader)),
                                 time=time.time() - start_time,
                                 loss=loss_string))
            start_time = time.time()
            # logging.info('z_e norm: {}'.format(float(torch.mean(torch.norm(outputs[1].contiguous().view(256,-1),2,0)))))
            # logging.info('z_q norm: {}'.format(float(torch.mean(torch.norm(outputs[2].contiguous().view(256,-1),2,0)))))
            for key in latest_losses:
                losses[key + '_train'] = 0
        if batch_idx == (len(train_loader) - 1):
            #print('----------------------------------')
            #print(np.mean(data.cpu().numpy()))
            #print(np.std(data.cpu().numpy()))
            #print('----------------------------------')
            save_reconstructed_images(data, epoch, outputs[0], save_path, 'reconstruction_train')
        if args.dataset == 'imagenet' and batch_idx * len(data) > 25000:
            break

    for key in epoch_losses:
        if args.dataset != 'imagenet':
            epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
        else:
            epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
    loss_string = '\t'.join(['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()])
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
    if len(outputs) > 3:
        model.print_atom_hist(outputs[3])
    return epoch_losses


def train_ycb(epoch, model, train_loader, optimizer, cuda, log_interval, save_path, args):
    model.train()
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    start_time = time.time()
    batch_idx, data = None, None
    data_to_save = []
    outputs_to_save = []
    for batch_idx, (data, labels) in enumerate(train_loader):
        if len(data_to_save) <= args.log_num_images_to_save:
            data_to_save.append(data[0, 0:3, :, :])
        if batch_idx == 0:
            logging.info('Processing first batch '
                         'with max memory batch size of {}'
                         ' for a batch size of {} '
                         'and logging info every {} batches'.
                         format(args.max_mem_batch_size,
                                args.batch_size,
                                args.log_interval))
        label_img, _ = labels
        if cuda:
            data = data.cuda()
            label_img = label_img.cuda()

        outputs = model(data)
        if len(outputs_to_save) <= args.log_num_images_to_save:
            outputs_to_save.append(outputs[0][0, 0:3, :, :])

        #image = ((data[0].cpu().detach().numpy() + 0.5) * 255.).astype(int)
        #image = image[0:3, :, :]
        #vis.plot_image(image)
        #vis.show()

        #label_img = ((label_img[0][0].cpu().detach().numpy() + 0.5) * 255.).astype(int)
        #vis.plot_image(label_img)
        #vis.show()
        loss = model.loss_function(label_img, *outputs)
        loss.backward()

        batch_num = (batch_idx + 1) * args.max_mem_batch_size
        if batch_num > 0 and batch_num % args.batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            num_processed_batches = int((batch_num / args.batch_size))
            if num_processed_batches % log_interval == 0:
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
                # logging.info('z_e norm: {}'.format(float(torch.mean(torch.norm(outputs[1].contiguous().view(256,-1),2,0)))))
                # logging.info('z_q norm: {}'.format(float(torch.mean(torch.norm(outputs[2].contiguous().view(256,-1),2,0)))))
                for key in latest_losses:
                    losses[key + '_train'] = 0

        if batch_idx == (len(train_loader) - 1):
            # print('----------------------------------')
            # print(np.mean(data.cpu().numpy()))
            # print(np.std(data.cpu().numpy()))
            # print('----------------------------------')
            # print('--------TRAIN-----------')
            # print(outputs[0][0])
            data = data[:, 0:3, :, :].permute(0, 1, 3, 2)
            output_to_save = outputs[0][:, 0:3, :, :].permute(0, 1, 3, 2)
            save_reconstructed_images(data, epoch, output_to_save, save_path, 'reconstruction_train_ycb')
            data_to_save = torch.cat(data_to_save).permute(0, 1, 3, 2)
            outputs_to_save = torch.cat(outputs_to_save).permute(0, 1, 3, 2)
            save_reconstructed_images(data_to_save, epoch, outputs_to_save, save_path, 'reconstruction_train_ycb_batch')

            data_to_save = []
            outputs_to_save = []
            # print('-----------------------')


    for key in epoch_losses:
        if args.dataset != 'imagenet':
            epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
        else:
            epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
    loss_string = '\t'.join(['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()])
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
    if len(outputs) > 3:
        model.print_atom_hist(outputs[3])
    return epoch_losses



def test_net(epoch, model, test_loader, cuda, save_path, args, log_interval):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    i, data = None, None
    with torch.no_grad():
        start_time = time.time()
        for i, (data, _) in enumerate(test_loader):
            if cuda:
                data = data.cuda()
            outputs = model(data)
            model.loss_function(data, *outputs)
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key + '_test'] += float(latest_losses[key])
            if i == 0:
                save_reconstructed_images(data, epoch, outputs[0], save_path, 'reconstruction_test')
            if args.dataset == 'imagenet' and i * len(data) > 1000:
                break
            if i % log_interval == 0:
                loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
                logging.info('Test Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time:'
                             ' {time:3.2f}   {loss}'
                             .format(epoch=epoch, batch=i * len(data), total_batch=len(test_loader) * len(data),
                                     percent=int(100. * i / len(test_loader)),
                                     time=time.time() - start_time,
                                     loss=loss_string))
                start_time = time.time()
    for key in losses:
        if args.dataset != 'imagenet':
            losses[key] /= (len(test_loader.dataset) / test_loader.batch_size)
        else:
            losses[key] /= (i * len(data))
    loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
    logging.info('====> Test set losses: {}'.format(loss_string))
    return losses



def test_net_ycb(epoch, model, test_loader, cuda, save_path, args, log_interval):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    i, data = None, None
    with torch.no_grad():
        start_time = time.time()
        for i, (data, labels) in enumerate(test_loader):
            label_img, _ = labels
            if cuda:
                data = data.cuda()
                label_img = label_img.cuda()
            outputs = model(data)
            model.loss_function(label_img, *outputs)
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key + '_test'] += float(latest_losses[key])
            if i < args.log_num_images_to_save:
                #print('--------TEST-----------')
                #print(outputs[0][0])
                data = data[:, 0:3, :, :].permute(0, 1, 3, 2)
                outputs_save = outputs[0][:, 0:3, :, :].permute(0, 1, 3, 2)
                save_reconstructed_images(data, epoch, outputs_save[0], save_path, 'reconstruction_test_ycb')
                #print('-----------------------')
            if args.dataset == 'imagenet' and i * len(data) > 1000:
                break
            if i % log_interval == 0:
                loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
                logging.info('Test Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time:'
                             ' {time:3.2f}   {loss}'
                             .format(epoch=epoch, batch=i * len(data), total_batch=len(test_loader) * len(data),
                                     percent=int(100. * i / len(test_loader)),
                                     time=time.time() - start_time,
                                     loss=loss_string))
                start_time = time.time()
    for key in losses:
        if args.dataset != 'imagenet':
            losses[key] /= (len(test_loader.dataset) / test_loader.batch_size)
        else:
            losses[key] /= (i * len(data))
    loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
    logging.info('====> Test set losses: {}'.format(loss_string))
    return losses

def save_reconstructed_images(data, epoch, outputs, save_path, name):
    size = data.size()
    n = min(data.size(0), 8)
    batch_size = data.size(0)
    comparison = torch.cat([data[:n],
                            outputs.view(batch_size, size[1], size[2], size[3])[:n]])
    save_image(comparison.cpu(),
               os.path.join(save_path, name + '_' + str(epoch) + '.png'), nrow=n, normalize=True)


if __name__ == "__main__":
    main(sys.argv[1:])
