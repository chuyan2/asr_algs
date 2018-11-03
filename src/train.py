import argparse
import errno
import json
import os
import time

import torch.distributed as dist
import torch.utils.data.distributed
from torch.autograd import Variable
from tqdm import tqdm
from warpctc_pytorch import CTCLoss

from data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler
from decoder import GreedyDecoder
import logging
from model import DeepSpeech, supported_rnns

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--gpu')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=2, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='configs/labels.json', help='Contains all characters for transcription')

parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')

parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--device-ids', default=None, nargs='+', type=int,
                    help='If using cuda, sets the GPU devices for the process')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--save-folder', default='test_models/', help='Location to save epoch models')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default='data/noises/',help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

def to_np(x):
    return x.data.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    args = parser.parse_args()
    if args.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    save_folder = args.save_folder
    model_path = save_folder+'model.pth'
    
    logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=save_folder+'train.log')

    main_proc = True
    loss_results, cer_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs)
    best_cer = None
   
    try:
        os.makedirs(save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            logging.info('Model Save directory already exists.')
        else:
            raise

    criterion = CTCLoss()
    if os.path.exists(model_path):  # Starting from previous model
        logging.info("Loading checkpoint model %s" % model_path)
        package = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = DeepSpeech.load_model_package(package)
        labels = DeepSpeech.get_labels(model)
        audio_conf = DeepSpeech.get_audio_conf(model)
        parameters = model.parameters()
        optimizer = torch.optim.SGD(parameters, lr=args.lr,momentum=args.momentum, nesterov=True)

        optimizer.load_state_dict(package['optim_dict'])

        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training
        avg_loss = int(package.get('avg_loss', 0))
        loss_results, cer_results = package['loss_results'], package['cer_results']
            
    else:
        avg_loss, start_epoch = 0, 0
        with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))

        rnn_type = args.rnn_type.lower()
        assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
        model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                           nb_layers=args.hidden_layers,
                           labels=labels,
                           rnn_type=supported_rnns[rnn_type],
                           audio_conf=audio_conf,
                           bidirectional=args.bidirectional)
        parameters = model.parameters()
        optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                    momentum=args.momentum, nesterov=True)

    decoder = GreedyDecoder(labels)
    audio_conf['noise_dir'] = args.noise_dir
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels)
    del audio_conf['noise_dir']
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels)


    train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    train_loader = AudioDataLoader(train_dataset,num_workers=args.num_workers, batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,num_workers=args.num_workers)
    if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
        logging.info("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    model = torch.nn.DataParallel(model, device_ids=args.device_ids).cuda()
    logging.info(model)
    logging.info("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        start_epoch_time = time.time()
        for i, (data) in enumerate(train_loader):
            batch_start_time = time.time()
            inputs, targets, input_percentages, target_sizes = data
            data_time.update(time.time() - batch_start_time)

            inputs = Variable(inputs, requires_grad=False)
            target_sizes = Variable(target_sizes, requires_grad=False)
            targets = Variable(targets, requires_grad=False)

            inputs = inputs.cuda()
            out = model(inputs)
            out = out.transpose(0, 1)  # TxNxH

            seq_length = out.size(0)
            sizes = Variable(input_percentages.mul_(int(seq_length)).int(), requires_grad=False)
            loss = criterion(out, targets, sizes, target_sizes)
            loss = loss / inputs.size(0)  # average the loss by minibatch

            loss_sum = loss.data.sum()
            inf = float("inf")
            if loss_sum == inf or loss_sum == -inf:
                loss_value = 0
            else:
                loss_value = loss.item()

            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            # SGD step
            optimizer.step()
            torch.cuda.synchronize()

            # measure elapsed time
            batch_time.update(time.time() - batch_start_time)
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time,
                    data_time=data_time, loss=losses))
            del loss
            del out

        avg_loss /= len(train_sampler)

        epoch_time = time.time() - start_epoch_time
        logging.info('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

        total_cer=0
       # model.eval()
        with torch.no_grad():
            for i, (data) in enumerate(test_loader):
                inputs, targets, input_percentages, target_sizes = data
                # unflatten targets
                split_targets = []
                offset = 0
                for size in target_sizes:
                    split_targets.append(targets[offset:offset + size])
                    offset += size

                inputs = inputs.cuda()
                out = model(inputs)  # NxTxH
                seq_length = out.size(1)
                sizes = input_percentages.mul_(int(seq_length)).int()

                decoded_output, _ = decoder.decode(out.data, sizes)
                target_strings = decoder.convert_to_strings(split_targets)
                cer = 0
                for x in range(len(target_strings)):
                    transcript, reference = decoded_output[x][0], target_strings[x][0]
                    cer += decoder.cer(transcript, reference) / float(len(reference))
                total_cer += cer

                torch.cuda.synchronize()
                del out

            cer = total_cer / len(test_loader.dataset)
            cer *= 100
            loss_results[epoch] = avg_loss
            cer_results[epoch] = cer
            logging.info('Validation Summary Epoch: [{0}]\t'
                  'Average CER {cer:.3f}\t'.format(epoch + 1, cer=cer))


        if args.checkpoint and main_proc:
            file_path = '%s/deepspeech_%d.pth.tar' % (save_folder, epoch + 1)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                            cer_results=cer_results),file_path)
        # anneal lr
        optim_state = optimizer.state_dict()
        optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / args.learning_anneal
        optimizer.load_state_dict(optim_state)
        logging.info('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

        if (best_cer is None or best_cer > cer) and main_proc:
            logging.info("Found better validated model, saving to %s" % model_path)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results, cer_results=cer_results), model_path)
            best_cer = cer

        avg_loss = 0
        if not args.no_shuffle:
            logging.info("Shuffling batches...")
            train_sampler.shuffle(epoch)
