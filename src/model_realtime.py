import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_

class Lookahead(nn.Module):
    def __init__(self, n_features, context):
        # should we handle batch_first=True?
        super(Lookahead, self).__init__()
        self.n_features = n_features
        self.weight = Parameter(torch.Tensor(n_features, context + 1))
        assert context > 0
        self.context = context
        self.register_parameter('bias', None)
        self.init_parameters()

    def init_parameters(self):  # what's a better way initialiase this layer?
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, tail_padding, pre_context=None):
        if tail_padding:
            padding = torch.zeros(self.context, *(x.size()[1:])).type_as(x.data)
            x = torch.cat((x, Variable(padding)), 0)
        if pre_context is not None:
            x = torch.cat((pre_context, x))
        seq_len = x.size(0) - self.context
        if seq_len < 0:
            raise Exception('data insufficient in lookahead,',x.size(0),self.context)
        if not self.training:
            tail_context =x[seq_len:]
        x = [x[i:i + self.context + 1] for i in range(seq_len)] 
        x = torch.stack(x)
        x = x.permute(0, 2, 3, 1)  # TxNxHxL - sequence, batch, feature, context
        x = torch.mul(x, self.weight).sum(dim=3)
        if self.training:
            return x
        else:
            return x,tail_context
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


class DeepSpeech(nn.Module):
    def __init__(self, rnn_type=nn.LSTM, labels="abc", rnn_hidden_size=768, nb_layers=5, audio_conf=None,bidirectional=False, context=10):
        super(DeepSpeech, self).__init__()

        if audio_conf is None:
            audio_conf = {}
        self._version = '0.0.1'
        self._hidden_size = rnn_hidden_size
        self._hidden_layers = nb_layers
        self._rnn_type = rnn_type
        self._audio_conf = audio_conf or {}
        self._labels = labels
        self._bidirectional = bidirectional

        sample_rate = self._audio_conf.get("sample_rate", 16000)
        window_size = self._audio_conf.get("window_size", 0.02)
        num_classes = len(self._labels)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size - 21) / 2 + 1)
        rnn_input_size *= 32

        self.rnn0 = rnn_type(input_size = rnn_input_size,hidden_size = rnn_hidden_size,bidirectional = False,bias=False)
        self.rnn_bn1 = SequenceWise(nn.BatchNorm1d(rnn_hidden_size))
        self.rnn1 = rnn_type(input_size = rnn_hidden_size,hidden_size = rnn_hidden_size,bidirectional = False,bias=False)

        self.rnn_bn2 = SequenceWise(nn.BatchNorm1d(rnn_hidden_size))
        self.rnn2 = rnn_type(input_size = rnn_hidden_size,hidden_size = rnn_hidden_size,bidirectional = False,bias=False)
        
        self.rnn_bn3 = SequenceWise(nn.BatchNorm1d(rnn_hidden_size))
        self.rnn3 = rnn_type(input_size = rnn_hidden_size,hidden_size = rnn_hidden_size,bidirectional = False,bias=False)
        
        self.rnn_bn4 = SequenceWise(nn.BatchNorm1d(rnn_hidden_size))
        self.rnn4 = rnn_type(input_size = rnn_hidden_size,hidden_size = rnn_hidden_size,bidirectional = False,bias=False)

        self.lookahead = Lookahead(rnn_hidden_size, context=context)

        fully_connected = nn.Sequential(
            nn.Hardtanh(0, 20, inplace=True),
            nn.BatchNorm1d(rnn_hidden_size),
            nn.Linear(rnn_hidden_size, num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()
        self.context = context

    def forward(self, x, tail_padding, needed =(None,None,None,None,None,None)):

        h0,h1,h2,h3,h4,pre_context= needed
        print(x.size())
        x = self.conv(x)
        
        print(x.size())
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        if self.training:
            x, _ = self.rnn0(x)
            x = self.rnn_bn1(x)
            x, _ = self.rnn1(x)
            x = self.rnn_bn2(x)
            x, _ = self.rnn2(x)
            x = self.rnn_bn3(x)
            x, _ = self.rnn3(x)
            x = self.rnn_bn4(x)
            x, _ = self.rnn4(x)
            x = self.lookahead(x,True,None)
        else:
            x, h0 = self.rnn0(x,h0)
            x = self.rnn_bn1(x)
            x, h1 = self.rnn1(x,h1)
            x = self.rnn_bn2(x)
            x, h2 = self.rnn2(x,h2)
            x = self.rnn_bn3(x)
            x, h3 = self.rnn3(x,h3)
            x = self.rnn_bn4(x)
            x, h4 = self.rnn4(x,h4)
            x, tail_context = self.lookahead(x,tail_padding,pre_context)
            print(x.size())
        x = self.fc(x)
        print(x.size())
        x = x.transpose(0, 1)
        x = self.inference_softmax(x)
        if self.training:
            return x
        else:
            return x, (h0, h1, h2, h3, h4, tail_context)

    @classmethod
    def load_model(cls, path, cuda=False):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls(rnn_hidden_size=package['hidden_size'], nb_layers=package['hidden_layers'],
                    labels=package['labels'], audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']], bidirectional=False)

        model.load_state_dict(package['state_dict'])

        model.rnn0.flatten_parameters()
        model.rnn1.flatten_parameters()
        model.rnn2.flatten_parameters()
        model.rnn3.flatten_parameters()
        model.rnn4.flatten_parameters()

        if cuda:
            model = torch.nn.DataParallel(model).cuda()
        return model

    @classmethod
    def load_model_package(cls, package, cuda=False):
        model = cls(rnn_hidden_size=package['hidden_size'], nb_layers=package['hidden_layers'],
                    labels=package['labels'], audio_conf=package['audio_conf'],
                    rnn_type=supported_rnns[package['rnn_type']], bidirectional=False)
        model.load_state_dict(package['state_dict'])
        if cuda:
            model = torch.nn.DataParallel(model).cuda()
        return model

    @staticmethod
    def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None,
                  cer_results=None, wer_results=None, avg_loss=None, meta=None):
        model_is_cuda = next(model.parameters()).is_cuda
        model = model.module if model_is_cuda else model
        package = {
            'version': model._version,
            'hidden_size': model._hidden_size,
            'hidden_layers': model._hidden_layers,
            'rnn_type': supported_rnns_inv.get(model._rnn_type, model._rnn_type.__name__.lower()),
            'audio_conf': model._audio_conf,
            'labels': model._labels,
            'state_dict': model.state_dict(),
            'bidirectional': model._bidirectional
        }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if avg_loss is not None:
            package['avg_loss'] = avg_loss
        if epoch is not None:
            package['epoch'] = epoch + 1  # increment for readability
        if iteration is not None:
            package['iteration'] = iteration
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['cer_results'] = cer_results
            package['wer_results'] = wer_results
        if meta is not None:
            package['meta'] = meta
        return package

    @staticmethod
    def get_labels(model):
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._labels if model_is_cuda else model._labels

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params

    @staticmethod
    def get_audio_conf(model):
        model_is_cuda = next(model.parameters()).is_cuda
        return model.module._audio_conf if model_is_cuda else model._audio_conf

    @staticmethod
    def get_meta(model):
        model_is_cuda = next(model.parameters()).is_cuda
        m = model.module if model_is_cuda else model
        meta = {
            "version": m._version,
            "hidden_size": m._hidden_size,
            "hidden_layers": m._hidden_layers,
            "rnn_type": supported_rnns_inv[m._rnn_type]
        }
        return meta

