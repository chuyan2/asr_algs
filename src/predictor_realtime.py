import torch
from decoder import GreedyDecoder
from model_realtime import DeepSpeech
from data_loader import load_audio,SpectrogramParser
import os
import configparser 
import numpy as np
import time
import scipy.signal
import librosa
windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

def decode_results(decoded_output):
    results = {"output":[]}
    for b in range(len(decoded_output)):
        for pi in range(len(decoded_output[b])):
            result = {'transcription': decoded_output[b][pi]}
            results['output'].append(result)
    return results



class SpectrogramParserRealtime(SpectrogramParser):
    def __init__(self, audio_conf):

        super(SpectrogramParserRealtime,self).__init__(audio_conf)
        self.audio_np_buffer = np.array([])
   
    def flush_audio_buffer(self):
        self.audio_np_buffer = np.array([])
        
    def parse_audio(self,x):
        appended = np.append(self.audio_np_buffer,x)
        current_len = len(appended)
        if current_len < self.n_fft:
            self.audio_np_buffer = appended
            return None
        end_ix = int(current_len/self.hop_length)*self.hop_length
        spect = self.wav_vector2nn_input(appended[:end_ix])
        self.audio_np_buffer = appended[end_ix-self.hop_length:]
        return spect 


class Predictor(object):
    def __init__(self, conf_path):

        config = configparser.ConfigParser()
        config.read(conf_path)
        if len(config.sections()) != 1:
            print("warning! read the first section in ",conf_path)
        section = config[config.sections()[0]]

        model_path = section.get("model_path")
        lm_path = section.get("lm_path")
        gpu = section.get("gpu")
        beam_width = 10
        cutoff_top_n = 31
        alpha = 0.8
        beta = 1
        lm_workers = 1
        if gpu:
            os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
        torch.set_grad_enabled(False)
        self.model_path = model_path
        self.lm_path = lm_path
        if gpu:
            self.model = DeepSpeech.load_model(model_path, True)
        else:
            self.model = DeepSpeech.load_model(model_path, False)
            
        self.model.eval()
        #self.model.train()
        labels = DeepSpeech.get_labels(self.model)
        audio_conf = DeepSpeech.get_audio_conf(self.model)
        if lm_path:
            from decoder import BeamCTCDecoder
            self.decoder = BeamCTCDecoder(labels, lm_path=lm_path, alpha=alpha, beta=beta,
                                 cutoff_top_n=cutoff_top_n, cutoff_prob=1.0,
                                 beam_width=beam_width, num_processes=lm_workers)
        else:
            self.decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
        self.parser = SpectrogramParserRealtime(audio_conf)

        self.t_kernel_length = 11
        self.t_stride = 2
        self.conv_input_buffer = torch.Tensor()
        self._realtime_info = (None,None,None,None,None,None)
        self._realtime_nn_out = torch.Tensor() 
        self._decoded_len = 0


    def _control_conv_input(self,slice_data):
        assert slice_data.dim() == 2
        concatenated = torch.cat((self.conv_input_buffer, slice_data),dim=1)
        t_length = concatenated.size()[1]
        stride_num = int((t_length - self.t_kernel_length)/self.t_stride)
        next_start_ix = (stride_num+1-10) * self.t_stride
        self.conv_input_buffer = concatenated[:,next_start_ix:]
        return concatenated.view(1, 1, concatenated.size(0), concatenated.size(1))
    
    def flush_realtime(self):
        self.conv_input_buffer = torch.Tensor()
        self.parser.flush_audio_buffer()
        self._realtime_info=(None,None,None,None,None,None)
        self._realtime_nn_out = torch.Tensor()
  
    def predict_realtime(self,x,tail_padding):
        x = self._control_conv_input(self.parser.parse_audio(x).contiguous())
#        self.model.train()
#        out1 = self.model(x,tail_padding,self._realtime_info)    

        out2,self._realtime_info = self.model(x,tail_padding,self._realtime_info)    
        self._realtime_nn_out = torch.cat((self._realtime_nn_out,out2),1)

    
    def realtime_res(self):
        if len(self._realtime_nn_out) == 0:
            return ''
        t1=time.time()
        decoded_output, _ = self.decoder.decode(self._realtime_nn_out.data)
        transcriptions = decode_results(decoded_output)['output'][0]['transcription']
        print('txt:',transcriptions,'consumed',time.time()-t1)
        return transcriptions 

    def predict(self,audio_path,label=None):
        x = load_audio(audio_path)
        self.predict_realtime(x,True)
        decoded_output, _ = self.decoder.decode(self._realtime_nn_out.data)
        transcriptions = decode_results(decoded_output)['output'][0]['transcription']
        self.flush_realtime()
        if label is None:
            return transcriptions
        else:
            cer = self.decoder.cer(label,transcriptions)/float(len(label))
            return transcriptions,cer

