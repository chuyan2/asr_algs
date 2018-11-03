import socket
from predictor import Predictor
import numpy as np
import threading
import time
import wave
import math
import audioop
from queue import Queue




_end = object()


class Receiver(threading.Thread):
    def __init__(self,queue,chunk):
        threading.Thread.__init__(self)
        self.queue = queue
        sk = socket.socket()
        sk.bind(("192.168.16.166",8080))
        sk.listen(5)
        self.cs,addr = sk.accept()
        self.chunk = chunk
        print('linked to %s'% str(addr))


    def run(self):
        print('receiver start')
        while True:
            received = self.cs.recv(self.chunk)
            if not received:break
            self.queue.put(received)

        self.queue.put(_end)
        self.cs.close()

class Sender(threading.Thread):
    def __init__(self,queue,chunk):
        threading.Thread.__init__(self)
        sk = socket.socket()
        sk.bind(("192.168.16.166",20003))
        sk.listen(3)
        self.cs,addr2 = sk.accept()
        print('sk2 lined to %s'% str(addr2))
        self.queue = queue
        self.wav_data=[]
        self.wav_data_score=[]
        self.text_all=[]
        self.sample_width = 2
        sample_rate = 16000
        self.chunk = chunk

        pause_threshold = 0.8

        seconds_per_buffer = float(self.chunk)/sample_rate

        self.pause_buffer_count = int(math.ceil(pause_threshold/seconds_per_buffer))
        self.energy_threshold = 500 
        self.wav_name=0
        self.wav_datalen_threshold = 5*sample_rate/self.chunk

    def predict_wave(self,wav_data):
        global predictor
        print("predict wave")
        wav_name = 'wav_tmp/'+str(self.wav_name)+'.wav'
        wavefile=wave.open(wav_name,'wb')
        wavefile.setnchannels(1)
        wavefile.setsampwidth(2)
        wavefile.setframerate(16000)
        wavefile.writeframes(wav_data)
        wavefile.close()
        text = predictor.predict(wav_name) 
        self.wav_name += 1
        return text

    
    def process_current(self,wav_data):
            predict_data =bytes().join(wav_data)
           # if len(predict_data) <5:return
            text = self.predict_wave(predict_data)
            self.text_all.append(text)
            print(self.text_all[-1])
            self.cs.send('\n'.join(self.text_all).encode('utf-8'))

    def run(self):
        print('sender start')
        pause_count = -1000000
        t1 = time.time()
        print(t1)
        while True:
            received = self.queue.get()
            if received is _end:
                break
            energy = audioop.rms(received,self.sample_width)
            if energy > self.energy_threshold:
                pause_count = 0
            else:
                pause_count += 1

            if len(self.wav_data) > self.wav_datalen_threshold:
                max_score = -1
                max_score_ix = -1
                i = 0
                while i < len(self.wav_data_score):
                    if self.wav_data_score[i] > max_score:
                        max_score = self.wav_data_score[i]
                        max_score_ix = i
                    i += 1
                if max_score_ix > -1:
                    self.process_current(self.wav_data[:max_score_ix+1]) 
                    self.wav_data = self.wav_data[max_score_ix+1:]
                    self.wav_data_score = self.wav_data_score[max_score_ix+1:]

            if pause_count > self.pause_buffer_count:
                self.process_current(self.wav_data)
                pause_count = -1000000
                self.wav_data = [received]     
                self.wav_data_score = [pause_count]
            else:
                self.wav_data.append(received)
                self.wav_data_score.append(pause_count)

        self.process_current(self.wav_data)
        print('sender end')
        t2=time.time()
        print("time consumed",t1,t2,t2-t1)
        self.cs.close()


if __name__=='__main__':
    import sys
    conf_path = sys.argv[1]
    predictor = Predictor(conf_path)
    chunk=512
    q = Queue()
    rv = Receiver(q,chunk)
    sd = Sender(q,chunk)
    rv.start()
    sd.start()
    rv.join()
    sd.join()
