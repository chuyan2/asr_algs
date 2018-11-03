import socket
import sys

import pyaudio
import wave
import time
import threading

class ClientSocket(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        print('client start`')
        self.chunk = 512 # 1024 causes inputoverflow error on mac
        self.format = pyaudio.paInt16
        self.channels = 1
        # something different, change this.
        self.rate = 16000
        self.record_seconds = 5
        self.finals = []

        self.s = socket.socket() 
        self.s.connect(('192.168.16.166', 8080))

    def run(self,timeout=100):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk
                        )

        print("* recording")
        loop_num = int(self.rate / self.chunk * timeout)
        datalen=0

        wavefile=wave.open('test.wav','wb')
        wavefile.setnchannels(1)
        wavefile.setsampwidth(2)
        wavefile.setframerate(16000)
        print('loop num',loop_num)
        for i in range(0, loop_num):
            data = stream.read(self.chunk,exception_on_overflow=False)
            self.s.sendall(data)
            datalen += len(data)
            wavefile.writeframes(data)
        wavefile.close()

        stream.stop_stream()
        self.s.close()
class Receiver(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        print('receiver start')
        self.s = socket.socket() 
        self.s.connect(('192.168.16.166', 20003))

    def test(self):
        help(self.s)
        self.s.send('aaa'.encode('utf-8'))
        self.receive()
    def run(self):
        while True:
            msg = self.s.recv(1024)
            if not msg:break
            print('/////////////')
            print(msg.decode('utf-8'),flush=True)
        print('receiver end')       
if __name__=='__main__':
    cs = ClientSocket()
    time.sleep(1)
    rv = Receiver()
    cs.start()  
    rv.start()
    rv.join()  
    cs.join()
