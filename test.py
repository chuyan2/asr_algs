import sys
import importlib
importlib.reload(sys)

import error_rate
from predictor_realtime import Predictor

class Demo():
    def __init__(self,conf_path='configs/test.config'):
        self.predictor = Predictor(conf_path)
    def test(self,audio_path):
        if audio_path.endswith("wav"):
            return self.predictor.predict(audio_path)
        else:
            assert False

    def test_result(self,audio_path):
        assert audio_path.endswith("csv")
        total_cer ,case_num = 0,0
        import time
        with open(audio_path, 'rb') as f:###############################
            lines = [x.decode('utf-8').strip() for x in f.readlines()]#####################
            #lines = f.read().split('\n')
            for line in lines:
                if ',' in line:
                    wav,txt = line.split(',')
                    with open(txt) as t:
                        label = t.read()
                        if label[-1] == '\n':
                            label = label[:-1]

                    print("label",label)
                    p,cer = self.predictor.predict(wav,label)
                    print("cer1",cer)

                    cer2 = error_rate.cer(label,p)
                    print("cer2",cer2)
                    print(p)
                    total_cer += cer
                    case_num += 1
        average_cer = total_cer/case_num
        return average_cer

        
if __name__ == "__main__":
    demo = Demo()
    import sys
    import unicodecsv
    data = sys.argv[1]
    if data.endswith("wav"):
        import time
        t1=time.time()
        print(demo.test(data))
        print(data,'consumed ',time.time()-t1)
    elif data.endswith("csv"):
        print("average cer:", demo.test_result(data))
    else:
        print("input wrong")
