import pynvml
import time
import configparser
import psutil


class ResourceTest(object):
    def __init__(self,pid):
        self.pid = pid
        self.p = psutil.Process(pid)

    def _get_cpu_info(self):
        try:
            cpu_percent = self.p.cpu_percent()
            m_info = self.p.memory_info()
            return cpu_percent,getattr(m_info,"rss")
        except:
            raise    

    def _get_gpu_info(self,handle):
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilinfo = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return meminfo.used,utilinfo.gpu

    def test_gpu(self,gpu_id):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        start_mem,start_gpu_util = self._get_gpu_info(handle)
        if start_gpu_util > 0:
            assert False
            
        max_gpu_mem_used,max_gpu_used = -1,-1
        max_cpu_mem_used,max_cpu_used = -1,-1
        while True:
            try:
                cpu_percent , cpu_mem_used  = self._get_cpu_info()
                max_cpu_used = max(cpu_percent,max_cpu_used)
                max_cpu_mem_used = max(max_cpu_mem_used,cpu_mem_used)
            except:
                break
            meminfo , utilinfo = self._get_gpu_info(handle)
            max_gpu_mem_used = max(max_gpu_mem_used,meminfo)
            max_gpu_used = max(max_gpu_used,utilinfo)
            time.sleep(0.05)
        mem_used = max_gpu_mem_used - start_mem
        print('gpu_mem_used',mem_used,'gpu_used',max_gpu_used)
        print('cpu_mem_used',max_cpu_mem_used,'cpu_used',max_cpu_used)

    def test_cpu(self):
        cpu_percent_max,mem_used_max = -1,-1
        while True:
            try:
                cpu_percent,mem_used = self._get_cpu_info()
                if cpu_percent > cpu_percent_max:
                    cpu_percent_max = cpu_percent
                if mem_used > mem_used_max:
                    mem_used_max = mem_used
                time.sleep(0.05)
            except:
                break
        print("cpu used:",cpu_percent_max)
        print("mem used:",mem_used_max)
if __name__=="__main__":
    config = configparser.ConfigParser()
    config.read('configs/test.config')
    section = config[config.sections()[0]]
    gpu = section.get("gpu")

    import sys
    x1= sys.argv[1]
    resource_test = ResourceTest(int(x1))
    if gpu=='':
        resource_test.test_cpu()        
    else:
        resource_test.test_gpu(int(gpu))
