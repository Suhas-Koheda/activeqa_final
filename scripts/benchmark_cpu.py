import argparse,sys,time,psutil,os
sys.path.insert(0,'src')
from activeqa.utils import load_config,setup_cpu
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/cpu.yaml');a=p.parse_args();c=load_config(a.config);setup_cpu(c['seed'],c['cpu_threads'],c.get('device','cpu'))
proc=psutil.Process();t=time.perf_counter(); x=0
for i in range(3000000): x += i%11
print({'device':'cpu','threads':c['cpu_threads'],'benchmark_seconds':round(time.perf_counter()-t,3),'rss_mb':round(proc.memory_info().rss/1e6,1),'target_models':c['models'],'dataset':'IndicQA Telugu test-only; download not performed'})
