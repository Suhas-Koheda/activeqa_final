import argparse,sys,json
sys.path.insert(0,'src')
from activeqa.utils import load_config,setup_cpu
from activeqa.reformulator import Reformulator
from activeqa.qa_model import TeluguQA
from activeqa.retrieval import Retriever
from activeqa.inference import run_question
from activeqa.data import load_splits
p=argparse.ArgumentParser();p.add_argument('--question',required=True);p.add_argument('--config',default='configs/cpu.yaml');a=p.parse_args();c=load_config(a.config);setup_cpu(c['seed'],c['cpu_threads'],c.get('device','cpu'));tr,_=load_splits(c['data'])
print(json.dumps(run_question(a.question,Reformulator(c['models']['reformulator'],c['models']['cache_dir'],c['models']['lora_rank'],c['models']['lora_alpha'],device=c.get('device','cpu')),TeluguQA(c['models']['qa'],c['models']['cache_dir'],device=c.get('device','cpu')),Retriever(tr),c),ensure_ascii=False,indent=2))
