import argparse,sys
sys.path.insert(0,"src")
from activeqa.data import load_splits,save_splits
from activeqa.utils import load_config
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/smoke_test.yaml');a=p.parse_args()
c=load_config(a.config);tr,ev=load_splits(c['data']);save_splits(tr,ev)
print(f'Exploratory train={len(tr)} eval={len(ev)} (both drawn disjointly from test-only source)')
