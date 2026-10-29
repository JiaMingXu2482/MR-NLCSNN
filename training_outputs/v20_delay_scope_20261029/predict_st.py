"""Frozen seed-7 A/E inference on ST train/validation only; never final_test."""
from pathlib import Path
import sys,csv,json,hashlib,array
import torch
HERE=Path(__file__).resolve().parent
SRC=Path(r'D:\OneDrive - mail.scut.edu.cn\Thesis\MR_NLCSNN\v20_training\v20')
DATA=Path(r'D:\v20_data\Parsed_v19_F42_I42')
RUNS=Path(r'D:\v20_data\v20_runs_20261029')
sys.path.insert(0,str(SRC))
from v20_models import V20Model
from v20_data import sample
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(1)
manifest=json.loads((DATA/'alignment_manifest.json').read_text(encoding='utf-8'))
selected=[r for r in manifest if r['split'] in ('train','validation') and Path(r['relative_path']).name.startswith('ST_')]
out=HERE/'predictions';out.mkdir(exist_ok=True)
records=[];checkpoints={}
device='cuda' if torch.cuda.is_available() else 'cpu'
for label in ['A','E']:
    path=RUNS/label/'best.pth';digest=sha(path)
    ck=torch.load(path,map_location='cpu',weights_only=True)
    assert ck['data_manifest_sha256']==sha(DATA/'alignment_manifest.json')
    m=V20Model(arch=ck['arch']);m.load_state_dict(ck['state_dict']);m.to(device).eval()
    checkpoints[label]=dict(path=str(path),sha256=digest,epoch=ck['epoch'],norm_cfg=ck['norm_cfg'],arch=ck['arch'])
    with torch.inference_mode():
        for r in selected:
            p=DATA/r['relative_path'];assert sha(p)==r['processed_sha256']
            with p.open(encoding='utf-8-sig',newline='') as fh:tab=list(csv.DictReader(fh))
            values=torch.tensor([[float(a[k]) for k in ('rod_length','velocity','accel','current','temp','force')] for a in tab])
            u,_,_=sample({'values':values},ck['norm_cfg'])
            pred=(m(u.to(device))[0]*ck['norm_cfg']['f_scale']).cpu().tolist()
            dest=out/(p.stem+'__'+label+'.f32')
            with dest.open('wb') as fh:array.array('f',pred).tofile(fh)
            records.append(dict(file=p.stem,split=r['split'],curve=label,n=len(pred),path=dest.name,sha256=sha(dest)))
    assert sha(path)==digest
    print(label,'inferred',len(selected),'files',flush=True)
(out/'manifest.json').write_text(json.dumps(dict(checkpoints=checkpoints,files=records,
    dtype='native float32 little-endian on Windows',source={n:sha(SRC/n) for n in ['v20_models.py','v20_data.py']},
    final_test_read=False),ensure_ascii=False,indent=2),encoding='utf-8')
