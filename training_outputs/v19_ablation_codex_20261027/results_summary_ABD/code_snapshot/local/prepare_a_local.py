"""Create an A-only copy of the completed B/D runner; preserve original files."""
from pathlib import Path
import difflib

HERE = Path(__file__).resolve().parent
original = (HERE / 'train_bd_local.py').read_text(encoding='utf-8')
target = HERE / 'train_a_local.py'
if target.exists():
    raise FileExistsError(target)
source = original

def replace(old, new):
    global source
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new)

replace('Local B/D Stage-1-only runner. No final-test loading, no A, no warm-start.',
        'Local A Stage-1-only runner, newly authorized. No final-test loading or warm-start.')
replace('from model_v19_bd import NLCSNN_v19_BD,INPUTS,FEATURES', '''from mr_nlcsnn.model_v19 import NLCSNN_v19

INPUTS={'A':['x','v','a','I','dI','T']}
FEATURES={'A':['x','v','a','I','dI','T','v_kin','xv','vI','vT',
    'tanh2vp','tanhvp','I_tanhvp','vn_abs_v','vn3','a_vn','I_vn_abs_v',
    'dI_v','dI_a','abs_dI_I','sign_v','I_sign_v','x2','x3','a_I',
    'abs_a','a_v','abs_a_sign_v','I_abs_a','slow','x_slow','abs_x_slow',
    'I_x_slow','I_abs_x_slow','sign_v_abs_x']}

def build_model(arm='A',**config):
    if arm!='A': raise ValueError('This runner is A-only')
    model=NLCSNN_v19(**config)
    # Metadata only; no module, weight, input, or execution changes.
    model.arm='A'; model.nfl_dim=35
    return model''')
replace("def compact(u,arm): return u[...,[0,1,3,4,5] if arm=='B' else [0,1,3,5]]", '''def compact(u,arm):
    if arm!='A' or u.shape[-1]!=6: raise ValueError('A requires all six input channels')
    return u''')
replace("paths=[Path(__file__),HERE/'model_v19_bd.py',SOURCE/'AA02Train/train_v19.py',", "paths=[Path(__file__),HERE/'train_bd_local.py',SOURCE/'AA02Train/train_v19.py',")
replace('model.force_at(ctx,motion[:,2:3])','model.force_at(ctx,motion[:,3:4])')
replace('model.commit(h,states[corner],motion[:,2])','model.commit(h,states[corner],motion[:,3])')
replace('model=NLCSNN_v19_BD(arm).to(device)','model=build_model(arm).to(device)')
replace("note='B observer dI is zero. B/D comparison is dead-channel removal, not dI information.'", "note='A uses the unchanged full-input model_v19.py; observer dI remains zero as defined upstream.'")
replace("selected=NLCSNN_v19_BD(**chosen['model_cfg'])", "selected=build_model(**chosen['model_cfg'])")
replace("choices=('B','D'),default=['B','D']", "choices=('A',),default=['A']")
compile(source,str(target),'exec')
target.write_text(source,encoding='utf-8')
(HERE/'a_runner_diff.txt').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),
    fromfile='train_bd_local.py',tofile='train_a_local.py')),encoding='utf-8')
print(target)
