"""Portable frozen audit entry; no dataset access and no training entry point.

python code/replay_periodic.py --out /new/writable/path [--legacy-center]
Requires Python 3.11 + PyTorch with a CUDA device. Never overwrite an output.
"""
import argparse,importlib,json,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--legacy-center',action='store_true'); p.add_argument('--smoke',action='store_true',help='Only D, 0.5mm, 8Hz,0A,30C; verifies packaging, not full audit')
    args=p.parse_args(); code=Path(__file__).resolve().parent; bundle=code.parent
    if args.out.exists(): raise FileExistsError(args.out)
    # Import exact Muse-patched dependencies BEFORE archived local runners add
    # their historical workstation paths. No source text or model is rewritten.
    sys.path.insert(0,str(code/'frozen_runtime'))
    importlib.import_module('AA02Train.train_v19')
    importlib.import_module('mr_nlcsnn.model_v19')
    importlib.import_module('mr_nlcsnn.v19_protocol')
    sys.path.insert(0,str(code))
    module=importlib.import_module('audit_capacity_periodic_legacy_center' if args.legacy_center else 'audit_capacity_periodic_work')
    args.out.mkdir(parents=True)
    cfg={'session_root':str(bundle/'training_delivery/sessions')}
    (args.out/'capacity_four_config_20261028.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    module.HERE=args.out.resolve(); module.OUT=module.HERE/'results'
    if args.smoke:
        module.run.ARMS={'D':(16,48)}; module.FREQS=[8]; module.AMPS=[.5]; module.CURRENTS=[0.]; module.TEMPS=[30.]
    module.main()
    (args.out/'replay_mode.json').write_text(json.dumps(dict(smoke=args.smoke,legacy_center=args.legacy_center,training=False)),encoding='utf-8')

if __name__=='__main__': main()
