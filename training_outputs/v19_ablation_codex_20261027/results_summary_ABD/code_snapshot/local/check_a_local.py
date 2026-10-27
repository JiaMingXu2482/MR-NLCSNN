"""A integration checks, without optimization or dataset/test-set reads."""
import json
import torch
import train_a_local as runner
from mr_nlcsnn.model_v19 import NLCSNN_v19, nfl_v19

torch.set_num_threads(1)
previous=json.loads((runner.HERE/'sessions/v19_B_seed7_30epochs/run_config.json').read_text())
for path,expected in previous['source_hashes'].items():
    assert runner.digest(path)==expected, path
torch.manual_seed(7); actual=runner.build_model()
torch.manual_seed(7); original=NLCSNN_v19()
assert actual.state_dict().keys()==original.state_dict().keys()
for name,tensor in actual.state_dict().items():
    torch.testing.assert_close(tensor,original.state_dict()[name],rtol=0,atol=0)
u=torch.randn(1,12,6); u[...,3]=torch.rand(1,12)
assert runner.compact(u,'A') is u
assert nfl_v19(u).shape[-1]==35==len(runner.FEATURES['A'])
with torch.no_grad():
    torch.testing.assert_close(actual(u),original(u),rtol=0,atol=0)
    state=actual.initial_state(1); sequential=[]
    for motion in u.unbind(1):
        ctx,h=actual.prepare(motion,state)
        sequential.append(actual.force_at(ctx,motion[:,3:4]).squeeze(1))
        state=actual.commit(h,state,motion[:,3])
    torch.testing.assert_close(actual(u),torch.stack(sequential,1),rtol=1e-5,atol=2e-6)
record=dict(passed=True,checks=['B/D source and protocol hashes unchanged',
    'A seed7 state_dict exactly equals unmodified upstream model',
    'all six inputs retained and 35 NFL features',
    'sequence forward exactly equals upstream',
    'step inference uses current column3 and matches sequence'],
    actual_parameters=sum(p.numel() for p in actual.parameters()),
    source_hashes=runner.provenance(),final_test_read=False)
runner.write_json(runner.HERE/'a_preflight_checks.json',record)
print(json.dumps(dict(passed=True,parameters=record['actual_parameters'],checks=len(record['checks']))))
