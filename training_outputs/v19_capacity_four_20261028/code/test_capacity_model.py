import unittest
import torch
from model_v19_capacity import CapacityModel
from mr_nlcsnn.model_v19 import NLCSNN_v19,nfl_v19

class CapacityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7)
        self.u=torch.randn(2,27,6,dtype=torch.double)*.1
        self.u[...,3]=torch.rand(2,27,dtype=torch.double)

    def test_disabled_equals_old_A(self):
        for h,m in [(16,48),(32,48),(16,96)]:
            old=NLCSNN_v19(gru_hidden=h,mlp_hidden=m).double()
            new=CapacityModel(gru_hidden=h,mlp_hidden=m,history_di=False).double()
            new.load_state_dict(old.state_dict())
            torch.testing.assert_close(old(self.u),new(self.u),rtol=0,atol=0)

    def test_sequence_matches_steps_and_chunks(self):
        m=CapacityModel().double(); s=m.initial_state(2); vals=[]
        for motion in self.u.unbind(1):
            ctx,h=m.prepare(motion,s); vals.append(m.force_at(ctx,motion[:,3:4]).squeeze(1)); s=m.commit(h,s,motion[:,3])
        torch.testing.assert_close(m(self.u),torch.stack(vals,1),rtol=1e-8,atol=1e-9)
        p,s=m(self.u[:,:8],return_state=True); q=m(self.u[:,8:],s)
        torch.testing.assert_close(m(self.u),torch.cat((p,q),1),rtol=1e-8,atol=1e-9)

    def test_history_derivative_and_acceleration_active(self):
        m=CapacityModel().double(); obs,_,_=m.observer(self.u,m.initial_state(2))
        expected=torch.cat((torch.zeros(2,1,dtype=torch.double),self.u[:,:-1,3]),1)
        older=torch.cat((torch.zeros(2,1,dtype=torch.double),expected[:,:-1]),1)
        torch.testing.assert_close(obs[...,4],expected-older)
        nfl=nfl_v19(obs)
        for col in [2,4,15,17,18,19,24,25,26,27,28]: self.assertGreater(float(nfl[...,col].abs().sum()),0.)

    def test_no_current_query_leak(self):
        m=CapacityModel().double(); v=self.u.clone(); v[:,10:,3]=1-v[:,10:,3]; v[...,4]=100
        c,_=m.contexts(self.u); d,_=m.contexts(v)
        for key in ('c','s','b','inverse_features','previous_effective_current'):
            torch.testing.assert_close(getattr(c,key)[:,:11],getattr(d,key)[:,:11],rtol=0,atol=0)

    def test_backward_finite_and_inverse_frozen(self):
        m=CapacityModel().double(); out=m(self.u); out.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        c,_=m.contexts(self.u); target=m.force_at(c,self.u[...,3]); before=c.inverse_features.clone()
        current,info=m.inverse_current(c,target)
        torch.testing.assert_close(info['force_achieved'],target,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(before,c.inverse_features,rtol=0,atol=0)

if __name__=='__main__': unittest.main()
