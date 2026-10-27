import unittest
import torch
from mr_nlcsnn.model_v19 import nfl_v19
from model_v19_bd import NLCSNN_v19_BD,nfl_bd,FEATURES


class BDModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7); torch.set_num_threads(1)
        self.u=torch.randn(2,30,6,dtype=torch.double)*0.1
        self.u[...,3]=torch.rand(2,30,dtype=torch.double)

    def test_exact_feature_removal(self):
        full=nfl_v19(self.u)
        for arm,removed,inputs in (
            ('B',{2,15,18,24,25,26,27,28},[0,1,3,4,5]),
            ('D',{2,4,15,17,18,19,24,25,26,27,28},[0,1,3,5])):
            retained=[i for i in range(full.shape[-1]) if i not in removed]
            result=nfl_bd(self.u[...,inputs],arm)
            torch.testing.assert_close(result,full[...,retained])
            self.assertEqual(result.shape[-1],len(FEATURES[arm]))

    def test_dimensions_and_counts(self):
        for arm,nfl,count in [('B',27,19121),('D',24,18401)]:
            m=NLCSNN_v19_BD(arm)
            self.assertEqual(m.gru.input_size,nfl)
            self.assertEqual(m.c_head[0].in_features,16+nfl+1)
            self.assertEqual(m.s_head[0].in_features,16+nfl+1)
            self.assertEqual(m.b_head[0].in_features,16+nfl)
            self.assertEqual(m.inv_head[0].in_features,16+nfl+3)
            self.assertEqual(sum(p.numel() for p in m.parameters()),count)

    def test_b_di_is_dead_and_current_causal(self):
        m=NLCSNN_v19_BD('B').double()
        u=self.u[...,[0,1,3,4,5]]
        first=m(u)
        changed=u.clone(); changed[...,3]=100
        torch.testing.assert_close(first,m(changed))
        ctx,_=m.contexts(u)
        changed[:,15:,2]=0.99
        other,_=m.contexts(changed)
        torch.testing.assert_close(ctx.inverse_features[:,:16],other.inverse_features[:,:16])

    def test_step_chunk_consistency_and_inverse(self):
        for arm,inputs in [('B',[0,1,3,4,5]),('D',[0,1,3,5])]:
            m=NLCSNN_v19_BD(arm).double()
            u=self.u[...,inputs]
            full=m(u)
            a,state=m(u[:,:11],return_state=True)
            b=m(u[:,11:],state)
            torch.testing.assert_close(full,torch.cat((a,b),1))
            state=m.initial_state(2); values=[]
            for k in range(u.shape[1]):
                ctx,h=m.prepare(u[:,k],state)
                values.append(m.force_at(ctx,u[:,k:k+1,2]))
                state=m.commit(h,state,u[:,k,2])
            torch.testing.assert_close(full,torch.cat(values,1))
            ctx,_=m.contexts(u)
            current,_=m.inverse_current(ctx,full.detach())
            torch.testing.assert_close(m.force_at(ctx,current),full,atol=1e-8,rtol=1e-6)


if __name__=='__main__':
    unittest.main()
