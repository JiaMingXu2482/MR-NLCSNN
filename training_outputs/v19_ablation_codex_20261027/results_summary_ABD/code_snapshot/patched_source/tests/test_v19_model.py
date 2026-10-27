import unittest
import torch
from mr_nlcsnn.model_v19 import NLCSNN_v19, CurrentLag, nfl_v19
from mr_nlcsnn.model_v12 import static_nfl_v12
from mr_nlcsnn.model_v10 import CurrentLag as LegacyLag
from mr_nlcsnn.v19_protocol import set_stage, reference_labels, inverse_loss, joint_gate
from AA02Train.audit_v19_rollout import rollout
from AA02Train.train_v19 import warm_transfer, sample, evaluate


class V19Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.m = NLCSNN_v19().double().eval()
        self.u = torch.randn(2, 48, 6, dtype=torch.double)*0.1
        self.u[..., 3] = torch.rand(2, 48, dtype=torch.double)

    def test_nfl_actual_source_equivalence(self):
        torch.testing.assert_close(nfl_v19(self.u), static_nfl_v12(self.u))

    def test_no_present_or_future_current_leak(self):
        a, _ = self.m.contexts(self.u)
        changed = self.u.clone()
        changed[:, 20:, 3:5] = 0.91
        b, _ = self.m.contexts(changed)
        torch.testing.assert_close(a.inverse_features[:, :21], b.inverse_features[:, :21])
        torch.testing.assert_close(a.c[:, :21], b.c[:, :21])
        # Previous applied current is intentionally observable, unlike current I[t].
        self.assertGreater(float((a.inverse_features[:, 21:]-b.inverse_features[:, 21:]).abs().max()), 0)

    def test_sequence_step_and_chunk_agree(self):
        expected = self.m(self.u)
        state = self.m.initial_state(2)
        outputs = []
        for k in range(self.u.shape[1]):
            ctx, h = self.m.prepare(self.u[:, k], state)
            outputs.append(self.m.force_at(ctx, self.u[:, k:k+1, 3]))
            state = self.m.commit(h, state, self.u[:, k, 3])
        torch.testing.assert_close(torch.cat(outputs, 1), expected)
        f1, state = self.m(self.u[:, :17], return_state=True)
        f2 = self.m(self.u[:, 17:], state)
        torch.testing.assert_close(torch.cat((f1, f2), 1), expected)

    def test_lag_matches_scalar_and_legacy_does_not(self):
        lag = CurrentLag(11).double()
        x = torch.ones(2, 160, dtype=torch.double)
        y = torch.zeros(2, dtype=torch.double)
        seq = []
        for k in range(160):
            y = lag.step(x[:, k], y)
            seq.append(y)
        torch.testing.assert_close(lag(x, torch.zeros(2, dtype=torch.double)), torch.stack(seq, 1))
        legacy = LegacyLag(11).double()
        self.assertGreater(float((legacy(x)-torch.stack(seq, 1)).abs().max()), 0.5)

    def test_same_context_inverse_and_query_nonmutation(self):
        ctx, _ = self.m.contexts(self.u)
        before = ctx.previous_effective_current.clone()
        desired = self.m.force_at(ctx, self.u[..., 3])
        current, info = self.m.inverse_current(ctx, desired)
        torch.testing.assert_close(self.m.force_at(ctx, current), desired, atol=1e-8, rtol=1e-7)
        torch.testing.assert_close(before, ctx.previous_effective_current)
        self.assertTrue(bool(info['reachable'].all()))
        _, outside = self.m.inverse_current(ctx, desired+100)
        self.assertFalse(bool(outside['reachable'].any()))

    def test_force_map_monotone_for_fixed_context(self):
        ctx, _ = self.m.contexts(self.u)
        f = [self.m.force_at(ctx, torch.full_like(ctx.v, k/20)) for k in range(21)]
        difference = torch.stack(f[1:])-torch.stack(f[:-1])
        self.assertLessEqual(float((difference*ctx.v).max()), 1e-10)

    def test_sensitivity_matches_finite_difference_and_empty_mask(self):
        ctx, _ = self.m.contexts(self.u)
        i = torch.full_like(ctx.v, 0.5, requires_grad=True)
        derivative, = torch.autograd.grad(self.m.force_at(ctx, i).sum(), i)
        fd = (self.m.force_at(ctx, i+1e-5)-self.m.force_at(ctx, i-1e-5))/2e-5
        torch.testing.assert_close(derivative, fd, atol=1e-7, rtol=1e-5)
        u = self.u.clone(); u[..., 1] = 0
        force = self.m(u).detach()
        labels = reference_labels(self.m, u, force, torch.ones_like(force, dtype=torch.bool),
                                  f_scale=1000, sensitivity_scale=100)
        p = self.m.inverse_predict(u, force)
        loss, stats = inverse_loss(p, u, labels, 'teacher')
        self.assertEqual(float(loss), 0)
        self.assertEqual(stats['eligible_fraction'], 0)

    def test_stages_and_budget(self):
        self.assertEqual(sum(p.numel() for p in self.m.parameters()), 21041)
        set_stage(self.m, 2)
        self.assertTrue(all(p.requires_grad == n.startswith('inv_head.') for n, p in self.m.named_parameters()))
        self.assertTrue(joint_gate(dict(pooled_nrmse_pct=17, hf_nrmse_pct=18),
                                   dict(pooled_nrmse_pct=16, hf_nrmse_pct=17)))
        self.assertFalse(joint_gate(dict(pooled_nrmse_pct=17.01, hf_nrmse_pct=18),
                                    dict(pooled_nrmse_pct=16, hf_nrmse_pct=17)))

    def test_rollout_does_not_use_measured_current_after_burn(self):
        force=self.m(self.u).detach()
        i1,f1=rollout(self.m,self.u,force,burn=10,hold=10)
        changed=self.u.clone(); changed[:,10:,3]=1-changed[:,10:,3]
        i2,f2=rollout(self.m,changed,force,burn=10,hold=10)
        torch.testing.assert_close(i1,i2); torch.testing.assert_close(f1,f2)
        torch.testing.assert_close(i1[:,10:20],i1[:,10:11].expand(-1,10))

    def test_synthetic_evaluation_only_no_weight_updates(self):
        model=NLCSNN_v19().eval()
        values=torch.zeros(2010,7); values[:,0]=250; values[:,1]=50
        values[:,3]=0.8; values[:,5]=30; values[:,6]=20
        item=dict(name='SH_synthetic_6.62Hz',values=values)
        norm=dict(x_ref=250,x_scale=10,f_scale=1000)
        before={k:v.clone() for k,v in model.state_dict().items()}
        report=evaluate(model,[item],norm,model,100)
        self.assertEqual(report['rows'][0]['samples'],10)
        for k,v in model.state_dict().items(): torch.testing.assert_close(v,before[k])

    def test_explicit_warm_mapping(self):
        from mr_nlcsnn.model_v14 import NLCSNN_v14
        old=NLCSNN_v14(gru_hidden=16,mlp_hidden=48,g_hidden=24,sigmas=(.5,1,2,4,8))
        mapping=warm_transfer(self.m,dict(state_dict=old.state_dict()))
        self.assertEqual(mapping['c_head.2.weight'],'c_head.3.weight')
        self.assertFalse(any(k.startswith(('lag.','G_c.','c_fast.','inv_head.')) for k in mapping))

    def test_inverse_gradients_only_reach_unfrozen_head(self):
        import copy
        reference=copy.deepcopy(self.m)
        for parameter in reference.parameters(): parameter.requires_grad_(False)
        parameters=set_stage(self.m,2)
        force=reference(self.u).detach()
        labels=reference_labels(reference,self.u,force,torch.ones_like(force,dtype=torch.bool),
                                f_scale=1000,sensitivity_scale=1e-6)
        prediction=self.m.inverse_predict(self.u,force)
        loss,_=inverse_loss(prediction,self.u,labels,'teacher')
        gradients=torch.autograd.grad(loss,parameters)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(bool(g.abs().sum()>0) for g in gradients))
        self.assertTrue(all(p.grad is None for p in reference.parameters()))


if __name__ == '__main__':
    unittest.main()
