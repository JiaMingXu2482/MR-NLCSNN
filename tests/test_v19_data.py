import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from AA01DataProcess.prepare_v19_data import align_raw, assign_splits
from AA01DataProcess.process_raw_card_archive import process_file, CANDIDATES
from AA02Train.audit_v18_metrics import summarize


class DataTests(unittest.TestCase):
    def test_raw_mapping_both_shift_directions(self):
        raw = pd.DataFrame(np.arange(3000*9).reshape(3000, 9).astype(float))
        original = raw.copy(deep=True)
        for f, i in ((42, 42), (40, 44), (-42, 42), (0, 0)):
            aligned, m = align_raw(raw, f, i)
            lo, hi = m['common_start'], m['common_stop_exclusive']
            np.testing.assert_array_equal(aligned.iloc[:, 3].to_numpy(), raw.iloc[lo:hi, 3].to_numpy())
            np.testing.assert_array_equal(aligned.iloc[:, 5].to_numpy(), raw.iloc[lo+f:hi+f, 5].to_numpy())
            np.testing.assert_array_equal(aligned.iloc[:, 6].to_numpy(), raw.iloc[lo+i:hi+i, 6].to_numpy())
        pd.testing.assert_frame_equal(raw, original)

    def test_group_closure_and_duplicate(self):
        spec = dict(validation_stems=['v_r1'], final_test_stems=['t_r1'], exclude_stems=[])
        rows = [dict(short_name=n+'.csv', sha256=h) for n,h in
                [('v_r1','a'), ('v_r2','b'), ('t_r1','c'), ('t_r2','d'),
                 ('alias','c'), ('train','e')]]
        result = assign_splits(rows, spec)
        self.assertEqual([r['split'] for r in result],
                         ['validation','validation','final_test','final_test','duplicate','train'])
        rows[0]['sha256'] = 'c'
        with self.assertRaises(ValueError):
            assign_splits(rows, spec)

    def test_alignment_affects_step_detection_before_crop(self):
        raw = pd.DataFrame(np.zeros((5000,9)))
        raw.iloc[:,3] = 4.0
        raw.iloc[1500:2500,6] = 1000.0
        raw.iloc[:,8] = 26.0
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'raw.csv'; path.touch()
            row = dict(output_path=str(path), category='5_ST', condition_key='STEP', short_name='test.csv')
            _, before = process_file(row, CANDIDATES['force_wide'], raw_override=raw)
            aligned, _ = align_raw(raw, 42, 42)
            frame, after = process_file(row, CANDIDATES['force_wide'], raw_override=aligned)
            self.assertEqual(before['active_start_raw_idx']-after['active_start_raw_idx'], 42)
            # Detection moves, while detected interval length need NOT become old length - 42.
            self.assertEqual(before['samples'], len(frame))

    def test_metrics_audit_uses_finite_evaluations(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'metrics.csv'
            p.write_text('total_epoch,pooled_nrmse_pct,inv_mae_A\n1,nan,nan\n2,36.24,0.464\n3,40,0.5\n')
            result = summarize(p)
            self.assertEqual(result['best_row']['total_epoch'], '2')
            self.assertEqual(result['evaluated_rows'], 2)


if __name__ == '__main__':
    unittest.main()
