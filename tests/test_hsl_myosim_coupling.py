import json
from pathlib import Path
import tempfile
import unittest

from validation.validate_hsl_myosim_coupling import run, ROOT

CONFIG = ROOT/'demos/ECM/sim_input/base_instruction_Br_ECM_60s_healthy_all_outputs_stride1.json'
FIXTURES = ROOT/'validation/fixtures/hsl_coupling_events.json'


class HSLCouplingTests(unittest.TestCase):
    def test_actual_source_functions_all_four_increments_and_stage_contracts(self):
        before = CONFIG.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'diagnostics'
            report = run(CONFIG, FIXTURES, path)
            self.assertFalse(report['historical_replay'])
            self.assertFalse(report['finite_element_solve_performed'])
            self.assertEqual([r['dof'] for r in report['results']], [2833,2497,4844,1580])
            for result in report['results']:
                self.assertEqual(result['evidence_class'], 'fixture-demonstrated')
                self.assertEqual(result['calls'][0]['population']['bound'], 0.)
                self.assertAlmostEqual(result['calls'][0]['population']['head_mass'], 1.)
                self.assertTrue(result['calls'][0]['population']['positive'])
                entry = result['calls'][1]
                self.assertEqual(entry['calcium_consumed'], 1e-7)
                self.assertNotEqual(entry['new_membrane_y_cytosol'], entry['calcium_consumed'])
                self.assertEqual(entry['stored_total_Pa'], 120.)
                self.assertEqual(result['after_call_before_update_data']['assigned_total_stress_Pa'], 340.)
                self.assertEqual(result['after_update_data']['calcium_dictionary'], entry['new_membrane_y_cytosol'])
                self.assertNotEqual(result['numeric_reset_before_catchup']['pre_reset_Pa'],
                                    result['numeric_reset_before_catchup']['post_reset_Pa'])
                self.assertLess(result['extreme_overlap_probe']['overlap'], 0.)
                self.assertAlmostEqual(result['after_call_before_update_data']['hsl_nm'],
                                       result['event_scalar']['hsl_before_nm']+result['event_scalar']['delta_hsl_nm'])
            probe = report['separate_positivity_probe']
            self.assertTrue(probe['before']['positive'])
            self.assertFalse(probe['after']['positive'])
            self.assertAlmostEqual(probe['before']['head_mass'], probe['after']['head_mass'])
            original = (path/'coupling_report.json').read_bytes()
            with self.assertRaises(ValueError):
                run(CONFIG, FIXTURES, path)
            self.assertEqual(original, (path/'coupling_report.json').read_bytes())
            self.assertEqual(len(list(path.glob('fixture_*.svg'))), 4)
        self.assertEqual(CONFIG.read_bytes(), before)

    def test_rejects_claimed_historical_population_and_production_output(self):
        with tempfile.TemporaryDirectory() as directory:
            fixtures = json.loads(FIXTURES.read_text())
            fixtures['historical_population_state_available'] = True
            path = Path(directory)/'input.json';path.write_text(json.dumps(fixtures))
            with self.assertRaises(ValueError):
                run(CONFIG, path, Path(directory)/'diagnostics')
            with self.assertRaises(ValueError):
                run(CONFIG, FIXTURES, Path(directory)/'sim_output/diagnostics')


if __name__ == '__main__':
    unittest.main()
