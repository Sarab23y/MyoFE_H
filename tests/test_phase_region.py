import csv
import json
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from validation import analyze_phase_region as ar
from validation import export_verified_quadrature_geometry as eq


def cardiac_data(beats=7):
    # 10 samples/beat: MVC=.1, AO=.3, AC=.5, MO=.7.
    times = np.arange(beats*10+1, dtype=float)/10
    phase = np.arange(len(times)) % 10
    return {'time': times, 'pressure_ventricle': np.where((phase >= 3) & (phase < 5), 100., 10.),
            'volume_ventricle': np.where(phase < 3, 100., np.where(phase < 5, 100.-20*(phase-2), 60.)),
            'flow_ventricle_to_aorta': ((phase >= 3) & (phase < 5)).astype(float),
            'flow_veins_to_ventricle': ((phase >= 7) | (phase < 1)).astype(float),
            'new_beat': (phase == 1).astype(float)}


class PhaseRegionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ids = [2833, 2497, 4844, 999]
        self.xyz = [[0, 0, 0], [1, 0, .4], [0, 1, .6], [1, 1, 1]]
        self.wall = [.1, .4, .6, .9]
        np.save(self.root/'quadrature_dof.npy', self.xyz)
        np.save(self.root/'quadrature_global_dof.npy', self.ids)
        np.save(self.root/'norm_dist_endo.npy', self.wall)
        self.report = {'passed': True, 'saved_mapping_verified': True, 'weights_verified': True,
                       'reference_volume': 6.,
                       'source_sha256': {name: ar.digest(self.root/name) for name in
                          ('quadrature_dof.npy', 'quadrature_global_dof.npy', 'norm_dist_endo.npy')},
                       'points': [{'global_dof': gid, 'coordinate': xyz, 'wall_depth': wall,
                                   'reference_weight': w} for gid, xyz, wall, w in
                                  zip(self.ids, self.xyz, self.wall, [1., 1., 3., 1.])]}
        for point in self.report['points']:
            center = np.asarray(point['coordinate'])
            point['cell_reference_vertices'] = (center + np.asarray([[-1,-1,-1], [3,-1,-1], [-1,3,-1], [-1,-1,3]])).tolist()
            point['quadrature_point_in_cell'] = 0
        self.geom = self.root/'geometry.json'
        self.input = self.root/'instruction.json'
        self.input.write_text('{}')
        self.save_report()

    def tearDown(self):
        self.tmp.cleanup()

    def save_report(self):
        self.geom.write_text(json.dumps(self.report))

    def make_suite(self):
        """Synthetic reports only; these fixtures never claim a DOLFIN run."""
        self.mapping = self.root/'mapping'
        self.mapping.mkdir()
        for ranks in (1, 32):
            for kind in ('mpi_comm', 'ecm', 'mixed', 'geometry'):
                report = dict(self.report) if kind == 'geometry' else {'passed': True}
                report.update(mpi_size=ranks, input_sha256=ar.digest(self.input), mesh_sha256='same',
                              mesh_partitioner='SCOTCH', vector_point_major_ordering_verified_all_ranks=True,
                              owned_index_and_rank_scatter_verified=True)
                if kind == 'mixed':
                    report['ranks'] = [{'rank': r, 'passed': True} for r in range(ranks)]
                (self.mapping/('%s_%d.json' % (kind, ranks))).write_text(json.dumps(report))
        for name in ('ecm_comparison.json', 'geometry_comparison.json'):
            (self.mapping/name).write_text(json.dumps({'passed': True}))
        self.image = self.root/'synthetic-image'
        self.image.write_text('synthetic test fixture, not a container')
        self.repo = Path(ar.__file__).resolve().parents[1]
        with mock.patch.object(eq.subprocess, 'check_output', return_value=b'synthetic-test-commit'):
            eq.seal_suite(str(self.mapping), str(self.input), str(self.image), str(self.repo))

    def test_suite_rejects_missing_owned_scatter_evidence(self):
        self.make_suite()
        path = self.mapping/'geometry_32.json'
        report = json.loads(path.read_text())
        del report['owned_index_and_rank_scatter_verified']
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'owned indices/rank scatter'):
            eq.check_reports(str(self.mapping), str(self.input))

    def test_suite_rejects_scalar_using_a_different_mesh_or_instruction(self):
        self.make_suite()
        path = self.mapping/'ecm_1.json'
        original = json.loads(path.read_text())
        for key in ('mesh_sha256', 'input_sha256'):
            report = dict(original); report[key] = 'different'
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                eq.check_reports(str(self.mapping), str(self.input))

    def field(self, name, times, order=None):
        order = order or self.ids
        with (self.root/(name+'_data.csv')).open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow([''] + order + ['time'])
            for t in times:
                value = {2833: .2, 2497: 2.+t, 4844: 6.+t, 999: 1.}
                writer.writerow([0]+[value[i] for i in order]+[t])

    def test_known_valve_boundaries_and_five_cycles(self):
        g = cardiac_data()
        cycles, rejected = ar.phase_cycles(g)
        self.assertFalse(rejected)
        np.testing.assert_allclose(cycles[0]['times'], [.1, .3, .5, .7, 1.1])
        np.testing.assert_allclose(list(cycles[0]['phase_durations'].values()), [.2, .2, .2, .4])
        chosen = ar.choose_cycles(cycles, set(range(len(g['time']))), 5, 0, 7)
        self.assertEqual(len(chosen), 5)
        self.assertEqual([c['candidate'] for c in chosen], list(range(5)))

    def test_variable_phase_timing_not_fixed_offsets(self):
        g = cardiac_data()
        g['flow_ventricle_to_aorta'][13] = 0
        cycles, _ = ar.phase_cycles(g)
        self.assertAlmostEqual(cycles[1]['times'][1]-cycles[1]['times'][0], .3)

    def test_default_window_uses_latest_available_cycles_not_40s(self):
        g = cardiac_data()
        cycles, _ = ar.phase_cycles(g)
        chosen = ar.choose_cycles(cycles, set(range(len(g['time']))), 5, None, float('inf'))
        self.assertEqual([c['candidate'] for c in chosen], [1, 2, 3, 4, 5])
        self.assertLess(chosen[-1]['times'][-1], 40)

    def test_reverse_flow_and_unquantified_stagger_are_reported(self):
        g = cardiac_data()
        g['flow_veins_to_ventricle'][2] = -1
        cycles, _ = ar.phase_cycles(g)
        self.assertEqual(cycles[0]['signal_consistency']['reverse_flow_samples']['flow_veins_to_ventricle'], 1)
        self.assertFalse(cycles[0]['exact_valve_events_identified'])
        self.assertIsNone(cycles[0]['coupling_stagger_uncertainty_s'])

    def test_venous_inlet_pressure_is_not_relabeled_as_atrial(self):
        g = cardiac_data()
        inlet = g.pop('flow_veins_to_ventricle')
        g['pressure_veins'] = np.where(inlet > 0, 200., 0.)
        cycles, _ = ar.phase_cycles(g)
        self.assertIn('venous', cycles[0]['inlet_method'])
        self.assertNotIn('atrial', cycles[0]['inlet_method'])

    def test_missing_valves_uses_labeled_pressure_estimates(self):
        g = cardiac_data()
        outgoing = g.pop('flow_ventricle_to_aorta')
        incoming = g.pop('flow_veins_to_ventricle')
        g['pressure_aorta'] = np.where(outgoing > 0, 1., 200.)
        g['pressure_atrium'] = np.where(incoming > 0, 200., 0.)
        cycles, _ = ar.phase_cycles(g)
        self.assertIn('estimate', cycles[0]['aortic_method'])
        self.assertIn('estimate', cycles[0]['inlet_method'])

    def test_chatter_rejected_without_fabricating_cycle(self):
        g = cardiac_data()
        g['flow_ventricle_to_aorta'][4] = 0
        g['flow_ventricle_to_aorta'][6] = 1
        cycles, rejected = ar.phase_cycles(g)
        self.assertTrue(rejected)
        self.assertNotIn(0, [c['candidate'] for c in cycles])

    def test_global_gap_rejected(self):
        g = cardiac_data()
        g['time'][4:] += .1
        _, rejected = ar.phase_cycles(g)
        self.assertIn('global time gap', [r['reason'] for r in rejected])

    def test_global_duplicate_time_and_missing_pressure_rejected(self):
        g = cardiac_data()
        g['time'][3] = g['time'][2]
        ar.write_csv(self.root/'data.csv', ({k: v[i] for k, v in g.items()} for i in range(len(g['time']))))
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            ar.read_global(self.root/'data.csv')
        ar.write_csv(self.root/'data.csv', [{'time': 0, 'volume_ventricle': 10}])
        with self.assertRaisesRegex(ValueError, 'pressure_ventricle'):
            ar.read_global(self.root/'data.csv')

    def test_incomplete_data_and_nonconsecutive_cycles_rejected(self):
        g = cardiac_data(6)
        cycles, _ = ar.phase_cycles(g)
        with self.assertRaises(ValueError):
            ar.choose_cycles(cycles, set(range(45)), 5, 0, 7)
        with self.assertRaises(ValueError):
            ar.choose_cycles(cycles[:2]+cycles[3:], set(range(100)), 5, 0, 7)

    def test_complete_region_and_weights(self):
        result = ar.geometry(self.geom, self.root, self.ids[:3])
        self.assertEqual(result[-2].tolist(), [1, 2])
        s = ar.stats([2, 6], result[-1][result[-2]])
        self.assertEqual(s['point_mean'], 4.)
        self.assertEqual(s['volume_weighted_mean'], 5.)
        self.assertAlmostEqual(s['volume_weighted_std'], np.sqrt(3))
        self.assertEqual(s['median'], 4.)

    def test_no_weights_never_claims_volume_mean(self):
        self.report['weights_verified'] = False
        self.save_report()
        self.assertIsNone(ar.geometry(self.geom, self.root, self.ids[:3])[-1])
        self.assertNotIn('volume_weighted_mean', ar.stats([2, 6]))

    def test_geometry_identity_hash_duplicates_and_bad_weights(self):
        self.report['points'][1]['coordinate'] = self.report['points'][0]['coordinate']
        self.save_report()
        with self.assertRaises(ValueError):
            ar.geometry(self.geom, self.root, self.ids[:3])
        self.report['points'][1]['coordinate'] = self.xyz[1]
        self.report['points'][1]['reference_weight'] = -1
        self.save_report()
        with self.assertRaises(ValueError):
            ar.geometry(self.geom, self.root, self.ids[:3])
        np.save(self.root/'quadrature_global_dof.npy', [1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            ar.geometry(self.geom, self.root, self.ids[:3])

    def test_reordered_columns_preserve_point_identities(self):
        self.field('hs_length', [0, .1, .2], list(reversed(self.ids)))
        rows = list(ar.spatial_rows(self.root/'hs_length_data.csv', self.ids))
        self.assertEqual(float(rows[0][1][rows[0][2][0]]), .2)
        self.assertEqual(float(rows[0][1][rows[0][2][2]]), 6.)
        with self.assertRaises(ValueError):
            list(ar.spatial_rows(self.root/'hs_length_data.csv', [1, 2, 3, 4]))

    def test_time_alignment_gap_shift_duplicate_and_truncation(self):
        p = self.root/'hs_length_data.csv'
        self.field('hs_length', [0, .2])
        self.assertEqual(ar.aligned_times(p, self.ids, np.array([0, .1, .2])), {0, 2})
        self.field('hs_length', [0, .1001])
        with self.assertRaisesRegex(ValueError, 'off the global grid'):
            ar.aligned_times(p, self.ids, np.array([0, .1, .2]))
        self.field('hs_length', [0, 0])
        with self.assertRaises(ValueError):
            list(ar.spatial_rows(p, self.ids))
        p.write_text(',2833,2497,4844,999,time\n0,1,2\n')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            list(ar.spatial_rows(p, self.ids))

    def test_end_to_end_reports_hotspots_and_cycle_comparison(self):
        g = cardiac_data()
        ar.write_csv(self.root/'data.csv', ({k: v[i] for k, v in g.items()} for i in range(len(g['time']))))
        for field in ar.FIELDS:
            self.field(field, g['time'], list(reversed(self.ids)))
        self.make_suite()
        args = ar.parser().parse_args(['--data', str(self.root), '--mapping', str(self.mapping), '--input-json', str(self.input),
                                      '--output', str(self.root/'out'), '--axis', '2', '--start', '0',
                                      '--hotspots', '2833', '2497', '4844'])
        ar.analyze(args)
        report = json.loads((self.root/'out/report.json').read_text())
        self.assertTrue(report['complete'])
        self.assertEqual(report['region_points'], 2)
        self.assertEqual(report['mean_kind'], 'reference_volume_weighted')
        variability = json.loads((self.root/'out/cycle_variability.json').read_text())
        np.testing.assert_allclose(variability[0]['consecutive_differences'], [1, 1, 1, 1])
        with (self.root/'out/hs_length_region.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertAlmostEqual(float(rows[0]['volume_weighted_mean']), 5.1)
        self.assertEqual(len(rows), 50)
        with (self.root/'out/fiber_stretch_raw_region.csv').open() as handle:
            stretch = next(csv.DictReader(handle))
        self.assertEqual(float(stretch['region_point_fraction_outside_screen']), 1.)
        self.assertEqual(float(stretch['whole_mesh_point_fraction_outside_screen']), .75)
        self.assertTrue((self.root/'out/regional_overview.svg').exists())
        for svg in (self.root/'out').glob('*.svg'):
            ET.parse(svg)
        with (self.root/'out/point_cycle_variability.csv').open() as handle:
            point_rows = list(csv.DictReader(handle))
        point = next(r for r in point_rows if r['field'] == 'hs_length' and r['phase'] == 'ejection'
                     and r['global_dof'] == '2497' and r['metric'] == 'minimum')
        self.assertAlmostEqual(float(point['cycle_1_to_2_difference']), 1.)
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            ar.analyze(args)

    def test_sealed_suite_requires_all_reports_input_and_hashes(self):
        self.make_suite()
        self.assertTrue(eq.require_suite(str(self.mapping), str(self.input), str(self.repo))['complete'])
        altered = self.mapping/'mixed_32.json'
        original = altered.read_text()
        altered.write_text(original+'\n')
        with self.assertRaisesRegex(ValueError, 'stale/modified'):
            eq.require_suite(str(self.mapping), str(self.input), str(self.repo))
        altered.write_text(original)
        self.input.write_text('{"different_input":true}')
        with self.assertRaisesRegex(ValueError, 'input mismatch'):
            eq.require_suite(str(self.mapping), str(self.input), str(self.repo))

    def test_suite_rejects_missing_reports_wrong_ranks_and_unverified_weights(self):
        self.make_suite()
        path = self.mapping/'geometry_32.json'
        report = json.loads(path.read_text())
        report['weights_verified'] = False
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'weights/vector'):
            eq.check_reports(str(self.mapping), str(self.input))
        report['weights_verified'] = True
        report['mpi_size'] = 1
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'wrong MPI size'):
            eq.check_reports(str(self.mapping), str(self.input))
        path.unlink()
        with self.assertRaises(OSError):
            eq.check_reports(str(self.mapping), str(self.input))

    def test_geometry_checks_volume_sum_and_cell_coordinates(self):
        self.report['reference_volume'] = 100
        self.save_report()
        with self.assertRaisesRegex(ValueError, 'reference volume'):
            ar.geometry(self.geom, self.root, self.ids[:3])
        self.report['reference_volume'] = 6
        self.report['points'][0]['cell_reference_vertices'] = [[10,10,10], [11,10,10], [10,11,10], [10,10,11]]
        self.save_report()
        with self.assertRaisesRegex(ValueError, 'outside'):
            ar.geometry(self.geom, self.root, self.ids[:3])

    def test_cross_mpi_geometry_compare_uses_coordinates_not_ids(self):
        first = dict(self.report, mesh_sha256='same', input_sha256='same')
        second = json.loads(json.dumps(first))
        second['points'].reverse()
        for i, point in enumerate(second['points']):
            point['global_dof'] = i
        a, b, result = [self.root/name for name in ('a.json', 'b.json', 'comparison.json')]
        a.write_text(json.dumps(first)); b.write_text(json.dumps(second))
        self.assertEqual(eq.compare(a, b, result), 0)
        second['points'][0]['reference_weight'] += 1
        b.write_text(json.dumps(second))
        result.unlink()
        self.assertEqual(eq.compare(a, b, result), 1)


if __name__ == '__main__':
    unittest.main()
