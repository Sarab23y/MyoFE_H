#!/usr/bin/env python3
"""Isolated source-method experiment. No FE solve or historical population replay.

Python >=3.7 with NumPy, SciPy and pandas. Outputs are exclusively diagnostic.
All non-scalar initial cellular states are explicitly synthetic fixtures.
"""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys
import textwrap
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'python_codes/LV_simulation'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def methods():
    # Import the actual cellular package, without importing LV_simulation.py
    # or DOLFIN. Its package initializers are empty in this repository.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    half = importlib.import_module('python_codes.LV_simulation.half_sarcomere.half_sarcomere')
    source = (BASE/'mesh/mesh.py').read_text()
    start = source.index('    def return_cb_stress(')
    end = source.index('\n    def ', start+5)
    scope = {'np': np}
    exec(compile(textwrap.dedent(source[start:end]), '<actual MeshClass.return_cb_stress>', 'exec'), scope)
    return half.half_sarcomere, scope['return_cb_stress']


def population(myof):
    y = myof.y.copy()
    bins = y[2:2+myof.no_of_x_bins]
    return dict(y=y.tolist(), minimum=float(y.min()), head_mass=float(y[:-2].sum()),
                thin_mass=float(y[-2:].sum()), bound=float(bins.sum()),
                on_fraction=float(y[-1]), off_fraction=float(y[-2]),
                bound_exceeds_on=bool(bins.sum() > y[-1]+1e-12),
                detached=float(y[1]), srx=float(y[0]), positive=bool(np.all(y >= 0)))


def numeric_fe_stress(method, hs, delta):
    # Actual method, numeric stand-ins for scalar UFL coefficients. No FEniCS
    # or claim of equilibrium, actual deformation, or actual historical y.
    class Fixture:
        pass
    fixture = Fixture()
    fixture.hs = hs
    fixture.y_split = hs.myof.y.copy()
    fixture.model = {'functions': hs.myof.data}
    return float(method(fixture, delta))


def experiment(hs_struct, event, fixture):
    half_class, fe_method = methods()
    if hs_struct['myofilaments']['implementation']['kinetic_scheme'][0] != '3_state_with_SRX':
        raise ValueError('This explicit fixture supports only 3_state_with_SRX; do not infer other layouts')
    hs = half_class(hs_struct)
    hs.data['hs_length'] = event['hsl_before_nm']
    hs.myof.y[:] = 0
    hs.myof.y[0:2] = [.4, .55]
    hs.myof.y[2:2+hs.myof.no_of_x_bins] = .05/hs.myof.no_of_x_bins
    hs.myof.y[-2:] = [.9, .1]
    if len(hs.memb.y) != 2:
        raise ValueError('Fixture requires simple two-compartment membrane')
    hs.memb.y[:] = [1e-7, hs.memb.data['Ca_content']-1e-7]
    hs.memb.update_data()
    stored = fixture['stored_feedback_Pa']
    hs.myof.cb_stress, hs.myof.pas_stress = stored
    hs.myof.total_stress = sum(stored)
    result = dict(dof=event['dof'], evidence_class='fixture-demonstrated',
                  historical_population_state_available=False, event_scalar=event,
                  interpretation='MyoSim catch-up call using observed prior FE increment, NOT replay of event FE solve',
                  before=population(hs.myof), x_bins_nm=hs.myof.x.tolist())
    result['numeric_reset_before_catchup'] = dict(
        pre_reset_Pa=numeric_fe_stress(fe_method, hs, event['delta_hsl_nm']),
        post_reset_Pa=numeric_fe_stress(fe_method, hs, 0),
        evidence_class='fixture-demonstrated',
        warning='Fixed synthetic y and scalar coefficients; no displacement/equilibrium was evaluated')
    original_move = hs.myof.move_cb_distributions
    original_kinetics = hs.myof.evolve_kinetics
    calls = []

    def move(delta):
        original_move(delta)
        calls.append(dict(stage='after_advection', delta_hsl_nm=delta, population=population(hs.myof)))

    def kinetics(dt, calcium):
        entry = dict(stage='kinetics', calcium_consumed=float(calcium),
                     new_membrane_y_cytosol=float(hs.memb.y[0]),
                     stored_cb_Pa=float(hs.myof.cb_stress), stored_passive_Pa=float(hs.myof.pas_stress),
                     stored_total_Pa=float(hs.myof.total_stress), hsl_nm=float(hs.data['hs_length']))
        original_kinetics(dt, calcium)
        entry.update(overlap=float(hs.myof.n_overlap), after=population(hs.myof))
        calls.append(entry)

    hs.myof.move_cb_distributions = move
    hs.myof.evolve_kinetics = kinetics
    hs.update_simulation(fixture['time_step_s'], event['delta_hsl_nm'], fixture['activation'],
                         *fixture['incoming_feedback_Pa'])
    result['after_call_before_update_data'] = dict(
        population=population(hs.myof), hsl_nm=float(hs.data['hs_length']),
        calcium_dictionary=float(hs.memb.data['Ca_cytosol']), calcium_y=float(hs.memb.y[0]),
        assigned_total_stress_Pa=float(hs.myof.total_stress))
    hs.update_data()
    result['after_update_data'] = dict(calcium_dictionary=float(hs.memb.data['Ca_cytosol']))
    result['calls'] = calls
    result['numeric_reset_after_catchup'] = dict(
        pre_reset_Pa=numeric_fe_stress(fe_method, hs, event['delta_hsl_nm']),
        post_reset_Pa=numeric_fe_stress(fe_method, hs, 0))
    hs.data['hs_length'] = fixture['extreme_overlap_probe_hsl_nm']
    result['extreme_overlap_probe'] = dict(hsl_nm=hs.data['hs_length'],
                                          overlap=float(hs.myof.return_n_overlap()),
                                          evidence_class='fixture-demonstrated',
                                          parameters='supplied configuration, not authenticated historical parameters')
    return result


def run(input_json, fixtures, output):
    input_json, fixtures, output = map(Path, (input_json, fixtures, output))
    if output.exists():
        raise ValueError('Refusing to overwrite output: '+str(output))
    # Require the caller to keep outputs outside all production and fixture inputs.
    for path in (input_json, fixtures):
        if path.resolve() == output.resolve() or path.resolve() in output.resolve().parents:
            raise ValueError('Output overlaps an input')
    if any(part in ('sim_output', 'mesh_output') for part in output.resolve().parts):
        raise ValueError('Diagnostic output must be outside production result directories')
    instruction = json.loads(input_json.read_text())
    fixture = json.loads(fixtures.read_text())
    if fixture.get('historical_population_state_available') is not False:
        raise ValueError('Only explicitly synthetic population fixtures are supported')
    results = [experiment(instruction['model']['half_sarcomere'], event, fixture)
               for event in fixture['events']]
    probe = fixture['positivity_probe']
    implementation = {k:v[0] for k,v in instruction['model']['half_sarcomere']['myofilaments']['implementation'].items()}
    state = types.SimpleNamespace(implementation=implementation,
        x=np.asarray(probe['x_bins_nm'], float), no_of_x_bins=len(probe['bound_bins']),
        y=np.r_[.1, 0., probe['bound_bins'], .05, .95])
    before_probe = population(state)
    actual_move = importlib.import_module('python_codes.LV_simulation.half_sarcomere.myofilaments.move')
    actual_move.move_cb_distributions(state, probe['delta_hsl_nm'])
    positivity = dict(evidence_class='fixture-demonstrated', before=before_probe, after=population(state),
                      definition=probe['definition'])
    output.mkdir(parents=True)
    files = ['half_sarcomere/half_sarcomere.py', 'half_sarcomere/membranes/membranes.py',
             'half_sarcomere/myofilaments/myofilaments.py', 'half_sarcomere/myofilaments/move.py',
             'half_sarcomere/myofilaments/kinetics.py', 'half_sarcomere/myofilaments/forces.py', 'mesh/mesh.py']
    report = dict(evidence_class='fixture-demonstrated', source_methods_executed=True,
                  historical_replay=False, finite_element_solve_performed=False,
                  source_sha256={f:digest(BASE/f) for f in files}, input_sha256=digest(input_json),
                  fixture_sha256=digest(fixtures), configuration_provenance=str(input_json),
                  python=sys.version, results=results,
                  separate_positivity_probe=positivity,
                  unresolved=['historical full populations and consumed calcium/stress',
                              'historical hsl_old, actual submitted parameters and source identity',
                              'exact CG2 displacement, F, J, solver residual and material mapping'])
    (output/'coupling_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (output/'input_fixture.json').write_text(json.dumps(fixture, indent=2))
    rows = []
    from diagnose_midwall_sarcomere_stdlib import svg_file
    for r in results:
        after_move = r['calls'][0]['population']
        after_kinetics = r['calls'][1]['after']
        for stage, state in [('initial_synthetic', r['before']), ('after_advection', after_move), ('after_kinetics', after_kinetics)]:
            rows.append(dict(dof=r['dof'], stage=stage, evidence_class='fixture-demonstrated',
                             minimum=state['minimum'], head_mass=state['head_mass'], thin_mass=state['thin_mass'],
                             detached=state['detached'], bound=state['bound'], srx=state['srx']))
        series=[]
        for label, state, color in [('Initial synthetic', r['before'], '#2463a6'),
                                    ('After advection', after_move, '#d45132'),
                                    ('After kinetics', after_kinetics, '#27824c')]:
            series.append((label, list(zip(r['x_bins_nm'], state['y'][2:2+len(r['x_bins_nm'])])), color))
        svg_file(output/('fixture_%d.svg'%r['dof']), [('Bound-bin fraction', series)],
                 'DOF %d increment: SYNTHETIC populations, actual source methods'%r['dof'], xlabel='Crossbridge extension (nm)')
    with (output/'population_stages.csv').open('w', newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-json', required=True, help='configuration parameters; not proof of historical producer')
    parser.add_argument('--fixtures', default=str(ROOT/'validation/fixtures/hsl_coupling_events.json'))
    parser.add_argument('--output', required=True, help='fresh diagnostic directory outside simulation results')
    args = parser.parse_args()
    run(args.input_json, args.fixtures, args.output)


if __name__ == '__main__':
    main()
