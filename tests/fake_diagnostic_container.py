"""Test-only Singularity boundary double. It never claims FE/MPI execution."""
import hashlib
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
if '--version' in args or '-c' in args:
    print('Synthetic container boundary for Bash workflow tests')
    raise SystemExit(0)
script = next(Path(arg).name for arg in args if arg.endswith('.py'))
ranks = int(args[args.index('-np')+1]) if '-np' in args else 1
with open(os.environ['DIAG_TEST_TRACE'], 'a') as handle:
    handle.write(json.dumps({'script': script, 'ranks': ranks, 'args': args})+'\n')
if os.environ.get('DIAG_TEST_FAIL') == script and ranks == 32:
    raise SystemExit(42)
if '--seal-suite' in args:
    sys.path.insert(0, os.environ['DIAG_TEST_REPO'])
    from validation.export_verified_quadrature_geometry import seal_suite
    def option(name):
        return args[args.index(name)+1]
    seal_suite(option('--seal-suite'), option('--input-json'), option('--image'), option('--repo'))
    raise SystemExit(0)
output = Path(args[args.index('--output')+1])
if script == 'preflight_diagnostics.py':
    output.mkdir()
    output = output/'preflight.json'
report = dict(passed=True, mpi_size=ranks, synthetic_boundary_only=True,
              mesh_partitioner='SCOTCH', mesh_sha256='synthetic-test-mesh',
              weights_verified=True, vector_point_major_ordering_verified_all_ranks=True,
              owned_index_and_rank_scatter_verified=True,
              saved_mapping_verified=True)
if '--input-json' in args:
    data = Path(args[args.index('--input-json')+1]).read_bytes()
    report['input_sha256'] = hashlib.sha256(data).hexdigest()
if script == 'validate_myosim_mixed_quadrature_ordering.py':
    report['ranks'] = [dict(rank=r, passed=True) for r in range(ranks)]
with output.open('x') as handle:
    json.dump(report, handle)
