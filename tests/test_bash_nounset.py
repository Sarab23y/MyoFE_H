"""Stdlib-only launcher argument tests; run with the actual LCC /bin/bash."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get('DIAGNOSTIC_TEST_BASH') or shutil.which('bash')
if not os.environ.get('DIAGNOSTIC_TEST_BASH') and os.name == 'nt':
    BASH = 'C:/Program Files/Git/bin/bash.exe'


@unittest.skipUnless(BASH, 'Bash is unavailable')
class BashNounsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        version = subprocess.check_output([BASH, '--version']).decode().splitlines()[0]
        print('ACTUAL BASH UNDER TEST: '+version, flush=True)

    def arguments(self, snippet, **variables):
        env = dict(os.environ, **variables)
        run = subprocess.run([BASH, '-c', 'set -euo pipefail\n'+snippet+
                              '\nfor arg in "$@"; do printf "%s\\0" "$arg"; done'],
                             env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(run.returncode, 0, run.stderr.decode())
        return run.stdout.decode().rstrip('\0').split('\0') if run.stdout else []

    def test_real_mapping_argument_block_serial_and_parallel(self):
        source = (ROOT/'validation/run_phase_mapping_validation.slurm').read_text()
        start = source.index('  set -- --input-json')
        end = source.index('\n  singularity exec', start)
        block = source[start:end]
        for n in ('1', '32'):
            args = self.arguments(block, N=n, SLURM_NTASKS='32', INPUT='/input with spaces.json',
                                  OUT='/new reports', DATA='/saved results')
            expected = ['--input-json', '/input with spaces.json', '--output', '/new reports/geometry_%s.json'%n]
            self.assertEqual(args, expected+(['--saved-output', '/saved results'] if n == '32' else []))

    def test_real_region_block_all_optional_combinations(self):
        source = (ROOT/'validation/run_phase_region_analysis.slurm').read_text()
        block = source[source.index('set --\n'):source.index('# The analyzer')]
        for start, end in (('', ''), ('8', ''), ('', '9'), ('8', '9')):
            args = self.arguments(block, START=start, END=end)
            self.assertEqual(args, (['--start', start] if start else [])+(['--end', end] if end else []))

    def test_every_diagnostic_launcher_parses_and_has_no_array_expansion(self):
        paths = sorted(list((ROOT/'validation').glob('*.slurm'))+list((ROOT/'validation').glob('*.sh')))
        self.assertTrue(paths)
        for path in paths:
            self.assertNotRegex(path.read_text(), r'\$\{[^}]*\[@\][^}]*\}', str(path))
            # Feed source through stdin: works with Windows and native Bash paths.
            result = subprocess.run([BASH, '-n'], input=path.read_bytes(),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 0, str(path)+result.stderr.decode())

    def test_original_empty_array_behavior_is_reported_not_simulated(self):
        run = subprocess.run([BASH, '-uc', 'EXTRA=(); printf "ok\\n" "${EXTRA[@]}"'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print('Original empty-array probe: exit=%d stderr=%s' %
              (run.returncode, run.stderr.decode().strip()), flush=True)
        # Do not pretend BASH_COMPAT emulates old nounset bugs on modern Bash.


if __name__ == '__main__':
    unittest.main()
