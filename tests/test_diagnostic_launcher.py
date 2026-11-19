"""Execute the real Bash launcher with a marked synthetic container boundary."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get('DIAGNOSTIC_TEST_BASH') or shutil.which('bash')
if not os.environ.get('DIAGNOSTIC_TEST_BASH') and os.name == 'nt' and Path('C:/Program Files/Git/bin/bash.exe').exists():
    BASH = 'C:/Program Files/Git/bin/bash.exe'


@unittest.skipUnless(BASH, 'Bash unavailable for launcher control-flow tests')
class DiagnosticLauncherTests(unittest.TestCase):
    def execute(self, directory, fail=None):
        directory = Path(directory)
        def shell_path(path):
            value = Path(path).as_posix()
            return '/'+value[0].lower()+value[2:] if os.name == 'nt' else value
        commands = directory/'bin'
        commands.mkdir(exist_ok=True)
        singularity = commands/'singularity'
        singularity.write_text('#!/bin/sh\nexec "$DIAG_TEST_PYTHON" "$DIAG_TEST_DRIVER" "$@"\n')
        singularity.chmod(0o755)
        image = directory/'fenics.img'
        image.write_text('Synthetic fixture')
        instruction = directory/'instruction.json'
        instruction.write_text('{}')
        data = directory/'sim_output'
        data.mkdir(exist_ok=True)
        env = dict(os.environ, REPO=shell_path(ROOT), IMAGE=shell_path(image),
                   INPUT=shell_path(instruction), DATA=shell_path(data), OUT=shell_path(directory/'reports'),
                   SLURM_NTASKS='32', SLURM_JOB_ID='synthetic-test',
                   DIAG_TEST_PYTHON=sys.executable.replace('\\', '/'),
                   BASH_TEST_PYTHON=sys.executable.replace('\\', '/'),
                   DIAG_TEST_DRIVER=(ROOT/'tests/fake_diagnostic_container.py').as_posix(),
                   DIAG_TEST_REPO=str(ROOT), DIAG_TEST_TRACE=str(directory/'commands.jsonl'))
        if fail:
            env['DIAG_TEST_FAIL'] = fail
        # Bash PATH uses ':' even when the parent is Windows PowerShell.
        driver = 'export PATH="$DIAG_TEST_BIN:/usr/bin:/bin:$PATH"; exec "$DIAG_TEST_BASH" "$DIAG_TEST_LAUNCHER"'
        # Git/bin/bash.exe is a native Windows wrapper that converts env paths;
        # use its actual MSYS interpreter for the inner shell.
        env['DIAG_TEST_BASH'] = '/usr/bin/bash' if os.name == 'nt' and 'Git/bin/bash' in BASH else BASH
        env['DIAGNOSTIC_TEST_BASH'] = BASH
        env['DIAG_TEST_BIN'] = shell_path(commands)
        env['DIAG_TEST_LAUNCHER'] = shell_path(ROOT/'validation/run_phase_mapping_validation.slurm')
        return subprocess.run([BASH, '-c', driver], env=env, cwd=directory,
                              capture_output=True, text=True, timeout=60)

    def test_launcher_success_sequence_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.execute(directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = Path(directory)/'reports/mapping_manifest.json'
            original = report.read_bytes()
            commands = [json.loads(line) for line in (Path(directory)/'commands.jsonl').read_text().splitlines()]
            self.assertEqual(len(commands), 12)  # preflight + 8 MPI commands + 2 comparisons + seal
            self.assertEqual([c['ranks'] for c in commands[1:9]], [1]*4+[32]*4)
            again = self.execute(directory)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(report.read_bytes(), original)

    def test_failed_parallel_stage_preserves_partial_reports_without_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.execute(directory, 'validate_ecm_quadrature_mpi.py')
            self.assertEqual(result.returncode, 42, result.stderr)
            output = Path(directory)/'reports'
            self.assertTrue((output/'ecm_1.json').exists())
            self.assertFalse((output/'mapping_manifest.json').exists())
            self.assertIn('exit=42', (output/'failure.txt').read_text())


if __name__ == '__main__':
    unittest.main()
