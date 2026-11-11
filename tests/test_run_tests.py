import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('run_tests', ROOT / 'run_tests.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ReleaseTests(unittest.TestCase):
    def test_environment_pin(self):
        environment = (ROOT / 'environment.yml').read_text()
        self.assertIn(f'  - openmc={runner.OPENMC_VERSION}=nodagmc_nompi_*\n', environment)
        self.assertIn('  - python=3.12\n', environment)
        self.assertEqual(len(runner.TESTS), 10)

    def test_versions(self):
        for api_version, binary_version, valid in (
            ('0.16.0', '0.16.0', True),
            ('0.15.3', '0.16.0', False),
            ('0.16.0', '0.15.3', False),
            ('0.16.0', '0.16.0-dev', False),
        ):
            with self.subTest(api=api_version, binary=binary_version):
                with patch.dict(sys.modules, {'openmc': SimpleNamespace(__version__=api_version)}):
                    with patch.object(runner.shutil, 'which', return_value='/env/bin/openmc'):
                        with patch.object(runner.subprocess, 'check_output', return_value=f'OpenMC version {binary_version}\n'):
                            if valid:
                                self.assertEqual(runner.check_version()['openmc_version'], api_version)
                            else:
                                with self.assertRaises(RuntimeError):
                                    runner.check_version()

    def test_fresh_results(self):
        name = 'fundamental_testing/7221_se'
        for status, exit_code, expected in (('PASS', 0, 0), ('FAIL', 0, 1), ('PASS', 1, 1), (None, 0, 1)):
            with self.subTest(status=status, exit_code=exit_code):
                with tempfile.TemporaryDirectory() as directory:
                    output_root = Path(directory) / runner.OPENMC_VERSION

                    def execute(command, cwd, env):
                        self.assertEqual(command[0], sys.executable)
                        self.assertEqual(env['MPLBACKEND'], 'Agg')
                        self.assertEqual(sorted(path.name for path in cwd.iterdir()), ['run_test.py'])
                        if status is not None:
                            (cwd / 'results.txt').write_text(status + '\n')
                        return subprocess.CompletedProcess(command, exit_code)

                    with patch.object(runner, 'OUTPUT_ROOT', output_root):
                        with patch.object(runner, 'check_version', return_value={'openmc_version': '0.16.0'}):
                            with patch.object(runner.subprocess, 'run', side_effect=execute):
                                with patch('sys.stdout', new=io.StringIO()):
                                    self.assertEqual(runner.main([name]), expected)
                                metadata = json.loads((output_root / name / 'run_metadata.json').read_text())
                                self.assertEqual(metadata['status'], status or 'ERROR')
                                self.assertEqual(metadata['exit_code'], exit_code)
                                with self.assertRaises(RuntimeError):
                                    runner.main([name])

    def test_required_inputs_exist(self):
        for name, inputs in runner.INPUTS.items():
            for relative in inputs:
                self.assertTrue((runner.SOURCE_ROOT / name / relative).is_file())


if __name__ == '__main__':
    unittest.main()