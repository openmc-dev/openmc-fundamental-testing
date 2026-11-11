import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ExitCodeTests(unittest.TestCase):
    def test_verdicts(self):
        scripts = sorted(ROOT.glob('*/**/run_test.py'))
        self.assertTrue(scripts)
        for script in scripts:
            tree = ast.parse(script.read_text())
            verdict = next(
                index for index, node in enumerate(tree.body)
                if (isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'result'
                            for target in node.targets))
                or (isinstance(node, ast.Expr) and 'results.txt' in ast.unparse(node))
            )
            tail = ast.unparse(ast.Module(body=tree.body[verdict:], type_ignores=[]))
            for passed in (True, False):
                with self.subTest(script=script.relative_to(ROOT), passed=passed):
                    with tempfile.TemporaryDirectory() as directory:
                        binary = Path(directory) / 'pcg_random_bytes.bin'
                        binary.touch()
                        program = (
                            'from pathlib import Path\n'
                            f'overall_pass = {passed!r}\n'
                            f'frac_agree = {1.0 if passed else 0.0!r}\n'
                            'pass_fraction = 0.95\n' + tail
                        )
                        completed = subprocess.run(
                            [sys.executable, '-c', program], cwd=directory,
                            capture_output=True, text=True,
                        )
                        self.assertEqual(completed.returncode, 0 if passed else 1,
                                         completed.stderr)
                        expected = 'PASS\n' if passed else 'FAIL\n'
                        self.assertEqual((Path(directory) / 'results.txt').read_text(),
                                         expected)
                        self.assertNotIn('Traceback', completed.stderr)
                        if script.parent.name == '2' and script.parent.parent.name == '7223_rng':
                            self.assertFalse(binary.exists())


if __name__ == '__main__':
    unittest.main()