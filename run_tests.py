import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


OPENMC_VERSION = '0.16.0'
ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / '0.15.3'
OUTPUT_ROOT = ROOT / OPENMC_VERSION
TESTS = tuple(sorted(
    script.parent.relative_to(SOURCE_ROOT).as_posix()
    for script in SOURCE_ROOT.glob('fundamental_testing/**/run_test.py')
))
INPUTS = {
    'fundamental_testing/7222_seww': ('model.xml',),
    'fundamental_testing/7224_ss/4': (
        'openmc_run/model.xml', 'mcnp_run/c_source-batch.h5',
    ),
}


def check_version():
    import openmc

    if sys.version_info < (3, 12):
        raise RuntimeError('OpenMC 0.16.0 requires Python 3.12 or newer')
    if openmc.__version__ != OPENMC_VERSION:
        raise RuntimeError(f'Expected OpenMC {OPENMC_VERSION}, found Python API {openmc.__version__}')
    executable = shutil.which('openmc')
    if executable is None:
        raise RuntimeError('The openmc executable is not on PATH')
    output = subprocess.check_output([executable, '--version'], text=True)
    match = re.search(r'^OpenMC version (\S+)', output, re.MULTILINE)
    if match is None or match.group(1) != OPENMC_VERSION:
        raise RuntimeError(f'Expected OpenMC {OPENMC_VERSION} executable, found: {output.strip()}')
    return {
        'openmc_version': openmc.__version__,
        'executable': executable,
        'executable_version': match.group(1),
        'python_version': platform.python_version(),
        'cross_sections': os.environ.get('OPENMC_CROSS_SECTIONS'),
    }


def prepare_test(name):
    destination = OUTPUT_ROOT / name
    destination.mkdir(parents=True, exist_ok=False)
    for relative in ('run_test.py', *INPUTS.get(name, ())):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_ROOT / name / relative, target)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run fresh tests with pinned OpenMC 0.16.0.')
    parser.add_argument('--list', action='store_true', help='List available fundamental tests')
    parser.add_argument('tests', nargs='*', help='Test directories shown by --list')
    args = parser.parse_args(argv)
    if args.list:
        print('\n'.join(TESTS))
        return 0
    if not args.tests or any(name not in TESTS for name in args.tests):
        parser.error('Choose test directories from --list')
    if len(set(args.tests)) != len(args.tests):
        parser.error('Choose each test only once')
    metadata = check_version()
    if any((OUTPUT_ROOT / name).exists() for name in args.tests):
        raise RuntimeError('A selected output directory already exists; archive it before rerunning')
    failed = False
    for name in args.tests:
        destination = prepare_test(name)
        completed = subprocess.run(
            [sys.executable, str(destination / 'run_test.py')], cwd=destination,
            env={**os.environ, 'MPLBACKEND': 'Agg'},
        )
        result = destination / 'results.txt'
        status = result.read_text().strip() if result.is_file() else 'ERROR'
        failed |= completed.returncode != 0 or status != 'PASS'
        (destination / 'run_metadata.json').write_text(json.dumps({
            **metadata, 'test': name, 'status': status,
            'exit_code': completed.returncode,
            'date': datetime.now(timezone.utc).isoformat(),
        }, indent=2) + '\n')
        print(f'{name}: {status}', flush=True)
    return int(failed)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ImportError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))