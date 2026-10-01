# %%
import ctypes
import subprocess
from pathlib import Path

import numpy as np
from openmc.lib import _dll

DIEHARDER_PATH = '/usr/bin/dieharder'
N_SAMPLES = 100_000_000
SEED = 123

_prn = _dll._ZN6openmc3prnEPm
_prn.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
_prn.restype = ctypes.c_double

# %%
def generate_openmc_samples(seed, count):
    state = ctypes.c_uint64(seed)
    return np.fromiter(
        (int(_prn(ctypes.byref(state)) * 2**32) for _ in range(count)),
        dtype=np.uint32, count=count,
    )


random_ints = generate_openmc_samples(SEED, N_SAMPLES)

# %%
binary_file = 'pcg_random_bytes.bin'
random_ints.tofile(binary_file)

# %%
# All avaliable tests
# TESTS_TO_RUN = [0, 2, 3, 4,5,6,7,8,9,10, 11, 12, 13, 15, 16, 17, 100, 101,102,200,201,202,203,201]
# Diehard original tests with 1 and 14 excluded as per DIEHARDER recomendation (https://webhome.phy.duke.edu/~rgb/General/dieharder.php)
TESTS_TO_RUN = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16]


def parse_dieharder_output(output):
    assessments = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.lstrip('# ').split('|')]
        if len(parts) < 6 or parts[0].lower() == 'test_name':
            continue
        status = parts[5].upper()
        if status not in ('PASSED', 'WEAK', 'FAILED'):
            continue
        try:
            pvalue = float(parts[4])
        except ValueError:
            continue
        assessments.append({'test': parts[0], 'pvalue': pvalue, 'status': status})
    return assessments

# %%
all_results = []
raw_outputs = []
errors = []
for test_id in TESTS_TO_RUN:
    try:
        output = subprocess.run(
            [DIEHARDER_PATH, '-d', str(test_id), '-g', '201', '-f', str(binary_file)],
            capture_output=True, text=True, timeout=300, check=True,
        ).stdout
        raw_outputs.append((test_id, output))
        assessments = parse_dieharder_output(output)
        if not assessments:
            errors.append((test_id, 'No assessments parsed'))
        for assessment in assessments:
            assessment['test_id'] = test_id
        all_results.extend(assessments)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        errors.append((test_id, str(exc)))

# %%
assessed_ids = {row['test_id'] for row in all_results}
overall_pass = (
    assessed_ids == set(TESTS_TO_RUN)
    and all(row['status'] in ('PASSED', 'WEAK') for row in all_results)
    and not errors
)

report_lines = ['DIEHARD TEST RESULTS']
for row in all_results:
    report_lines.append(
        f"{row['test_id']:>3} {row['test']:<32} {row['pvalue']!s:<12} {row['status']}"
    )
for test_id, message in errors:
    report_lines.append(f'{test_id:>3} ERROR: {message}')
report_lines.append(f'OVERALL: {"PASS" if overall_pass else "FAIL"}')
for test_id, output in raw_outputs:
    report_lines.extend((f'\nRAW TEST {test_id}', output))
Path('dieharder_report.txt').write_text('\n'.join(report_lines) + '\n')
Path('results.txt').write_text('PASS\n' if overall_pass else 'FAIL\n')
Path('pcg_random_bytes.bin').unlink(missing_ok=True)


