
from jinja2 import Environment, FileSystemLoader
import subprocess
from datetime import date
import requests
from pathlib import Path

repo = "openmc-dev/openmc"
version = "v0.15.3"

# Get tag reference
url = f"https://api.github.com/repos/{repo}/git/ref/tags/{version}"
ref = requests.get(url).json()

# Annotated tags need one extra lookup
if ref["object"]["type"] == "tag":
    tag_obj = requests.get(ref["object"]["url"]).json()
    sha = tag_obj["object"]["sha"]
else:
    sha = ref["object"]["sha"]

# def read_result(file_path):
#     return (Path(file_path) / "results.txt").read_text()

def read_result(file_path):
    results_file = Path(file_path) / "results.txt"

    if not results_file.is_file():
        test_name = Path(file_path).name
        print(f"ccccccccccccccccccccc\nERROR: Test '{test_name}' was not run. Missing file: {results_file}\nccccccccccccccccccccc")
        return "Test not run"

    return results_file.read_text().strip()


# -----------------
# Data that normally comes from tests
# -----------------------------
data = {
    "cs_name": "OpenMC Development Community/OpenMC Project",
    "iv_name": "TBD",
    "openmc": {
        "version": version,
        "git_hash": sha,
        "nuclear_data": "ENDF/B-VIII.0"
    },
    "run": {
        "date": date.today().isoformat()
    },

    "results_functional": [
        {
            "id": "7.2.1.1",
            "title": "Neutron/Photon flux",
            "status": read_result("../functional_testing/7211_np_flux/")
        },
        {
            "id": "7.2.1.2.1",
            "title": "Nuclear heat",
            "status": read_result("../functional_testing/72121_nuclear_heat/")
        },
        {
            "id": "7.2.1.2.2",
            "title": "Tritium Production Rate",
            "status": read_result("../functional_testing/72122_tpr/")
        },
        {
            "id": "7.2.1.2.3",
            "title": "Dose",
            "status": read_result("../functional_testing/72123_dose/")
        },
        {
            "id": "7.2.1.2.4",
            "title": "Helium production rate",
            "status": read_result("../functional_testing/72124_hpr/")
        },
        {
            "id": "7.2.1.2.5",
            "title": "DPA",
            "status": read_result("../functional_testing/72125_dpa/")
        }
    ],
    "results_fundamental": [
        {
            "id": "7.2.2.1",
            "title": "Test of statistical error",
            "status": read_result("../fundamental_testing/7221_se/")
        },
        {
            "id": "7.2.2.2",
            "title": "Test of statistical error with weight window",
            "status": read_result("../fundamental_testing/7222_seww/")
        },
        {
            "id": "7.2.2.3",
            "title": "Test of random number generator - Knuth",
            "status": read_result("../fundamental_testing/7223_rng/1/")
        },
        {
            "id": "7.2.2.3",
            "title": "Test of random number generator - Diehard",
            "status": read_result("../fundamental_testing/7223_rng/2/")
        },
        {
            "id": "7.2.2.4 ",
            "title": "Test of source sampling Case 1",
            "status": read_result("../fundamental_testing/7224_ss/1/")
        },
        {
            "id": "7.2.2.4",
            "title": "Test of source sampling Case 2",
            "status": read_result("../fundamental_testing/7224_ss/2/")
        },
        {
            "id": "7.2.2.4",
            "title": "Test of source sampling Case 3",
            "status": read_result("../fundamental_testing/7224_ss/3/")
        },
        {
            "id": "7.2.2.4",
            "title": "Test of source sampling Case 4",
            "status": read_result("../fundamental_testing/7224_ss/4/")
        },
        {
            "id": "7.2.2.5",
            "title": "Test of conservation of energy for nuclear reaction",
            "status": read_result("../fundamental_testing/7225_ce/2/")
        }
    ]
}

# -----------------------------
# Load and render LaTeX template
# -----------------------------
env = Environment(loader=FileSystemLoader("."))
template = env.get_template("ITER-VandV-report.tex.j2")

rendered_tex = template.render(**data)

with open("ITER-VandV-report.tex", "w") as f:
    f.write(rendered_tex)

# -----------------------------
# Compile LaTeX -> PDF
# -----------------------------
subprocess.run(
    ["pdflatex", "-interaction=nonstopmode", "ITER-VandV-report.tex"],
    check=True
)
