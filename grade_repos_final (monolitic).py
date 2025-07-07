#!/usr/bin/env python3
"""
grade_repos_final.py – IE University · Intro MLOps Automatic Rubric Grader

This script provides a comprehensive, automated grading solution for MLOps projects.
It systematically runs a suite of static analysis, testing, and MLOps integration
checks against a collection of student repositories.

Key Features:
- Executes over 20 distinct tests covering code quality, testing, security,
  documentation, and integration with tools like MLflow, DVC, and Docker.
- Uses a combination of absolute, relative, and checklist-based scoring for
  fair and nuanced evaluation.
- Implements robust logging for traceability and easier debugging of test failures.
- Generates a detailed report in `grading_results.xlsx` with raw results,
  normalized scores, and actionable feedback for each repository.
- Provides a command-line interface (CLI) to run specific tests or grade a
  single repository, facilitating targeted analysis.

Dependencies:
- pandas: For generating the final Excel report (`pip install pandas`).
- openpyxl: Required by pandas for `.xlsx` support (`pip install openpyxl`).
- PyYAML: For safely parsing YAML configuration files (`pip install pyyaml`).
- All command-line tools specified in the `TESTS` registry (e.g., black,
  pylint, detect-secrets) must be installed and available in the system's PATH.

Usage:
1.  Place the script in a root folder.
2.  Create a subfolder named `repos_to_grade/` and clone all student
    repositories into it.
3.  Run the script from the root folder:
    `python grade_repos_final.py`
4.  To run a single test on all repos:
    `python grade_repos_final.py [test_key]`
    (e.g., `python grade_repos_final.py pytest`)
5.  To run all tests on a single repository:
    `python grade_repos_final.py --repo [repo_name]`
    (e.g., `python grade_repos_final.py --repo student_project_1`)

---
Potential Modular Structure for Scaling:

As this script grows, it could be refactored into a more modular package:

- `main.py`: Handles CLI parsing and orchestrates the grading process.
- `config.py`: Contains the `TESTS` and `COMMENTS_MAP` dictionaries.
- `runners.py`: Includes functions that execute external processes, like `run_subprocess`
  and the specific test runners (e.g., `run_secrets_scan`).
- `parsers.py`: Contains functions dedicated to parsing the output from shell
  commands and files (e.g., `parse_pytest_output`, `parse_yaml_safe`).
- `scoring.py`: Manages score calculation and normalization logic.
- `reporting.py`: Handles the generation of the final Excel report.
---
"""
from __future__ import annotations

# Standard library imports
import ast
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Third-party imports
try:
    import pandas as pd
    import yaml
except ImportError as e:
    print(
        f"Error: Missing required libraries. Please run 'pip install pandas openpyxl pyyaml'. Details: {e}")
    sys.exit(1)


# --- Global Configuration ---

# Suppress openpyxl UserWarning about data validation
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


# --- Constants and Maps ---

COMMENTS_MAP = {
    # Production-Ready Code
    "black --check .": "Run `black .` to auto-format the code",
    "isort --check-only .": "Run `isort .` to auto-format all imports",
    "flake8 .": "Action on console feedback; Use auto-formatters (black, isort)",
    "flake8 --select=LOG .": "Action on console feedback for logging practices",
    "pylint src --score y": "Action on console feedback to improve code quality",
    "pytest --cov=src": "Improve test coverage and fix failing tests",
    "radon cc -s -a src": "Refactor high-complexity functions to improve maintainability",
    "detect-secrets scan": "Review and remove any found secrets; use environment variables",
    "pydocstyle src": "Add or fix docstrings according to style guidelines",
    "Config usage scan": "Remove hard-coded paths and use configuration files",
    "Error-handling scan": "Increase use of try/except blocks for better robustness",
    # MLOps Integration
    "MLflow Pipeline Orchestration": "The main script should loop through steps in the config and execute each one using `mlflow.run()`.",
    "Hydra Config check": "Ensure conf/config.yaml is well-formed with main, steps, and params keys",
    "W&B Integration check": "Code should use wandb.init, wandb.log, and log_artifact for tracking",
    "GitHub Actions CI check": "The ci.yml workflow should run tests, the pipeline, and use secrets",
    "Dockerfile check": "Dockerfile should follow best practices (e.g., layer caching, non-root user)",
    "FastAPI Endpoint check": "A POST endpoint must be defined at '/predict' using AST validation",
    "Pipeline Modularity check": "Project structure in 'src/' must match the 'steps' list in Hydra config",
    "README Documentation check": "README should clearly document MLflow, W&B, Docker, and API usage",
    "DVC Usage check": "Project must use DVC to track at least one large file (>10MB)",
    "Render Deployment check": "A 'render.yaml' file must exist and declare a web service",
}


# --- Helper & Utility Functions ---

def run_subprocess(cmd: List[str], cwd: Path, timeout: int = 300, env: Dict[str, str] = None) -> Tuple[str, int]:
    """
    Executes a shell command safely and captures its output.

    Logs errors with context instead of printing directly to stderr.

    Args:
        cmd: The command to execute as a list of strings.
        cwd: The working directory for the command.
        timeout: The timeout in seconds for the command.
        env: A dictionary of environment variables to set or augment.

    Returns:
        A tuple containing the command's stdout and its return code.
    """
    try:
        process_env = os.environ.copy()
        if env:
            for key, value in env.items():
                # Prepend to PATH-like variables, otherwise overwrite
                if key in process_env and key.endswith("PATH"):
                    process_env[key] = f"{value}{os.pathsep}{process_env[key]}"
                else:
                    process_env[key] = value

        res = subprocess.run(
            cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout, env=process_env, check=False
        )

        if res.returncode != 0 and res.stderr:
            logging.debug(
                "Command failed: '%s' in repo '%s'. Stderr: %s",
                " ".join(cmd),
                cwd.name,
                res.stderr.strip()
            )

        return res.stdout, res.returncode

    except subprocess.TimeoutExpired:
        logging.error("Command timed out: '%s' in repo '%s'",
                      " ".join(cmd), cwd.name)
        return "TIMEOUT", 1
    except FileNotFoundError:
        logging.error(
            "Command not found: '%s'. Please ensure it is installed and in your PATH.", cmd[0])
        return f"COMMAND NOT FOUND: '{cmd[0]}'", 1
    except Exception as e:
        logging.error("An unexpected error occurred while running command '%s' in '%s': %s", " ".join(
            cmd), cwd.name, e)
        return f"UNEXPECTED ERROR: {e}", 1


def parse_yaml_safe(path: Path) -> Dict[str, Any]:
    """
    Safely parses a YAML file using PyYAML.

    Args:
        path: The path to the YAML file.

    Returns:
        A dictionary with the parsed YAML content, or an empty dictionary on failure.
    """
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, IOError) as e:
        logging.warning("Could not parse YAML file at '%s': %s", path, e)
        return {}


def scan_file_for_patterns(fpath: Path, patterns: Dict[str, str]) -> Dict[str, bool]:
    """
    Scans a single file for a dictionary of regex patterns.

    Args:
        fpath: The path to the file to scan.
        patterns: A dictionary where keys are check names and values are regex patterns.

    Returns:
        A dictionary with boolean results for each pattern.
    """
    results = {key: False for key in patterns}
    if not fpath.exists():
        return results
    try:
        content = fpath.read_text(encoding="utf-8", errors="ignore")
        for key, pattern in patterns.items():
            if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                results[key] = True
    except Exception as e:
        logging.warning("Could not read or scan file '%s': %s", fpath, e)
    return results


def find_config_file(repo: Path) -> Path | None:
    """
    Finds the main config.yaml file in common project locations.

    Args:
        repo: The path to the repository's root directory.

    Returns:
        The path to the config file if found, otherwise None.
    """
    common_paths = [
        repo / "conf" / "config.yaml",
        repo / "configs" / "config.yaml",
        repo / "src" / "config.yaml",
        repo / "config.yaml",
    ]
    for path in common_paths:
        if path.exists():
            return path
    return None


def calculate_checklist_score(results: Dict[str, bool]) -> Tuple[float, str]:
    """
    Calculates a score based on a checklist of boolean results.

    Args:
        results: A dictionary of boolean check results.

    Returns:
        A tuple containing the calculated score (0-10) and a display string (e.g., "2/3 passed").
    """
    if not isinstance(results, dict):
        return 0.0, "Check failed (invalid data)"

    passed = sum(bool(v) for v in results.values())
    total = len(results)
    score = (passed / total) * 10 if total > 0 else 0
    display = f"{passed}/{total} checks passed"
    return score, display


# --- Test-Specific Parsers & Runners ---

def parse_pytest_output(out: str) -> Tuple[int, int, int]:
    """
    Parses pytest output for passed tests and test coverage.

    This parser prioritizes the detailed coverage report but falls back to the
    summary total if the table is not found. It excludes common entry-point
    files from coverage calculation for fairness.

    Args:
        out: The stdout string from the pytest command.

    Returns:
        A tuple containing (tests_passed, tests_total, coverage_percentage).
    """
    passed = total = 0
    # Match summary line like "== 42 passed, 3 skipped in 5.12s =="
    summary_match = re.search(r"(\d+)\s+passed", out)
    if summary_match:
        passed = int(summary_match.group(1))

    # Match collected items line like "collected 42 items"
    collected_match = re.search(r"collected\s+(\d+)\s+items", out)
    if collected_match:
        total = int(collected_match.group(1))

    # Fallback for total if only summary is present
    if total == 0 and passed > 0:
        total = passed

    # --- Fair Coverage Calculation ---
    total_relevant_stmts = 0
    total_relevant_miss = 0
    filenames_to_exclude = {'main.py', 'run.py', 'app.py'}
    fair_coverage = 0

    try:
        # Find the start of the coverage report table
        table_header_match = re.search(r"Name\s+Stmts\s+Miss", out)
        if not table_header_match:
            raise ValueError("Coverage table not found")

        report_table = out[table_header_match.end():]
        for line in report_table.splitlines():
            # Match lines like "src/data.py   15   0   100%"
            match = re.match(
                r"^(?P<name>\S+)\s+(?P<stmts>\d+)\s+(?P<miss>\d+)", line)
            if not match:
                continue

            file_path = match.group('name')
            # Only include files from 'src/' and exclude common entry points
            if 'src/' in file_path and os.path.basename(file_path) not in filenames_to_exclude:
                total_relevant_stmts += int(match.group('stmts'))
                total_relevant_miss += int(match.group('miss'))

        if total_relevant_stmts > 0:
            coverage_fraction = (total_relevant_stmts -
                                 total_relevant_miss) / total_relevant_stmts
            fair_coverage = round(100 * coverage_fraction)
        elif passed > 0:  # If tests passed but no relevant code was covered, assume 100%
            fair_coverage = 100

    except (ValueError, AttributeError):
        # Fallback to total coverage percentage if detailed parsing fails
        cov_match = re.search(r"TOTAL.*\s(\d+)%", out)
        if cov_match:
            fair_coverage = int(cov_match.group(1))

    return passed, total, fair_coverage


def parse_pydocstyle_output(out: str, repo: Path) -> Tuple[int, int]:
    """
    Parses pydocstyle output to count violations and scanned files.

    Args:
        out: The stdout string from the pydocstyle command.
        repo: The path to the repository, used to count total Python files.

    Returns:
        A tuple containing (violation_count, file_count).
    """
    # A violation line typically looks like: "src/module.py:12 D100 Missing docstring..."
    violations = sum(1 for line in out.splitlines()
                     if re.match(r"^src/.*:\d+", line))
    num_files = sum(1 for _ in (repo / "src").rglob("*.py"))
    return violations, num_files


def parse_detect_secrets_output(out: str, repo_name: str) -> int:
    """
    Parses the JSON output from detect-secrets.

    Args:
        out: The JSON string output from the detect-secrets command.
        repo_name: The name of the repository being scanned, for logging.

    Returns:
        The number of files with detected secrets (0 for clean).
    """
    logging.debug("Raw detect-secrets output for repo '%s': %s",
                  repo_name, out.strip() or "[EMPTY]")
    try:
        if not out.strip():
            return 0  # No output means no secrets found

        data = json.loads(out)
        results = data.get("results")
        if results:
            logging.info("Secrets found in repo '%s'. Evidence:", repo_name)
            for file_path, findings in results.items():
                for finding in findings:
                    logging.info(
                        "  - File: %s, Line: %s, Type: %s",
                        file_path,
                        finding["line_number"],
                        finding["type"],
                    )
            return len(results)  # Return number of files with secrets
        return 0
    except json.JSONDecodeError:
        logging.error(
            "Failed to parse detect-secrets JSON output for repo '%s'. Assuming failure.", repo_name)
        return 1  # Treat parsing failure as a found secret for safety


def run_secrets_scan(repo: Path) -> int:
    """
    Dynamically builds and runs a targeted detect-secrets scan.

    Scans high-risk files and directories while excluding common data and
    dependency files to reduce false positives.

    Args:
        repo: The path to the repository's root directory.

    Returns:
        The number of files with detected secrets.
    """
    potential_targets = ['src', 'conf', '.github',
                         'Dockerfile', 'render.yaml', 'MLproject']
    existing_targets = [
        target for target in potential_targets if (repo / target).exists()]
    if not existing_targets:
        return 0

    # Exclude common file types that are not source code
    exclude_files = r"\.(ipynb|txt|csv|parquet|pkl|h5|model|md)$"
    # Exclude directories that store run artifacts
    exclude_dirs = r"wandb|mlruns|\.dvc/cache"

    command = ["detect-secrets", "scan"] + existing_targets + \
        ["--exclude-files", exclude_files, "--exclude-dirs", exclude_dirs]
    output, _ = run_subprocess(command, repo, timeout=300)
    return parse_detect_secrets_output(output, repo.name)


def check_config_usage(src: Path) -> float:
    """
    Scans for hard-coded absolute file paths in the 'src' directory.

    Args:
        src: The path to the 'src' directory.

    Returns:
        1.0 if no hard-coded paths are found, 0.0 otherwise.
    """
    if not src.is_dir():
        logging.debug(
            "Config usage check: 'src' directory not found in '%s'", src.parent.name)
        return 0.0

    # This regex looks for paths like "C:\..." or "/usr/..."
    hardcoded_path_pattern = r'["\']([A-Za-z]:[\\/][^"\']+|/[^"\']{2,})["\']'
    found_paths = []

    for py_file in src.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for match in re.finditer(hardcoded_path_pattern, content):
                # Ignore relative paths starting with './' or '/.'
                if not match.group(1).startswith(('./', '/.')):
                    found_paths.append(
                        f"  - File: {py_file.relative_to(src.parent)}, Path: {match.group(0)}")
        except Exception as e:
            logging.warning(
                "Could not read file '%s' for config usage check: %s", py_file, e)

    if found_paths:
        logging.info("Hard-coded paths found in repo '%s':\n%s",
                     src.parent.name, "\n".join(found_paths))
        return 0.0
    return 1.0


def check_error_handling(src: Path) -> float:
    """
    Calculates an error-handling score based on `try...except` usage.

    Identifies functions performing "risky" operations (e.g., I/O, network
    requests) and checks if they are wrapped in a `try...except` block.

    Args:
        src: The path to the 'src' directory.

    Returns:
        A ratio (0.0 to 1.0) of protected functions to total risky functions.
    """
    if not src.is_dir():
        return 0.0

    RISKY_FUNCTION_CALLS = {
        'open', 'load', 'loads', 'dump', 'dumps',
        'read_csv', 'to_csv', 'read_parquet', 'to_parquet', 'read_excel',
        'get', 'post', 'put', 'delete',
        'connect', 'execute', 'cursor'
    }
    candidate_functions = 0
    protected_functions = 0

    for py_file in src.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(
                encoding="utf-8", errors="ignore"))
            for func_node in ast.walk(tree):
                if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

                is_risky = any(
                    (isinstance(sub_node, ast.Call) and
                     hasattr(sub_node.func, 'attr') and
                     isinstance(sub_node.func, ast.Attribute) and
                     sub_node.func.attr in RISKY_FUNCTION_CALLS)
                    for sub_node in ast.walk(func_node)
                )

                if is_risky:
                    candidate_functions += 1
                    # Check if the function body contains a 'try' block
                    if any(isinstance(n, ast.Try) for n in func_node.body):
                        protected_functions += 1
        except Exception as e:
            logging.warning("AST parsing failed for file '%s': %s", py_file, e)

    if candidate_functions == 0:
        return 1.0  # No risky functions found, so no errors to handle

    return protected_functions / candidate_functions


def check_mlflow_orchestration(repo: Path) -> Tuple[float, str]:
    """
    Checks for valid MLflow orchestration patterns.

    Prioritizes a Python-based orchestrator using `mlflow.run()` or a
    centralized `MLproject` file integrated with CI.

    Args:
        repo: The path to the repository's root directory.

    Returns:
        A tuple containing the score (0-10) and a description of the detected pattern.
    """
    highest_score = 0.0
    best_description = "No clear orchestration pattern detected"

    # Pattern 1: Python orchestrator with `mlflow.run()` (10/10)
    main_py = repo / "main.py" or repo / "src" / "main.py"
    config_file = find_config_file(repo)
    if main_py.exists() and config_file:
        try:
            tree = ast.parse(main_py.read_text(
                encoding="utf-8", errors="ignore"))
            if any(
                isinstance(node, ast.Call) and
                isinstance(getattr(node.func, 'value', None), ast.Name) and
                node.func.value.id == 'mlflow' and
                getattr(node.func, 'attr', '') == 'run'
                for node in ast.walk(tree)
            ):
                highest_score = 10.0
                best_description = "Excellent: Python-based orchestrator using mlflow.run"
        except Exception as e:
            logging.debug(
                "AST parsing for MLflow orchestration failed in '%s': %s", repo.name, e)

    # Pattern 2: Centralized MLproject (10/10 with CI, 8/10 without)
    if (repo / "MLproject").exists():
        current_score = 8.0
        current_desc = "Good: Centralized MLproject found"
        # Check for CI integration
        workflow_dir = repo / ".github" / "workflows"
        if workflow_dir.is_dir():
            for workflow_file in workflow_dir.glob("*.yml"):
                content = workflow_file.read_text(
                    encoding="utf-8", errors="ignore")
                if "mlflow run" in content:
                    current_score = 10.0
                    current_desc = "Excellent: Centralized MLproject with CI integration"
                    break
        if current_score > highest_score:
            highest_score = current_score
            best_description = current_desc

    return highest_score, best_description


def check_pipeline_modularity(repo: Path) -> float:
    """
    Compares declared pipeline steps in config with the `src` directory structure.

    This function uses fuzzy matching and synonym checks to link declared steps
    (e.g., "data_validation") to implemented components (e.g., `src/validation/`
    or `src/validate.py`).

    Args:
        repo: The path to the repository's root directory.

    Returns:
        A ratio (0.0 to 1.0) of matched steps to declared steps.
    """
    config_path = find_config_file(repo)
    if not config_path:
        return 0.0

    config_data = parse_yaml_safe(config_path)
    steps_raw = config_data.get("main", {}).get(
        "steps") or config_data.get("steps")
    declared_steps = []
    if isinstance(steps_raw, list):
        declared_steps = [str(s) for s in steps_raw]
    elif isinstance(steps_raw, str):
        declared_steps = [s.strip() for s in steps_raw.split(',') if s.strip()]

    if not declared_steps:
        return 0.0

    src_root = repo / "src"
    if not src_root.is_dir():
        return 0.0

    # Discover components: any directory or top-level .py file in src
    impl_components = {
        item.stem for item in src_root.iterdir() if item.stem != '__init__'}

    # --- Matching Logic ---
    from difflib import SequenceMatcher
    SYNONYMS = {'predict': 'inference', 'train': 'model',
                'eval': 'evaluation', 'validate': 'validation'}

    def normalize(name: str) -> str:
        name = name.lower().replace('_', '').replace('-', '')
        return SYNONYMS.get(name, name)

    matched_count = 0
    used_components = set()

    for step in declared_steps:
        best_match_component = None
        highest_ratio = 0.75  # Minimum similarity threshold

        for component in impl_components:
            if component in used_components:
                continue

            # Check for exact or synonym match first
            if normalize(step) == normalize(component):
                best_match_component = component
                break

            # Check fuzzy string similarity
            ratio = SequenceMatcher(None, normalize(
                step), normalize(component)).ratio()
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match_component = component

        if best_match_component:
            matched_count += 1
            used_components.add(best_match_component)

    return matched_count / len(declared_steps)


# --- Test Registry ---

TESTS: Dict[str, Dict[str, Any]] = {
    # --- Production-Ready Code ---
    "black": {"label": "black --check .", "purpose": "PEP 8 formatting", "dimension": "Code Quality & Efficiency", "run": lambda repo: sum(1 for ln in run_subprocess(["black", "--check", "--verbose", "."], repo)[0].splitlines() if "would be reformatted" in ln), "scoring_type": "relative_inverted"},
    "isort": {"label": "isort --check-only .", "purpose": "Import order style", "dimension": "Code Quality & Efficiency", "run": lambda repo: sum(1 for ln in run_subprocess(["isort", "--check-only", "."], repo)[0].splitlines() if ln.lstrip().startswith("ERROR:")), "scoring_type": "relative_inverted"},
    "flake8": {"label": "flake8 .", "purpose": "Linting for style & bugs", "dimension": "Code Quality & Efficiency", "run": lambda repo: len(run_subprocess(["flake8", "."], repo)[0].splitlines()), "scoring_type": "relative_inverted"},
    "flake8_log": {"label": "flake8 --select=LOG .", "purpose": "Logging best practice", "dimension": "Logging & Monitoring", "run": lambda repo: len(run_subprocess(["flake8", "--select=LOG", "."], repo)[0].splitlines()), "scoring_type": "relative_inverted"},
    "pylint": {"label": "pylint src --score y", "purpose": "Advanced linting", "dimension": "Code Quality & Documentation", "run": lambda repo: float(m.group(1)) if (m := re.search(r"rated at ([\d\.-]+)/10", run_subprocess(["pylint", "src", "--score", "y"], repo, timeout=180)[0])) else 0.0, "scoring_type": "absolute"},
    "pytest": {"label": "pytest --cov=src", "purpose": "Unit tests & coverage", "dimension": "Testing & Coverage", "run": lambda repo: parse_pytest_output(run_subprocess(["python", "-m", "pytest", "-q", "--cov=src", "--cov-report=term-missing", "--cov-omit", "*/run.py,*/main.py,*/app.py", "tests"], repo, 600, env={"PYTHONPATH": f".{os.pathsep}{repo / 'src'}", "WANDB_MODE": "disabled"})[0]), "scoring_type": "custom_pytest"},
    "radon": {"label": "radon cc -s -a src", "purpose": "Cyclomatic complexity", "dimension": "Modularization & Code Quality", "run": lambda repo: {"A": 10, "B": 8, "C": 6, "D": 4, "E": 2, "F": 0}.get(m.group(1) if (m := re.search(r"Average complexity:\s+([A-F])", run_subprocess(["radon", "cc", "-s", "-a", "src"], repo)[0])) else "F", 0), "scoring_type": "absolute"},
    "secrets": {"label": "detect-secrets scan", "purpose": "Detect hard-coded secrets", "dimension": "Security", "run": run_secrets_scan, "scoring_type": "binary_inverted"},
    "pydocstyle": {"label": "pydocstyle src", "purpose": "Docstring presence & style", "dimension": "Documentation & Clarity", "run": lambda repo: parse_pydocstyle_output(run_subprocess(["pydocstyle", "src"], repo)[0], repo), "scoring_type": "custom_pydocstyle"},
    "config_usage": {"label": "Config usage scan", "purpose": "Hard-coded paths vs config", "dimension": "Config Mgmt", "run": lambda repo: check_config_usage(repo / "src"), "scoring_type": "binary_10_0"},
    "error_handling": {"label": "Error-handling scan", "purpose": "try/except usage", "dimension": "Error Handling & Validation", "run": lambda repo: check_error_handling(repo / "src"), "scoring_type": "relative_direct"},

    # --- MLOps Integration ---
    "mlflow_orchestration": {"label": "MLflow Pipeline Orchestration", "purpose": "Validate the method used to orchestrate pipeline steps", "dimension": "MLflow Integration", "run": check_mlflow_orchestration, "scoring_type": "custom_tuple"},
    "hydra_config": {"label": "Hydra Config check", "purpose": "Configuration management", "dimension": "Hydra Config Management", "run": lambda repo: (lambda conf: {"has_main": "main" in conf, "has_steps": "steps" in conf.get("main", {}), "is_valid_yaml": bool(conf)})(parse_yaml_safe(find_config_file(repo))) if find_config_file(repo) else {}, "scoring_type": "checklist"},
    "wandb_integration": {"label": "W&B Integration check", "purpose": "Experiment & artifact tracking", "dimension": "W&B Tracking", "run": lambda repo: {check: any(scan_file_for_patterns(f, {check: pat}).get(check, False) for f in (repo / "src").rglob('*.py')) for check, pat in {"uses_init": r"wandb\.init", "uses_log": r"wandb\.log", "uses_log_artifact": r"log_artifact"}.items()}, "scoring_type": "checklist"},
    "github_actions": {"label": "GitHub Actions CI check", "purpose": "CI/CD automation", "dimension": "CI/CD Pipeline", "run": lambda repo: (lambda paths: (lambda content: {"installs_deps": "pip install" in content or "conda install" in content, "runs_pytest": "pytest" in content})(" ".join(p.read_text(encoding='utf-8', errors='ignore') for p in paths)))(list((repo / ".github" / "workflows").glob("*.yml"))), "scoring_type": "checklist"},
    "fastapi_endpoint": {"label": "FastAPI Endpoint check", "purpose": "Inference API endpoint", "dimension": "Docker & Serving", "run": lambda repo: (lambda files: {"has_predict": any(re.search(r"@[\w_]+\.post\s*\(\s*[\"']/predict", f) for f in files), "has_health": any(re.search(r"@[\w_]+\.get\s*\(\s*[\"']/health", f) for f in files), "has_schema": any("BaseModel" in f for f in files)})(list(f.read_text(encoding='utf-8', errors='ignore') for f in repo.rglob('src/**/*.py'))), "scoring_type": "checklist"},
    "pipeline_modularity": {"label": "Pipeline Modularity check", "purpose": "Modular step-based structure", "dimension": "Pipeline Structure", "run": check_pipeline_modularity, "scoring_type": "ratio_to_10"},
    "readme_docs": {"label": "README Documentation check", "purpose": "Final project documentation", "dimension": "Documentation & Usability", "run": lambda repo: scan_file_for_patterns(repo / "README.md", {"mentions_mlflow": "mlflow", "mentions_wandb": "wandb|weights & biases", "mentions_docker": "docker", "mentions_api": "fastapi|/predict", "has_ci_badge": r"\!\[.*\]\(.*github\.com.*actions"}), "scoring_type": "checklist"},
    "dvc_usage": {"label": "DVC Usage check", "purpose": "Large data file management", "dimension": "Data Management", "run": lambda repo: (repo / ".dvc").is_dir() and any(repo.glob("**/*.dvc")), "scoring_type": "binary_10_0"},
    "render_deployment": {"label": "Render Deployment check", "purpose": "Deployment configuration", "dimension": "Deployment", "run": lambda repo: "services" in parse_yaml_safe(repo / "render.yaml"), "scoring_type": "binary_10_0"},
}


# --- Main Application Logic ---

def check_dependencies():
    """Checks for required command-line tools before running."""
    logging.info("Checking for required command-line tools...")
    required_tools = ["black", "isort", "flake8", "pylint",
                      "pytest", "radon", "detect-secrets", "pydocstyle"]
    missing_tools = [tool for tool in required_tools if not shutil.which(tool)]

    if missing_tools:
        logging.error(
            "Missing required tools: %s. Please install them and ensure they are in your PATH.", ", ".join(missing_tools))
        sys.exit(1)
    logging.info("All required tools found.")


def calculate_scores(raw_results: Dict[str, Dict[str, Any]]) -> List[List[Any]]:
    """
    Calculates normalized scores from raw test results.

    Args:
        raw_results: A nested dictionary of {repo_name: {test_key: raw_result}}.

    Returns:
        A list of rows, where each row contains the detailed results for one test on one repo.
    """
    logging.info("Calculating final scores...")
    all_rows = []
    repo_names = list(raw_results.keys())

    # Pre-calculate stats for relative scoring
    stats = {}
    for key, test in TESTS.items():
        scoring_type = test["scoring_type"]
        raw_values = [raw_results[repo].get(key) for repo in repo_names]

        if scoring_type in ("relative_inverted", "relative_direct"):
            numeric_values = [
                v for v in raw_values if isinstance(v, (int, float))]
            stats[key] = {"max": max(numeric_values) if numeric_values else 1}
        elif scoring_type == "custom_pytest":
            pytest_tuples = [v for v in raw_values if isinstance(
                v, tuple) and len(v) == 3]
            stats[key] = {"max_cov": max(
                (res[2] for res in pytest_tuples), default=100)}
        elif scoring_type == "custom_pydocstyle":
            pydoc_tuples = [v for v in raw_values if isinstance(
                v, tuple) and len(v) == 2]
            ratios = [(v / f) if f > 0 else 0 for v, f in pydoc_tuples]
            stats[key] = {"max_r": max(
                ratios, default=0), "min_r": min(ratios, default=0)}

    # Calculate score for each test and repo
    for repo_name in repo_names:
        for key, test in TESTS.items():
            raw_val = raw_results[repo_name].get(key)
            score, raw_display = 0.0, str(raw_val)
            scoring_type = test["scoring_type"]

            if isinstance(raw_val, str) and ("ERROR" in raw_val or "TIMEOUT" in raw_val or "COMMAND NOT FOUND" in raw_val):
                score, raw_display = 0.0, raw_val
            elif scoring_type == "absolute":
                score, raw_display = float(raw_val), f"{raw_val}/10"
            elif scoring_type == "binary_inverted":
                score, raw_display = (10.0, "OK (0 issues)") if raw_val == 0 else (
                    0.0, f"{raw_val} issue(s) found")
            elif scoring_type == "binary_10_0":
                score, raw_display = (10.0, "Compliant") if raw_val else (
                    0.0, "Non-compliant")
            elif scoring_type == "ratio_to_10":
                score, raw_display = float(
                    raw_val) * 10, f"{raw_val:.2f} ratio"
            elif scoring_type == "checklist":
                score, raw_display = calculate_checklist_score(raw_val)
            elif scoring_type == "relative_inverted":
                worst = stats.get(key, {}).get("max", 1) or 1
                score = 10 * (1 - raw_val / worst) if worst > 0 else 10.0
                raw_display = f"{raw_val} issues"
            elif scoring_type == "relative_direct":
                best = stats.get(key, {}).get("max", 1) or 1
                score = 10 * (raw_val / best) if best > 0 else 0
                raw_display = f"{raw_val:.1%} coverage"
            elif scoring_type == "custom_pytest":
                passed, total, cov = raw_val
                norm_cov = cov / \
                    (stats.get(key, {}).get("max_cov", 100) or 100)
                pass_ratio = passed / (total or 1)
                score = (norm_cov * 0.6 + pass_ratio * 0.4) * 10
                raw_display = f"{passed}/{total} passed, {cov}% cov"
            elif scoring_type == "custom_pydocstyle":
                v, f = raw_val
                ratio = (v / f) if f > 0 else 0
                max_r, min_r = stats[key]["max_r"], stats[key]["min_r"]
                score = 10.0 if (max_r - min_r) == 0 else max(0,
                                                              10 * (1 - (ratio - min_r) / (max_r - min_r)))
                raw_display = f"{v} violations / {f} files"
            elif scoring_type == "custom_tuple":
                score, raw_display = raw_val

            all_rows.append([
                repo_name, test["label"], test["purpose"], test["dimension"],
                raw_display, round(score, 2), COMMENTS_MAP.get(
                    test["label"], "")
            ])
    return all_rows


def main():
    """Main execution function to run the grading script."""
    check_dependencies()

    base_dir = Path(__file__).resolve().parent
    repos_root = base_dir / "repos_to_grade"

    if not repos_root.is_dir():
        logging.error(
            "Directory not found: '%s'. Please create it and clone student repos inside.", repos_root)
        sys.exit(1)

    all_repos = sorted([p for p in repos_root.iterdir()
                       if p.is_dir() and (p / ".git").exists()])
    if not all_repos:
        logging.error("No valid git repositories found in '%s'.", repos_root)
        sys.exit(1)

    # --- CLI Argument Parsing ---
    repos_to_grade = all_repos
    single_test_key = None

    args = sys.argv[1:]
    if "--repo" in args:
        try:
            repo_name = args[args.index("--repo") + 1]
            repos_to_grade = [r for r in all_repos if r.name == repo_name]
            if not repos_to_grade:
                logging.error("Repo '%s' not found in '%s'.",
                              repo_name, repos_root)
                sys.exit(1)
            logging.info("Running all tests on a single repo: %s", repo_name)
        except IndexError:
            logging.error("--repo flag used but no repo name was provided.")
            sys.exit(1)
    else:
        # Check for a single test key if not filtering by repo
        non_flag_args = [arg for arg in args if not arg.startswith('--')]
        if non_flag_args:
            single_test_key = non_flag_args[0]
            if single_test_key not in TESTS:
                logging.error("Test key '%s' not found. Available keys: %s",
                              single_test_key, ", ".join(TESTS.keys()))
                sys.exit(1)
            logging.info("Running single test on all repos: %s",
                         TESTS[single_test_key]['label'])

    # --- Test Execution ---
    raw_results = {repo.name: {} for repo in repos_to_grade}
    tests_to_run = {
        single_test_key: TESTS[single_test_key]} if single_test_key else TESTS

    logging.info("Collecting raw data for %d repo(s)...", len(repos_to_grade))
    for repo in repos_to_grade:
        logging.info("Processing %s...", repo.name)
        for key, test in tests_to_run.items():
            try:
                raw_results[repo.name][key] = test["run"](repo)
            except Exception as e:
                logging.error(
                    "An unhandled exception occurred during test '%s' for repo '%s': %s", key, repo.name, e)
                raw_results[repo.name][key] = f"ERROR: {e}"

    # --- Scoring and Reporting ---
    final_rows = calculate_scores(raw_results)
    df = pd.DataFrame(final_rows, columns=[
        "Repo name", "Test/ command", "Purpose", "Rubric dimension",
        "Raw", "Score (normalised 0-10)", "Comments"
    ])

    if single_test_key:
        print("\n--- Results ---")
        print(df.to_string(index=False))
    else:
        output_file = "grading_results.xlsx"
        try:
            df.to_excel(output_file, index=False, engine="openpyxl")
            logging.info(
                "✔ Grading complete. Report saved to '%s'", output_file)
        except Exception as e:
            logging.error("Failed to write Excel report: %s", e)


if __name__ == "__main__":
    main()
