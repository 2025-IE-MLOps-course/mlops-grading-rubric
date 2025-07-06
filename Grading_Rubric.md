# Automated Grading Rubric Explained

This document provides a clear and transparent overview of how your project is automatically graded.  
Each test assesses a specific, real-world MLOps skill. The goal is to reward adherence to best practices in a fair and consistent manner.

---

## Scoring Philosophy

- **Absolute Scoring**: Used where there is a clear, objective standard (e.g., Pylint score, cyclomatic complexity). Your score depends only on your own code's quality.
- **Relative Scoring**: Used for tests like code formatting and linting issues. Your score is compared to the class's best and worst results:
    - **A perfect submission always receives 10 points**
    - Your score is proportional to your issue count compared to the repo with the most issues
    - Minor deviations will not significantly impact your score
- **Checklists**: Used for integration or multi-part checks. You receive points for each required item present.
- **Binary**: Used for critical checks such as security or DVC usage. The test is either passed (10 points) or failed (0 points).

All automated tests are fully objective—no subjective scoring except where explicitly noted for manual/qualitative items.

---

## Production-Ready Code

This section evaluates the quality, robustness, and maintainability of your Python code.

### 1. `black --check .`
- **Purpose**: Ensures code is formatted according to the Black standard, a widely adopted industry convention.
- **Rubric Dimension**: Code Quality & Efficiency
- **How it Works**: Counts the number of files Black would reformat.
- **Scoring**: **Relative, Inverted**. 10 for zero issues; otherwise, score is normalized relative to the worst result.

### 2. `isort --check-only .`
- **Purpose**: Enforces standard practices for ordering imports, improving readability.
- **Rubric Dimension**: Code Quality & Efficiency
- **How it Works**: Counts the number of files with unsorted imports.
- **Scoring**: **Relative, Inverted**. Same logic as Black.

### 3. `flake8 .`
- **Purpose**: General-purpose linter for style violations and possible bugs (PEP 8).
- **Rubric Dimension**: Code Quality & Efficiency
- **How it Works**: Counts total Flake8 violations.
- **Scoring**: **Relative, Inverted**. Same as above.

### 4. `flake8 --select=LOG .`
- **Purpose**: Checks for best practices in logging.
- **Rubric Dimension**: Logging & Monitoring
- **How it Works**: Counts logging-related Flake8 violations.
- **Scoring**: **Relative, Inverted**.

### 5. `pylint src --score y`
- **Purpose**: Advanced static analysis, producing an overall quality score.
- **Rubric Dimension**: Code Quality & Documentation
- **How it Works**: Runs pylint on your `src` folder, outputs a score out of 10.
- **Scoring**: **Absolute**. Score directly reflects Pylint rating.

### 6. `pytest --cov=src`
- **Purpose**: Measures the quality and coverage of your unit tests.
- **Rubric Dimension**: Testing & Coverage
- **How it Works**:
    - **Pass Rate**: Percentage of unit tests that pass
    - **Coverage**: Percentage of code exercised by the tests
- **Scoring**: **Weighted custom**:  
    `(0.6 × Normalized Coverage) + (0.4 × Pass Rate)`  
    Coverage is normalized to the class's highest coverage, ensuring fairness.

### 7. `radon cc -s -a src`
- **Purpose**: Measures cyclomatic complexity, indicating code maintainability.
- **Rubric Dimension**: Modularization & Code Quality
- **How it Works**: Assigns a letter grade (A–F) based on average function complexity.
- **Scoring**: **Absolute**. A=10, B=8, C=6, D=4, E=2, F=0.

### 8. `detect-secrets scan`
- **Purpose**: Ensures no passwords, API keys, or secrets are committed to code.
- **Rubric Dimension**: Security
- **How it Works**: Scans for secret patterns in code and config files.
- **Scoring**: **Binary**. 10 if clean, 0 if any secret found.

### 9. `pydocstyle src`
- **Purpose**: Checks for required docstrings and style in all functions/classes.
- **Rubric Dimension**: Documentation & Clarity
- **How it Works**: Calculates ratio of violations to Python files in `src`.
- **Scoring**: **Relative, Inverted**. Lower violation ratio = higher score, compared to class.

### 10. Config usage scan
- **Purpose**: Ensures no hard-coded file paths; promotes config-driven code.
- **Rubric Dimension**: Config Mgmt
- **How it Works**: Scans Python files for absolute file paths.
- **Scoring**: **Binary**. 10 if none found, 0 if any are present.

### 11. Error-handling scan
- **Purpose**: Measures robustness in handling errors (especially I/O).
- **Rubric Dimension**: Error Handling & Validation
- **How it Works**: Checks what percentage of "risky" functions use try/except.
- **Scoring**: **Relative, Direct**. Your percent is normalized to the class's best.

---

## MLOps Integration & Automation

This section evaluates your adoption of standard MLOps tools and practices.

### 12. MLflow Project check
- **Purpose**: Assesses orchestration of your pipeline using MLflow.
- **Rubric Dimension**: MLflow Integration
- **How it Works**: Detects most robust orchestration pattern:
    - **10**: Python main.py with `mlflow.run()` or MLproject file integrated into CI
    - **8**: MLproject file present, not integrated into CI
- **Scoring**: **Absolute**. Highest pattern detected.

### 13. Hydra Config check
- **Purpose**: Checks for structured configuration with Hydra.
- **Rubric Dimension**: Hydra Config Management
- **How it Works**: Three-point checklist:
    1. `config.yaml` exists and is valid
    2. Contains a `main` section
    3. `main.steps` is a non-empty list
- **Scoring**: (Checks passed / 3) × 10

### 14. W&B Integration check
- **Purpose**: Checks experiment/artifact tracking with Weights & Biases.
- **Rubric Dimension**: W&B Tracking
- **How it Works**: Three-point checklist for `wandb.init()`, `wandb.log()`, and `log_artifact()`
- **Scoring**: (Checks passed / 3) × 10

### 15. GitHub Actions CI check
- **Purpose**: Ensures basic CI pipeline is in place.
- **Rubric Dimension**: CI/CD Pipeline
- **How it Works**: Two-point checklist:
    1. Dependency installation step
    2. Test execution step
- **Scoring**: (Checks passed / 2) × 10

### 16. Dockerfile check
- **Purpose**: Checks Dockerfile existence and best practices.
- **Rubric Dimension**: Docker & Serving
- **How it Works**: Two-point checklist (valid Dockerfile, best practices)
- **Scoring**: (Checks passed / 2) × 10

### 17. FastAPI Endpoint check
- **Purpose**: Ensures a functional inference API.
- **Rubric Dimension**: Docker & Serving
- **How it Works**: Three-point checklist:
    1. POST endpoint at `/predict`
    2. GET health endpoint (e.g., `/health`)
    3. Defined request schema (e.g., Pydantic BaseModel)
- **Scoring**: (Checks passed / 3) × 10

### 18. Pipeline Modularity check
- **Purpose**: Assesses alignment between declared steps and code modules.
- **Rubric Dimension**: Pipeline Structure
- **How it Works**: Compares pipeline steps in config to modules/scripts in `src` (uses fuzzy matching).
- **Scoring**: (Matched steps / Total steps) × 10

### 19. README Documentation check
- **Purpose**: Ensures documentation covers key MLOps aspects.
- **Rubric Dimension**: Documentation & Usability
- **How it Works**: Five-point checklist:
    1. MLflow usage
    2. W&B usage
    3. Docker commands
    4. API endpoint usage
    5. CI status badge
- **Scoring**: (Checks passed / 5) × 10

### 20. DVC Usage check (Extra)
- **Purpose**: Assesses use of DVC for large data file management.
- **Rubric Dimension**: Data Management
- **How it Works**: Checks for .dvc directory and at least one pointer file.
- **Scoring**: 10 if compliant, 0 otherwise (binary).

### 21. Render Deployment check
- **Purpose**: Checks for a valid deployment configuration for Render or similar platforms.
- **Rubric Dimension**: Deployment
- **How it Works**: Looks for `render.yaml` and ensures it declares a web service.
- **Scoring**: 10 if present and valid, 0 otherwise (binary).

---

## Manual & Qualitative Assessment

Some items are graded by human review:

- **Final (live) presentation**: Based on clarity, depth, and communication of your business case. See the separate rubric.
- **README Qualitative**: Based on documentation quality, usability, and clarity. See the separate rubric.

---

## Appeals & Questions

All tests (except manual reviews) are fully automated and reproducible.  
If any score seems unclear or incorrect, you may request your raw results and a detailed explanation.

---

*All grading criteria, logic, and scoring methods are published for full transparency and fairness.*
