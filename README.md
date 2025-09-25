# Portfolio_Optimisation — Notebook

**File:** `Portfolio_Optimisation.ipynb`
**Purpose:** Oil-price / time-series (or tabular) prediction using Support Vector Regression (SVR) whose hyperparameters are tuned/optimized with a Multi‑Objective Particle Swarm Optimization (MOPSO) algorithm. This Colab notebook demonstrates the full workflow from data loading and preprocessing → feature engineering → MOPSO hyperparameter search → model training → evaluation and visualization.

---

## Table of contents

1. [Project overview](#project-overview)
2. [Notebook structure](#notebook-structure)
3. [Getting started (Colab & local)](#getting-started-colab--local)
4. [Dependencies](#dependencies)
5. [Dataset](#dataset)
6. [How the code works (high level)](#how-the-code-works-high-level)
7. [Reproduce results](#reproduce-results)
8. [Output / What to expect](#output--what-to-expect)
9. [Troubleshooting](#troubleshooting)
10. [Contributing / Notes](#contributing--notes)


---

## Project overview

This notebook implements Support Vector Regression (SVR) for regression tasks and uses a Multi‑Objective Particle Swarm Optimization (MOPSO) algorithm to find optimal SVR hyperparameters (for example, `C`, `epsilon`, and kernel parameters). Typical use-cases: oil price forecasting, generic time-series regression, or any continuous target prediction where SVR is appropriate.

Goals:

* Demonstrate MOPSO for hyperparameter search across multiple objectives (e.g., minimize validation error while minimizing model complexity or training time).
* Provide reusable, well-commented Colab code so you can reproduce and adapt the pipeline.

---

## Notebook structure

The notebook is organized into the following logical sections (top to bottom):

1. **Imports & Environment setup** — install required libs (if running in a fresh Colab environment) and import modules.
2. **Load dataset** — local CSV / Google Drive / remote link. (Cell(s) include examples for mounting Drive.)
3. **Exploratory Data Analysis (EDA)** — quick statistics and visualizations.
4. **Preprocessing & Feature Engineering** — scaling, lag features (for time series), train/val/test split, and target transformation if needed.
5. **SVR baseline** — train baseline SVR and evaluate.
6. **MOPSO implementation** — MOPSO algorithm implementation or usage of helper functions to perform multi-objective optimization.
7. **Hyperparameter search** — run MOPSO to discover Pareto-optimal hyperparameter sets.
8. **Train final models** — retrain with best hyperparameters and evaluate on test set.
9. **Results & Visualizations** — metrics table, residual plots, prediction vs actual plots.
10. **Save/Export models and metrics** — save models to disk / Drive, export metrics as CSV.

---

## Getting started (Colab & local)

### Run in Google Colab (recommended)

1. Upload the notebook `SVR_MOPSO.ipynb` to your GitHub repo or open directly in Colab.
2. If using Drive for datasets/models, run the Drive mount cell: `from google.colab import drive; drive.mount('/content/drive')`.
3. Run cells top-to-bottom. If dependencies are missing, run the provided pip install cell (see Dependencies).

### Run locally

1. Clone this repo.
2. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Open the notebook with Jupyter and execute cells.

---

## Dependencies

Minimal recommended packages (additions can be used in the notebook):

* Python 3.8+
* numpy
* pandas
* scikit-learn
* matplotlib
* seaborn (optional, used for EDA)
* tqdm
* joblib (for saving models)

**Example `requirements.txt` entries:**

```
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
joblib
```

If you need a single pip command (Colab):

```python
!pip install numpy pandas scikit-learn matplotlib seaborn tqdm joblib
```

---

## Dataset

This project expects a CSV (or similar) dataset containing features and a continuous target column. Typical layout for a time-series (oil price) dataset:

* `date` — timestamp (optional)
* `target` — e.g., `price` (the column used for regression)
* other explanatory features (macroeconomic indicators, lagged prices, technical indicators, etc.)

**Where to put the dataset:**

* Colab: upload to session storage or mount Google Drive and point the notebook to the file path (recommended for larger files).
* Local: place in a `data/` directory at the repository root and update the notebook path.

If no dataset is included, the notebook contains an example synthetic dataset generator / example CSV loader so you can test pipeline behavior.

---

## How the code works (high level)

1. **Preprocessing:** scaling features using `StandardScaler` (or `MinMaxScaler`) and creating lag features for time-series.
2. **SVR baseline:** a baseline SVR with default hyperparameters is trained to provide a reference performance.
3. **Multi-objective MOPSO:** the algorithm treats hyperparameter tuning as a search in parameter space. Each particle represents a candidate hyperparameter vector. Objectives can include:

   * validation RMSE (to minimize)
   * model complexity (e.g., number of support vectors or large `C`) or training time (to minimize)
4. **Pareto front:** MOPSO identifies a set of non-dominated solutions; you can pick one based on a trade-off or retrain ensemble across Pareto-optimal solutions.
5. **Final evaluation:** train final SVR(s) with chosen hyperparameters and evaluate on test set using RMSE / MAE / R².

---

## Reproduce results

1. Make sure the dataset path in the notebook is correct.
2. Run all cells (Colab: `Runtime -> Run all`).
3. The MOPSO hyperparameter search may take time — for quick tests reduce `n_particles` and `n_iterations`.

Suggested quick-debug settings for development:

* `n_particles = 10`
* `n_iterations = 20`

When you want final results, increase `n_particles` and `n_iterations` for a more thorough search.

---

## Output / What to expect

* Metric tables: training/validation/test RMSE, MAE, R².
* Plots: prediction vs actual, residuals, Pareto front visualization for objective trade-offs.
* Saved models: serialized `joblib` or `pickle` files in `outputs/` or mounted Drive.

---

## Troubleshooting

* **"Module not found"** — run the pip install cell in Colab or install requirements locally.
* **Long runtime** — reduce the number of particles/iterations for debugging; use smaller datasets.
* **Reproducibility** — set random seeds where applicable: `numpy.random.seed()` and scikit-learn `random_state`.

---

## Contributing / Notes

* This notebook is intended as an educational, reproducible example. Feel free to open issues / PRs if you want features added (e.g., support for different optimizers, more objectives, or different regression models).
* If you adapt the notebook, consider converting heavy computations into standalone Python scripts or modules for cleaner experimentation.


**Author / Contact:** soum singh (adapt as desired).
If you want me to customize the README content for a different audience (technical report, short summary, or a one-page README for GitHub), tell me what tone and length you prefer and I will update it.
