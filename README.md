# FE5211 Project Workflow

This project builds a quarterly multi-asset return simulation, applies unsmoothing to private assets, and runs univariate distribution and volatility modeling to estimate 3-Year 95% CVaR with ARMA-GARCH & Vine Copula.

## Setup

This project is developed on `Python 3.12.12` and uses `uv` for dependency management.

1. Ensure you have `uv` installed (`pip install uv` or via your system's package manager).
2. Sync the project dependencies:
   ```bash
   uv sync
   ```
3. Activate the virtual environment provided by `uv`:
   ```bash
   source .venv/bin/activate
   ```

## Folder Structure

- `pyproject.toml`: The configuration file containing project metadata and dependency requirements.
- `code/`: Analysis notebooks (data pipeline, modeling, and riskfolio exploration) and helper scripts.
- `data/`: Intermediate and final CSV files, as well as initial data notebooks.

## Main Analysis File ⭐️

**`code/rvine_cvar_pipeline.ipynb`** is the file containing the **main modeling, analysis, and results visualization**. 
It executes the primary pipeline which includes:
- Fitting ARMA(1,1)-GARCH(1,1) marginals to the quarterly returns.
- Fitting a vine copula to the PIT residuals.
- Simulating 3-year quarterly paths and estimating 95% CVaR.
- Visualizing results and risk contributions.

## Recommended Run Order & Other Files

1. **Private asset data exploration**
   Notebook: `data/Private_Asset_Returns.ipynb`
   Purpose: Raw NCREIF and PE data records and basic explorations.

2. **Get public market quarterly returns**  
   Notebook: `code/get_data_yfinance.ipynb`  
   Purpose: Download market data (SPY, AGG) from Yahoo Finance, combine with PE and NPI data, and export quarterly return CSV (`data/quarterly_return_SPY_AGG.csv` and `data/quarterly_returns.csv`).

3. **Build and preprocess combined return dataset**  
   Notebook: `code/data_preprocess.ipynb`  
   Input: Quarterly return data including SPY, AGG, PE, NPI.
   Output: `data/processed_returns.csv`  
   Key logic:
   - Apply unsmoothing (FGW-style) to **PE** and **NPI**.
   - Keep **SPY** and **AGG** as raw observed returns (no unsmoothing).

4. **Univariate modeling + GARCH per asset**  
   Notebook: `code/univariate_distribution.ipynb`  
   Input: `data/processed_returns.csv`  
   Purpose:
   - Fit candidate univariate distributions asset-by-asset.
   - Compare models with AIC/BIC.
   - Run GARCH modeling per asset and inspect diagnostics.

### Helper Scripts

- **`code/visualizations.py`**: A Python module containing plotting and visualization functions (using Altair, Matplotlib) utilized by the main pipeline.
