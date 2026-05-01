# FE5211 Project Workflow

This project builds a quarterly multi-asset return simulation pipeline. It applies unsmoothing to private assets, fits univariate distributions and volatility models, then estimates 3-year 95% CVaR using ARMA-GARCH with a Vine Copula.

## Environment and Dependencies

This project uses `uv` for dependency management.

1. Install `uv` (`pip install uv` or via your system package manager).
2. Sync dependencies:
   ```bash
   uv sync
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

## Folder Structure

- `pyproject.toml`: Project configuration and dependencies.
- `code/`: Main analysis and modeling notebooks, plus visualization helpers.
- `data/`: Raw/intermediate/result data and data exploration notebooks.

## Main Analysis Entry Point

**`code/rvine_cvar_pipeline.ipynb`** is the primary pipeline notebook, including:

- ARMA(1,1)-GARCH(1,1) marginal fits for quarterly returns.
- R-vine copula fit on PIT residuals.
- 3-year quarterly path simulation and 95% CVaR estimation.
- Risk contribution and key visualization outputs.

## Recommended Run Order

1. **Fetch public market data**
   - Notebook: `code/get_data_yfinance.ipynb`
   - Goal: Download SPY/AGG from Yahoo Finance, merge with PE/NPI, and export quarterly returns.
   - Outputs: `data/quarterly_returns.csv`

2. **Build and preprocess combined dataset**
   - Notebook: `code/data_preprocess.ipynb`
   - Input: `data/quarterly_returns.csv`
   - Output: `data/processed_returns.csv`
   - Key logic:
     - Apply FGW-style unsmoothing to PE and NPI.
     - Keep SPY and AGG as observed returns (no unsmoothing).

3. **Univariate distribution and GARCH modeling**
   - Notebook: `code/univariate_distribution.ipynb`
   - Input: `data/processed_returns.csv`
   - Goal: Compare distributions via AIC/BIC and fit GARCH per asset with diagnostics.

4. **Main pipeline and risk evaluation**
   - Notebook: `code/rvine_cvar_pipeline.ipynb`
   - Goal: Copula fitting, path simulation, and 3-year 95% CVaR estimation.

## Visualization and Helper Scripts

- `code/visualizations.py`: Plotting helpers (Altair/Matplotlib) for GARCH bands, copula structure, simulated paths, and risk distributions.

## Outputs

- All final outputs (charts and risk metrics, including 12Q VaR/CVaR, copula dependence, and simulation fan charts) are produced in `code/rvine_cvar_pipeline.ipynb`.

## Notes

- The project uses quarterly frequency with a 12-quarter (3-year) risk window.
- For reproducibility, run notebooks in the recommended order with dependencies synced.