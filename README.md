# FE5211 Project Workflow by now

This project builds a quarterly multi-asset return dataset, applies unsmoothing to private assets, and runs univariate distribution and volatility modeling.

## Folder Structure

- `code/`: analysis notebooks (data pipeline, modeling, and riskfolio exploration)
- `data/`: intermediate and final CSV files

## Recommended Run Order

1. **Get public market quarterly returns**  
   Notebook: `code/get_data_yfinance.ipynb`  
   Purpose: download market data (SPY, AGG) from Yahoo Finance, combine with PE and NPI data and export quarterly return CSV.

2. **Build and preprocess combined return dataset**  
   Notebook: `code/data_preprocess.ipynb`  
   Input: quarterly return data including SPY, AGG, PE, NPI  
   Output: `data/processed_returns.csv`  
   Key logic:
   - Apply unsmoothing (FGW-style) to **PE** and **NPI**.
   - Keep **SPY** and **AGG** as raw observed returns (no unsmoothing).

3. **Univariate modeling + GARCH per asset**  
   Notebook: `code/univariate_distribution.ipynb`  
   Input: `data/processed_returns.csv`  
   Purpose:
   - Fit candidate univariate distributions asset-by-asset.
   - Compare models with AIC/BIC.
   - Run GARCH modeling per asset and inspect diagnostics.

4. **Riskfolio CVaR exploration**  
   Notebook: `code/riskfolio_CVar.ipynb`  
   Purpose: exploratory use of `riskfolio-lib` for CVaR-related functions (e.g., CVaR, risk contribution, risk margin).

## Riskfolio References

You can further explore the Riskfolio library for additional portfolio/risk analytics:

- Installation / docs: https://riskfolio-lib.readthedocs.io/en/latest/install.html
- Risk functions: https://riskfolio-lib.readthedocs.io/en/latest/risk.html
- Plot functions: https://riskfolio-lib.readthedocs.io/en/latest/plot.html

