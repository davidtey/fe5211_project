from __future__ import annotations

from itertools import combinations, product
from typing import Any, Mapping, Sequence

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import gaussian_kde, kendalltau
from statsmodels.tsa.stattools import acf

import pyvinecopulib as pv

alt.data_transformers.disable_max_rows()

DEFAULT_PORTFOLIO_WEIGHTS: dict[str, float] = {
    "SPY": 0.40,
    "AGG": 0.20,
    "PE": 0.25,
    "NPI": 0.15,
}


def _as_pandas(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pd.Series):
        return data.to_frame(name=data.name or "value").reset_index()
    if isinstance(data, pl.DataFrame):
        return data.to_pandas()
    if isinstance(data, np.ndarray):
        return pd.DataFrame(data)
    raise TypeError(f"Unsupported tabular input type: {type(data)!r}")


def _to_series(values: Any, name: str = "value") -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(float).dropna()
    if isinstance(values, (pd.DataFrame, pl.DataFrame)):
        frame = _as_pandas(values)
        if frame.shape[1] != 1:
            raise ValueError("Expected a one-column table or a Series.")
        return frame.iloc[:, 0].astype(float).dropna()
    if isinstance(values, np.ndarray):
        return pd.Series(values.astype(float), name=name).dropna()
    return pd.Series(np.asarray(values, dtype=float), name=name).dropna()


def _asset_names(marginals: Sequence[Mapping[str, Any]], asset_names: Sequence[str] | None = None) -> list[str]:
    if asset_names is not None:
        return list(asset_names)
    return [str(m.get("asset", f"Asset {idx + 1}")) for idx, m in enumerate(marginals)]


def _portfolio_weights(asset_names: Sequence[str], portfolio_weights: Mapping[str, float] | None = None) -> np.ndarray:
    weights_map = dict(DEFAULT_PORTFOLIO_WEIGHTS)
    if portfolio_weights is not None:
        weights_map.update(portfolio_weights)
    return np.asarray([float(weights_map.get(name, 0.0)) for name in asset_names], dtype=float)


def _quarter_labels(horizon_quarters: int) -> list[str]:
    labels = []
    year = 2026
    quarter = 3
    for _ in range(horizon_quarters):
        labels.append(f"{year} Q{quarter}")
        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1
    return labels


def _arma_residuals_from_fit(values: np.ndarray, mu: float, phi: float, theta: float) -> np.ndarray:
    eps = np.zeros_like(values, dtype=float)
    for idx in range(1, len(values)):
        eps[idx] = values[idx] - mu - phi * values[idx - 1] - theta * eps[idx - 1]
    return eps


def _garch_variance_path(resid: np.ndarray, omega: float, alpha: float, beta: float, sigma2_lr: float) -> np.ndarray:
    sigma2 = np.empty(len(resid), dtype=float)
    sigma2[0] = sigma2_lr
    for idx in range(1, len(resid)):
        sigma2[idx] = omega + alpha * resid[idx - 1] ** 2 + beta * sigma2[idx - 1]
        sigma2[idx] = max(sigma2[idx], 1e-12)
    return sigma2


def _terminal_asset_returns(simulation: Mapping[str, Any], asset_names: Sequence[str]) -> np.ndarray:
    returns = np.asarray(simulation["returns"], dtype=float)
    return np.prod(1.0 + returns, axis=1) - 1.0


def _simulate_single_path(
    marginals: Sequence[Mapping[str, Any]],
    copula: Any,
    horizon_quarters: int = 12,
    seed: int = 7,
    portfolio_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    asset_names = _asset_names(marginals)
    weights = _portfolio_weights(asset_names, portfolio_weights)
    d = len(marginals)

    u_path = []
    for quarter in range(horizon_quarters):
        u = np.asarray(copula.simulate(1, seeds=[seed + quarter]), dtype=float)[0]
        u_path.append(u)
    u_path = np.asarray(u_path, dtype=float)

    asset_log_returns = np.zeros((horizon_quarters, d), dtype=float)
    asset_sigma = np.zeros((horizon_quarters, d), dtype=float)
    asset_mean = np.zeros((horizon_quarters, d), dtype=float)
    asset_simple = np.zeros((horizon_quarters, d), dtype=float)
    prev_return = np.asarray([float(m["last_return"]) for m in marginals], dtype=float)
    prev_eps = np.asarray([float(m["last_eps"]) for m in marginals], dtype=float)
    prev_sigma = np.asarray([float(max(m["last_sigma"], 1e-12)) for m in marginals], dtype=float)

    for quarter in range(horizon_quarters):
        for idx, fit in enumerate(marginals):
            z = float(fit["ppf"](u_path[quarter, idx]))
            sigma2 = float(fit["omega"]) + float(fit["alpha"]) * prev_eps[idx] ** 2 + float(fit["beta"]) * prev_sigma[idx] ** 2
            sigma = float(np.sqrt(max(sigma2, 1e-12)))
            mean = float(fit["simulation_const"]) + float(fit["phi"]) * prev_return[idx] + float(fit["theta"]) * prev_eps[idx]
            eps = sigma * z
            log_return = mean + eps

            asset_log_returns[quarter, idx] = log_return
            asset_sigma[quarter, idx] = sigma
            asset_mean[quarter, idx] = mean
            asset_simple[quarter, idx] = np.expm1(log_return)

            prev_return[idx] = log_return
            prev_eps[idx] = eps
            prev_sigma[idx] = sigma

    portfolio_log = asset_log_returns @ weights
    portfolio_sigma = np.sqrt(np.sum((asset_sigma * weights.reshape(1, -1)) ** 2, axis=1))
    portfolio_simple = np.expm1(portfolio_log)

    return {
        "asset_names": asset_names,
        "weights": weights,
        "u_path": u_path,
        "asset_log_returns": asset_log_returns,
        "asset_sigma": asset_sigma,
        "asset_mean": asset_mean,
        "asset_simple_returns": asset_simple,
        "portfolio_log_returns": portfolio_log,
        "portfolio_simple_returns": portfolio_simple,
        "portfolio_sigma": portfolio_sigma,
        "quarter_labels": _quarter_labels(horizon_quarters),
    }


def plot_single_path_return_volatility_overlay(
    marginals: Sequence[Mapping[str, Any]],
    copula: Any,
    target: str = "portfolio",
    asset_name: str | None = None,
    horizon_quarters: int = 12,
    seed: int = 7,
    portfolio_weights: Mapping[str, float] | None = None,
) -> alt.Chart:
    """Expected input: `marginals` from `fit_arma_garch_marginal(...)` and `copula` from `fit_rvine_copula(...)`.

    The `target` can be `portfolio` or `asset`. For `asset`, pass `asset_name` from the existing notebook's `asset_cols`.
    The chart is drawn in simulated log-return space so the GARCH volatility band is symmetric.
    """

    path = _simulate_single_path(marginals, copula, horizon_quarters=horizon_quarters, seed=seed, portfolio_weights=portfolio_weights)
    quarter = np.arange(1, horizon_quarters + 1)
    quarter_labels = path["quarter_labels"]
    asset_names = path["asset_names"]

    if target not in {"portfolio", "asset"}:
        raise ValueError("target must be either 'portfolio' or 'asset'.")

    if target == "portfolio":
        log_ret = path["portfolio_log_returns"]
        sigma = path["portfolio_sigma"]
        val_path = np.cumprod(1 + path["portfolio_simple_returns"])
        val_lower = val_path*(1 - np.expm1(sigma))
        val_upper = val_path*(1 + np.expm1(sigma))
        
        frame = pd.DataFrame(
            {
                "quarter": quarter,
                "label": quarter_labels,
                "lower": val_lower,
                "upper": val_upper,
                "path": val_path,
            }
        )
        title = "Simulated 12-Quarter Portfolio Value with ±1σ Band"
        color = "#1f77b4"
    else:
        if asset_name is None:
            raise ValueError("asset_name is required when target='asset'.")
        if asset_name not in asset_names:
            raise ValueError(f"asset_name must be one of {asset_names!r}.")
        idx = asset_names.index(asset_name)
        log_ret = path["asset_log_returns"][:, idx]
        mean_ret = path["asset_mean"][:, idx]
        sigma = path["asset_sigma"][:, idx]
        
        val_path = np.cumprod(1 + path["asset_simple_returns"][:, idx])
        val_lower = val_path*(1 - np.expm1(sigma))
        val_upper = val_path*(1 + np.expm1(sigma))
        
        frame = pd.DataFrame(
            {
                "quarter": quarter,
                "label": quarter_labels,
                "lower": val_lower,
                "upper": val_upper,
                "path": val_path,
            }
        )
        title = f"Simulated 12-Quarter {asset_name} Value with ±1σ Band"
        color = "#2a9d8f"

    band = (
        alt.Chart(frame)
        .mark_area(opacity=0.2, color=color)
        .encode(x=alt.X("label:O", sort=quarter_labels, title="Time"), y="lower:Q", y2="upper:Q")
    )
    path_line = (
        alt.Chart(frame)
        .mark_line(color="#111111", strokeWidth=1.6)
        .encode(x="label:O", y=alt.Y("path:Q", title="Simulated Cumulative Value"))
    )
    points = alt.Chart(frame).mark_point(color="#111111", size=50).encode(x="label:O", y="path:Q", tooltip=["label", alt.Tooltip("path:Q", format=".4f")])

    chart = (band + path_line + points).properties(title=title, width=820, height=320)
    return chart


def plot_acf_squared_residuals(
    marginal: Mapping[str, Any],
    lags: int = 12,
) -> alt.Chart:
    """Expected input: one element from the notebook's `marginals` list.

    The function recomputes the ARMA residuals from the stored GARCH parameters, then compares the ACF of raw squared residuals
    against standardized squared residuals after GARCH filtering.
    """

    observed = _to_series(marginal["observed"], name=str(marginal.get("asset", "series"))).to_numpy(dtype=float)
    fit = marginal["garch_fit"]
    eps = _arma_residuals_from_fit(observed, float(fit.mu), float(fit.phi), float(fit.theta))
    sigma2 = _garch_variance_path(eps, float(fit.omega), float(fit.alpha), float(fit.beta), float(fit.sigma2_lr))
    std_resid = eps / np.sqrt(np.maximum(sigma2, 1e-12))

    raw_acf = acf(np.square(eps), nlags=lags, fft=False)
    filt_acf = acf(np.square(std_resid), nlags=lags, fft=False)
    lag_index = np.arange(0, lags + 1)
    frame = pd.DataFrame(
        {
            "lag": np.tile(lag_index, 2),
            "acf": np.concatenate([raw_acf, filt_acf]),
            "series": ["Raw squared residuals"] * (lags + 1) + ["GARCH-filtered squared residuals"] * (lags + 1),
        }
    )

    bars = alt.Chart(frame).mark_bar().encode(
        x=alt.X("lag:O", title="Lag"),
        y=alt.Y("acf:Q", title="Autocorrelation"),
        color=alt.Color("series:N", title="Series", scale=alt.Scale(range=["#c0392b", "#2a9d8f"])),
        xOffset=alt.XOffset("series:N"),
        tooltip=["series", "lag", alt.Tooltip("acf:Q", format=".4f")],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="#444444").encode(y="y:Q")
    chart = (bars + zero).properties(title=f"ACF of Squared Residuals - {marginal.get('asset', 'Series')}", width=760, height=300)
    return chart


def plot_dependence_matrix_heatmap(
    marginals: Sequence[Mapping[str, Any]],
    copula: Any,
    asset_names: Sequence[str] | None = None,
    n_sim: int = 10_000,
) -> alt.Chart:
    """Expected input: `marginals` and `copula` from the notebook.

    The left matrix is the empirical Kendall's tau from the PIT residuals; the right matrix is model-implied Kendall's tau from
    a simulation draw from the fitted vine copula.
    """

    asset_names = _asset_names(marginals, asset_names)
    u_obs = np.column_stack([np.asarray(m["pit"], dtype=float) for m in marginals])
    u_sim = np.asarray(copula.simulate(n_sim, seeds=[7]), dtype=float)

    records = []
    for source, u in (("Empirical", u_obs), ("Model-implied", u_sim)):
        for i, j in combinations(range(len(asset_names)), 2):
            tau, _ = kendalltau(u[:, i], u[:, j])
            records.append({"source": source, "row": asset_names[i], "col": asset_names[j], "tau": float(tau)})
            records.append({"source": source, "row": asset_names[j], "col": asset_names[i], "tau": float(tau)})
        for name in asset_names:
            records.append({"source": source, "row": name, "col": name, "tau": 1.0})

    frame = pd.DataFrame(records)
    base = (
        alt.Chart(frame)
        .mark_rect()
        .encode(
            x=alt.X("col:N", title=None, sort=list(asset_names)),
            y=alt.Y("row:N", title=None, sort=list(asset_names)),
            color=alt.Color("tau:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
            tooltip=["source", "row", "col", alt.Tooltip("tau:Q", format=".3f")],
        )
    )
    labels = alt.Chart(frame).mark_text(fontSize=11, fontWeight=600).encode(x=alt.X("col:N", sort=list(asset_names)), y=alt.Y("row:N", sort=list(asset_names)), text=alt.Text("tau:Q", format=".2f"), color=alt.value("#111111"))
    chart = alt.hconcat(
        (base + labels).transform_filter(alt.datum.source == "Empirical").properties(title="Empirical Kendall's τ", width=240, height=240),
        (base + labels).transform_filter(alt.datum.source == "Model-implied").properties(title="Model-implied Kendall's τ", width=240, height=240),
        spacing=25,
    ).resolve_scale(color="shared")
    return chart


def plot_pit_scatter_matrix(
    marginals: Sequence[Mapping[str, Any]],
    copula: Any,
    asset_names: Sequence[str] | None = None,
    n_sim: int = 8_000,
    max_points: int = 2_000, # Reduced slightly for better browser performance
) -> alt.Chart:
    
    asset_names = _asset_names(marginals, asset_names)
    u_obs = np.column_stack([np.asarray(m["pit"], dtype=float) for m in marginals])
    u_sim = np.asarray(copula.simulate(n_sim, seeds=[11]), dtype=float)

    data_list = []
    
    for source, u in (("Observed", u_obs), ("Simulated", u_sim)):
        if len(u) > max_points:
            rng = np.random.default_rng(7)
            u = rng.choice(u, size=max_points, replace=False)
        
        # Use product to get every combination (i, j) including i == j
        for i, j in product(range(len(asset_names)), repeat=2):
            data_list.append({
                "source": source,
                "x_asset": asset_names[i],
                "y_asset": asset_names[j],
                "u_x": u[:, i],
                "u_y": u[:, j],
            })

    # explode and pivot as before
    frame = pd.DataFrame(data_list).explode(["u_x", "u_y"])
    
    # Ensure asset_names is a list for the sort argument
    grid_order = list(asset_names)

    chart = (
        alt.Chart(frame)
        .mark_point(size=12, opacity=0.22)
        .encode(
            x=alt.X("u_x:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("u_y:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("source:N", scale=alt.Scale(domain=["Observed", "Simulated"], range=["#2a9d8f", "#c0392b"])),
        )
        .properties(width=140, height=140)
        .facet(
            # Use the same order for both to keep the matrix aligned
            row=alt.Row("y_asset:N", sort=grid_order),
            column=alt.Column("x_asset:N", sort=grid_order),
        )
    )
    
    return chart


def plot_historical_return_time_series_with_garch_band(
    series: pd.Series,
    marginal: Mapping[str, Any],
) -> alt.Chart:
    """Expected input: one historical return series from `model_df[column]` and the matching entry in `marginals`.

    The chart uses the notebook's modeled log-return space and overlays the in-sample fitted conditional volatility band.
    """

    values = _to_series(series, name=series.name or str(marginal.get("asset", "series"))).to_numpy(dtype=float)
    index = pd.Index(series.index if isinstance(series, pd.Series) else np.arange(len(values)), name="date")
    fit = marginal["garch_fit"]
    eps = _arma_residuals_from_fit(values, float(fit.mu), float(fit.phi), float(fit.theta))
    sigma2 = _garch_variance_path(eps, float(fit.omega), float(fit.alpha), float(fit.beta), float(fit.sigma2_lr))
    sigma = np.sqrt(np.maximum(sigma2, 1e-12))
    mean = float(fit.mu) + float(fit.phi) * np.r_[values[0], values[:-1]] + float(fit.theta) * np.r_[0.0, eps[:-1]]

    frame = pd.DataFrame({"date": index, "observed": values, "mean": mean, "lower": mean - sigma, "upper": mean + sigma})
    band = alt.Chart(frame).mark_area(opacity=0.18, color="#2a9d8f").encode(x=alt.X("date:T", title="Date"), y="lower:Q", y2="upper:Q")
    mean_line = alt.Chart(frame).mark_line(color="#1f77b4", strokeWidth=2).encode(x="date:T", y=alt.Y("mean:Q", title="Log return"))
    obs_line = alt.Chart(frame).mark_line(color="#111111", strokeWidth=1.4).encode(x="date:T", y="observed:Q", tooltip=[alt.Tooltip("date:T"), alt.Tooltip("observed:Q", format=".4f")])
    chart = (band + mean_line + obs_line).properties(title=f"Historical Return Series with GARCH σ Band - {series.name}", width=850, height=320)
    return chart


def plot_bootstrap_cvar_ci(
    returns: Any,
    alpha: float = 0.05,
    n_boot: int = 200,
    ci: float = 0.95,
    seed: int = 7,
) -> alt.Chart:
    """Expected input: a one-dimensional return series, for example `simulation['compounded_return']` or the historical portfolio return.

    The estimate remains in the same return units as the input series.
    """

    series = _to_series(returns)
    values = series.to_numpy(dtype=float)
    if len(values) < 10:
        raise ValueError("Bootstrap CVaR needs a reasonable sample size.")

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        draw = rng.choice(values, size=len(values), replace=True)
        var_level = float(np.quantile(draw, alpha))
        cvar = float(draw[draw <= var_level].mean())
        boot.append(cvar)
    boot = np.asarray(boot, dtype=float)
    point_var = float(np.quantile(values, alpha))
    point_cvar = float(values[values <= point_var].mean())
    lower = float(np.quantile(boot, (1 - ci) / 2))
    upper = float(np.quantile(boot, 1 - (1 - ci) / 2))

    frame = pd.DataFrame({"bootstrap_cvar": boot})
    hist = alt.Chart(frame).transform_density("bootstrap_cvar", as_=["bootstrap_cvar", "density"]).mark_area(opacity=0.28, color="#2a9d8f").encode(
        x=alt.X("bootstrap_cvar:Q", title="Bootstrap CVaR"),
        y=alt.Y("density:Q", title="Density"),
    )
    rules = alt.Chart(pd.DataFrame({"x": [point_cvar, lower, upper], "label": ["Point estimate", "CI lower", "CI upper"]})).mark_rule(strokeWidth=2, strokeDash=[6, 4]).encode(
        x="x:Q",
        color=alt.Color("label:N", scale=alt.Scale(domain=["Point estimate", "CI lower", "CI upper"], range=["#111111", "#c0392b", "#c0392b"]), legend=alt.Legend(title=None)),
    )
    points = alt.Chart(pd.DataFrame({"x": [point_cvar], "label": ["Point estimate"]})).mark_point(size=90, color="#111111").encode(x="x:Q")
    chart = (hist + rules + points).properties(title=f"Bootstrap CVaR Confidence Interval (alpha={alpha:.2f})", width=720, height=280)
    return chart

def plot_vine_copula_structure():
    import networkx as nx
    import pandas as pd
    import altair as alt
    import matplotlib.pyplot as plt

    # 1. Initialize Graph
    G = nx.DiGraph()

    # Add nodes with their hierarchical layer and extra info from your table
    nodes_data = [
        ("SPY", 0, "Base"),
        ("AGG", 0, "Base"),
        ("PE",  0, "Base"),
        ("NPI", 0, "Base"),
        ("SPY,PE", 1, "Gaussian (τ=0.28)"),
        ("AGG,PE", 1, "Clayton 90° (τ=-0.03)"),
        ("PE,NPI", 1, "Gumbel (τ=0.40)"),
        ("SPY,NPI|PE", 2, "Gumbel 270° (τ=-0.05)"),
        ("AGG,NPI|PE", 2, "Student (τ=0.02)"),
        ("SPY,AGG|NPI,PE", 3, "Clayton 270° (τ=-0.09)")
    ]

    for node, layer, info in nodes_data:
        G.add_node(node, layer=layer, info=info)

    # Define how base nodes combine into pairwise, and pairwise into conditional
    edges_data = [
        ("SPY", "SPY,PE"), ("PE", "SPY,PE"),
        ("AGG", "AGG,PE"), ("PE", "AGG,PE"),
        ("PE", "PE,NPI"), ("NPI", "PE,NPI"),
        ("SPY,PE", "SPY,NPI|PE"), ("PE,NPI", "SPY,NPI|PE"),
        ("AGG,PE", "AGG,NPI|PE"), ("PE,NPI", "AGG,NPI|PE"),
        ("SPY,NPI|PE", "SPY,AGG|NPI,PE"), ("AGG,NPI|PE", "SPY,AGG|NPI,PE")
    ]
    G.add_edges_from(edges_data)

    # 2. Map fixed geometric positions to avoid crossed lines
    pos = {
        "SPY": (0, 4), "PE": (2, 4), "NPI": (4, 4), "AGG": (6, 4),
        "SPY,PE": (1, 3), "PE,NPI": (3, 3), "AGG,PE": (5, 3),  
        "SPY,NPI|PE": (2, 2), "AGG,NPI|PE": (4, 2), 
        "SPY,AGG|NPI,PE": (3, 1) 
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    nx.draw_networkx_edges(G, pos, edge_color="#888888", ax=ax, arrows=True, arrowstyle="-|>", arrowsize=15, width=1.5)
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color="#ADD8E6", edgecolors="#5D92B5", linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

    for node, (x, y) in pos.items():
        info = G.nodes[node]["info"]
        if info != "Base":
            ax.text(x, y - 0.2, info, ha='center', va='top', fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.axis("off")
    plt.title("Vine Copula Structure", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    
def plot_asset_path(
    simulation,
    path_id=10,
    width=800,
    height=300,
    num_simulations=1000,
    asset_names=None,
):

    path_id = int(np.clip(path_id, 0, num_simulations - 1))

    df = (
        pd.DataFrame(simulation["returns"][path_id, :, :], columns=asset_names)
        .add(1)
        .cumprod()
        .sub(1)
        .reset_index()
        .rename(columns={"index": "period"})
    )

    df_long = df.melt(
        id_vars="period",
        var_name="asset",
        value_name="compounded_return"
    )

    chart = (
        alt.Chart(df_long)
        .mark_line(size=2)
        .encode(
            x=alt.X("period:Q", title="Period"),
            y=alt.Y("compounded_return:Q", title="Compounded Return"),
            color=alt.Color("asset:N", title=None),
            tooltip=[
                alt.Tooltip("period:Q", title="Period"),
                alt.Tooltip("asset:N", title="Asset"),
                alt.Tooltip("compounded_return:Q", title="Compounded Return", format=".2%")
            ]
        )
        .properties(
            width=width,
            height=height,
            title=f"Simulation Path {path_id+1}"
        )
        .configure_view(
            stroke="black",
            strokeWidth=1
        )
    )

    return chart

def plot_simulations(
    simulation,
    historical_portfolio_return,
    risk_summary,
    y_min: float = -0.88,
    y_max: float = 2.0,
    n_paths: int = 2000,
    width_left: int = 520,
    width_right: int = 140,
    height: int = 380,
    random_seed: int = None,
):
    """
    Plot a 12-Quarter Path Portfolio Return Simulation chart using Altair.

    Parameters
    ----------
    simulation : dict
        Must contain:
          - 'quarter_portfolio': array-like of shape (n_paths, 12)
          - 'compounded_return': array-like of length n_paths
    historical_portfolio_return : pd.Series
        Quarterly historical returns (at least 82 entries for random window).
    risk_summary : pd.DataFrame
        Must contain a 'Value' column where:
          - index 2 → 12Q VaR 95%
          - index 3 → 12Q CVaR 95%
    y_min : float
        Lower bound of y-axis (default -0.88).
    y_max : float
        Upper bound of y-axis (default 2.0).
    n_paths : int
        Number of simulation paths to plot (default 1000).
    width_left : int
        Width of the left (fan) panel in pixels (default 520).
    width_right : int
        Width of the right (histogram) panel in pixels (default 140).
    height : int
        Height of both panels in pixels (default 380).
    random_seed : int or None
        Seed for reproducible random path selection (default None).

    Returns
    -------
    alt.Chart
        Combined Altair hconcat chart.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    var_val  = risk_summary['Value'][2]
    cvar_val = risk_summary['Value'][3]

    # ── Simulation paths ──────────────────────────────────────────────────
    rows = []
    for path_idx in range(n_paths):
        cum = (np.insert(simulation['quarter_portfolio'][path_idx], 0, 0) + 1).cumprod() - 1
        for q, v in enumerate(cum):
            rows.append({'path': path_idx, 'quarter': q,
                         'value': float(np.clip(v, y_min, y_max))})
    paths_df = pd.DataFrame(rows)

    # ── Two random historical windows ─────────────────────────────────────
    max_start = len(historical_portfolio_return) - 12
    idx1 = np.random.randint(0, max_start)
    idx2 = np.random.randint(0, max_start)

    def _make_hist_path(idx, label):
        cum = (np.insert(historical_portfolio_return[idx:idx+12].values, 0, 0) + 1).cumprod() - 1
        return pd.DataFrame({'quarter': range(13), 'value': cum.clip(y_min, y_max), 'label': label})

    hist_df = pd.concat([_make_hist_path(idx1, 'random historical period 1'),
                         _make_hist_path(idx2, 'random historical period 2')], ignore_index=True)

    # ── Risk lines ────────────────────────────────────────────────────────
    risk_df = pd.DataFrame({
        'y':     [var_val,        cvar_val],
        'label': ['12Q VaR 95%',  '12Q CVaR 95%'],
    })

    # ── Terminal returns ──────────────────────────────────────────────────
    terminal_df = pd.DataFrame({
        'value': np.clip(simulation['compounded_return'], y_min, y_max)
    })

    # ── Unified scales ────────────────────────────────────────────────────
    all_labels  = ['random historical period 1',  'random historical period 2',   '12Q VaR 95%', '12Q CVaR 95%']
    all_colors  = ['steelblue', 'darkorange', 'red',          'black']
    all_dashes  = [[1, 0],      [1, 0],       [6, 4],         [1, 0]]

    unified_color = alt.Scale(domain=all_labels, range=all_colors)
    unified_dash  = alt.Scale(domain=all_labels, range=all_dashes)
    y_scale       = alt.Scale(domain=[y_min, y_max], clamp=True)
    one_legend    = alt.Legend(title=None, orient='top-left', offset=5, symbolStrokeWidth=2)

    # ── Left panel ────────────────────────────────────────────────────────
    sim_lines = (
        alt.Chart(paths_df)
        .mark_line(opacity=0.1, color='grey', strokeWidth=0.8)
        .encode(
            x=alt.X('quarter:Q', scale=alt.Scale(domain=[0, 12]),
                    axis=alt.Axis(title='Quarters', tickMinStep=1)),
            y=alt.Y('value:Q', scale=y_scale,
                    axis=alt.Axis(title='Compounded Return')),
            detail='path:N'
        )
    )

    hist_lines = (
        alt.Chart(hist_df)
        .mark_line(strokeWidth=2)
        .encode(
            x='quarter:Q',
            y=alt.Y('value:Q', scale=y_scale),
            color=alt.Color('label:N', scale=unified_color, legend=one_legend),
            strokeDash=alt.StrokeDash('label:N', scale=unified_dash, legend=None)
        )
    )

    risk_rules_left = (
        alt.Chart(risk_df)
        .mark_rule(strokeWidth=1.8)
        .encode(
            y=alt.Y('y:Q', scale=y_scale),
            color=alt.Color('label:N', scale=unified_color, legend=one_legend),
            strokeDash=alt.StrokeDash('label:N', scale=unified_dash, legend=None)
        )
    )

    left_panel = (
        alt.layer(sim_lines, hist_lines, risk_rules_left)
        .properties(width=width_left, height=height)
        .resolve_scale(color='shared', strokeDash='shared')
    )

    # ── Right panel ───────────────────────────────────────────────────────
    histo = (
        alt.Chart(terminal_df)
        .mark_bar(color='steelblue', opacity=0.7)
        .encode(
            y=alt.Y('value:Q', bin=alt.Bin(maxbins=100), scale=y_scale, axis=None),
            x=alt.X('count():Q', axis=None)
        )
    )

    risk_rules_right = (
        alt.Chart(risk_df)
        .mark_rule(strokeWidth=1.8)
        .encode(
            y=alt.Y('y:Q', scale=y_scale),
            color=alt.Color('label:N', scale=unified_color, legend=None),
            strokeDash=alt.StrokeDash('label:N', scale=unified_dash, legend=None)
        )
    )

    right_panel = (
        alt.layer(histo, risk_rules_right)
        .properties(width=width_right, height=height)
    )

    # ── Combine ───────────────────────────────────────────────────────────
    chart = (
        alt.hconcat(left_panel, right_panel, spacing=2)
        .properties(title=alt.TitleParams(
            '12 Quarter Path Portfolio Return Simulation',
            fontSize=14, anchor='start'))
        .resolve_scale(y='shared', color='independent', strokeDash='independent')
        .configure_view(stroke='black', strokeWidth=0.5)
    )

    return chart


def plot_quarter_return_hist(simulation, asset_name, opacity=0.85, step=0.005, color='blue', horizon_quarters=12, alpha=0.05):
    asset_idx_dict = {'SPY': 0, 'AGG': 1, 'PE': 2, 'NPI': 3}

    if asset_name is None:
        asset_name = 'Portfolio'
        data = pd.DataFrame({
            "value": simulation['quarter_portfolio'].flatten()
        })
    else:
        data = pd.DataFrame({
            "value": simulation['returns'][:, :, asset_idx_dict[asset_name]].flatten()
        })

    Quantile = simulation['returns'].shape[0] * horizon_quarters * alpha
    var1Q = data["value"].nsmallest(int(Quantile)).iloc[-1]
    cvar1Q = data["value"].nsmallest(int(Quantile)).mean()

    # Histogram
    hist = alt.Chart(data).mark_bar(color=color, opacity=opacity).encode(
        x=alt.X("value:Q", bin=alt.Bin(step=step), title='Singel quarter return'),
        y=alt.Y('count()', title='Freq')).properties(title=f'Simulated Single Quarter {asset_name} Return Distribution', width=820, height=300)

    line1 = alt.Chart(pd.DataFrame({'x': [var1Q]})).mark_rule(color='red',strokeDash=[6,4]).encode(x='x:Q')
    line2 = alt.Chart(pd.DataFrame({'x': [cvar1Q]})).mark_rule(color='black').encode(x='x:Q')

    # 合併l
    chart = (hist + line1 + line2)
    return chart

def plot_compound_return_dist_by_asset(simulation, asset_name, opacity=0.85, step=0.05, color='purple', withline=True, alpha=0.05):
    asset_idx_dict = {'SPY': 0, 'AGG': 1, 'PE': 2, 'NPI': 3}

    data = pd.DataFrame({"value": simulation["asset_compounded_return"][:, asset_idx_dict[asset_name]]})

    hist = (alt.Chart(data)
            .mark_bar(color=color, opacity=opacity)
            .encode(x=alt.X("value:Q", bin=alt.Bin(step=step), title="12Q Compounded Return"),y='count()')
            ).properties(title=f'Simulated Compounded {asset_name} Return Distribution',
                width=600,
                height=300
            )

    if withline:
        Quantile = simulation["asset_compounded_return"].shape[0] * alpha
        var = data["value"].nsmallest(int(Quantile)).iloc[-1]
        cvar = data["value"].nsmallest(int(Quantile)).mean()
        line1 = alt.Chart(pd.DataFrame({'x': [var]})).mark_rule(color='red',strokeDash=[6,4]).encode(x='x:Q')
        line2 = alt.Chart(pd.DataFrame({'x': [cvar]})).mark_rule(color='black').encode(x='x:Q')
        chart = (hist + line1 + line2)
    else:
        chart = hist

    return chart