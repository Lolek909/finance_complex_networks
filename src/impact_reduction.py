import pandas as pd


def sector_impact_reduction(log_returns: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    sectors = {}
    for ticker, sector in sector_map.items():
        sectors.setdefault(sector, []).append(ticker)

    adjusted = log_returns.copy()
    for sector, tickers in sectors.items():
        cols = [t for t in tickers if t in log_returns.columns]
        if not cols:
            continue
        sector_mean = log_returns[cols].mean(axis=1)
        adjusted[cols] = log_returns[cols].sub(sector_mean, axis=0)

    return adjusted
