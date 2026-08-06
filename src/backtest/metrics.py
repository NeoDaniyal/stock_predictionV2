import numpy as np
import pandas as pd


def calculate_total_return(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    return (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]


def calculate_cagr(
    equity_curve: pd.Series, periods_per_year: int = 252
) -> float:
    if len(equity_curve) < 2:
        return 0.0
    total_return = calculate_total_return(equity_curve)
    n_years = len(equity_curve) / periods_per_year
    return (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0


def calculate_volatility(
    equity_curve: pd.Series, periods_per_year: int = 252
) -> float:
    returns = equity_curve.pct_change().dropna()
    return float(returns.std() * np.sqrt(periods_per_year))


def calculate_sharpe(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess_return = returns.mean() - (risk_free_rate / periods_per_year)
    return float((excess_return / returns.std()) * np.sqrt(periods_per_year))


def calculate_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return float(drawdown.min())


def calculate_win_rate(trade_returns: pd.Series) -> float:
    if len(trade_returns) == 0:
        return 0.0
    return float((trade_returns > 0).sum() / len(trade_returns))


def calculate_profit_factor(trade_profits: pd.Series) -> float:
    gross_profit = trade_profits[trade_profits > 0].sum()
    gross_loss = abs(trade_profits[trade_profits < 0].sum())
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def calculate_average_trade(trade_returns: pd.Series) -> float:
    return float(trade_returns.mean()) if len(trade_returns) > 0 else 0.0


def calculate_average_win_loss(trade_returns: pd.Series):
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    return avg_win, avg_loss