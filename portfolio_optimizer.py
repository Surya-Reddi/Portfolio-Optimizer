import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

tickers = ["MSFT", "GOOG"]  # Example tickers for Apple, Microsoft, and Google
data = yf.download(tickers, start="2020-01-01", end="2023-01-01")["Adj Close"]
returns = data.pct_change().dropna()

mean_returns = returns.mean() * 252  # Annualized return
cov_matrix = returns.cov() * 252     # Annualized covariance matrix

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, risk

def portfolio_sharpe_ratio(weights, mean_returns, cov_matrix):
    returns, risk = portfolio_performance(weights, mean_returns, cov_matrix)
    return -returns / risk  # Negative for minimization

def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(portfolio_sharpe_ratio, num_assets * [1. / num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

optimal_weights = optimize_portfolio(mean_returns, cov_matrix)
expected_return, expected_risk = portfolio_performance(optimal_weights, mean_returns, cov_matrix)

print("Optimal Weights:", optimal_weights)
print("Expected Annual Return:", expected_return)
print("Expected Annual Risk:", expected_risk)
