import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from portfolio_optimizer import optimize_portfolio, portfolio_performance


# Streamlit App Code
st.title("Portfolio Optimizer")

# Get user input for stock tickers
tickers_input = st.text_input("Enter stock tickers separated by commas (e.g., AAPL, MSFT, GOOG):")
tickers = [ticker.strip() for ticker in tickers_input.split(",")]

if st.button("Optimize Portfolio"):
    # Fetch Data
    data = yf.download(tickers, start="2020-01-01", end="2023-01-01")["Adj Close"]
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # Optimize Portfolio
    optimal_weights = optimize_portfolio(mean_returns, cov_matrix)
    expected_return, expected_risk = portfolio_performance(optimal_weights, mean_returns, cov_matrix)

    # Display Results
    st.write("Optimal Weights:", optimal_weights)
    st.write("Expected Annual Return:", expected_return)
    st.write("Expected Annual Risk:", expected_risk)

    # # Plot Efficient Frontier
    # st.write("Efficient Frontier:")
    # plot_efficient_frontier(mean_returns, cov_matrix, optimal_weights)
    # st.pyplot()
