import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb

def get_historical_data(tickers, start_date, end_date):
    """Fetches historical adjusted close prices for given tickers."""
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_returns(data):
    """Calculates daily returns from adjusted close prices."""
    returns = data.pct_change().dropna()
    return returns

def calculate_portfolio_return(weights, returns):
    """Calculates the portfolio return given weights and returns."""
    return np.sum(returns.mean() * weights) * 252  # Annualized return

def calculate_portfolio_volatility(weights, returns):
    """Calculates the portfolio volatility given weights and returns."""
    return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))) # Annualized volatility

def optimize_portfolio(returns):
    """Optimizes the portfolio to maximize Sharpe Ratio."""
    num_assets = len(returns.columns)
    initial_weights = np.array([1 / num_assets] * num_assets)
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    def negative_sharpe_ratio(weights, returns):
        """Calculates the negative Sharpe Ratio (to minimize)."""
        portfolio_return = calculate_portfolio_return(weights, returns)
        portfolio_volatility = calculate_portfolio_volatility(weights, returns)
        return - (portfolio_return / portfolio_volatility) #Assumes risk free rate of 0

    result = minimize(negative_sharpe_ratio, initial_weights, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def predict_future_returns(returns, test_size=0.2):
    """Predicts future returns using LightGBM."""
    predictions = {}
    for ticker in returns.columns:
        X = returns.drop(ticker, axis=1)
        y = returns[ticker]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        model = lgb.LGBMRegressor(random_state=42)
        model.fit(X_train, y_train)
        predictions[ticker] = model.predict(X_test)
    return pd.DataFrame(predictions, index=y_test.index)

def compare_indices_to_sp500(tickers, start_date, end_date):
    """Compares optimized index portfolio to S&P 500 and ML predictions."""
    try:
        data = get_historical_data(tickers, start_date, end_date)
        returns = calculate_returns(data)
        optimal_weights = optimize_portfolio(returns)

        optimal_portfolio_return = calculate_portfolio_return(optimal_weights, returns)
        optimal_portfolio_volatility = calculate_portfolio_volatility(optimal_weights, returns)
        sharpe_ratio_optimized = optimal_portfolio_return / optimal_portfolio_volatility

        sp500_returns = returns['^GSPC']
        sp500_annual_return = sp500_returns.mean() * 252
        sp500_annual_volatility = sp500_returns.std() * np.sqrt(252)
        sharpe_ratio_sp500 = sp500_annual_return / sp500_annual_volatility

        st.write("Optimal Portfolio Weights:")
        for ticker, weight in zip(tickers, optimal_weights):
            st.write(f"{ticker}: {weight:.4f}")

        st.write("\nOptimal Portfolio Performance:")
        st.write(f"  Annual Return: {optimal_portfolio_return:.4f}")
        st.write(f"  Annual Volatility: {optimal_portfolio_volatility:.4f}")
        st.write(f"  Sharpe Ratio: {sharpe_ratio_optimized:.4f}")

        st.write("\nS&P 500 Performance:")
        st.write(f"  Annual Return: {sp500_annual_return:.4f}")
        st.write(f"  Annual Volatility: {sp500_annual_volatility:.4f}")
        st.write(f"  Sharpe Ratio: {sharpe_ratio_sp500:.4f}")

        # Calculate and plot cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        optimal_portfolio_cumulative_returns = (1 + np.sum(returns * optimal_weights, axis=1)).cumprod()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(cumulative_returns['^GSPC'], label='S&P 500')
        ax.plot(optimal_portfolio_cumulative_returns, label='Optimal Portfolio')

        # Machine Learning Predictions
        predicted_returns = predict_future_returns(returns)
        ml_optimal_portfolio_cumulative_returns = (1 + np.sum(predicted_returns * optimal_weights, axis=1)).cumprod()
        ax.plot(ml_optimal_portfolio_cumulative_returns, label='ML Predicted Optimal Portfolio')

        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns')
        ax.set_title('Cumulative Returns Comparison')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.write(f"An error occurred: {e}")

# Streamlit App
st.title("Index Portfolio Optimization with ML")

# User Inputs
start_date = st.date_input("Start Date", pd.to_datetime('2018-01-01'))
end_date = st.date_input("End Date", pd.to_datetime('2023-12-31'))

index_options = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX', '^TNX'] # Add more indices as needed
selected_indices = st.multiselect("Select Indices", index_options, default=['^GSPC', '^DJI', '^IXIC', '^RUT'])

if st.button("Run Optimization"):
    if selected_indices:
        compare_indices_to_sp500(selected_indices, start_date, end_date)
    else:
        st.warning("Please select at least one index.")