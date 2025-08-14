import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import logging

class PortfolioOptimizer:
    def __init__(self, tsla_csv, bnd_csv, spy_csv,column='forecast'):
        """
        Initialize the PortfolioOptimizer class with paths to CSV files for forecast data.

        Parameters:
        - tsla_csv: Path to Tesla forecast CSV file
        - bnd_csv: Path to BND forecast CSV file
        - spy_csv: Path to SPY forecast CSV file
        - logger: Optional logger instance
        - column: Column name containing forecast prices
        """
        self.tsla_csv = tsla_csv
        self.bnd_csv = bnd_csv
        self.spy_csv = spy_csv
        self.column = column

        self.df = self._load_data()
        self.weights = np.array([1/3, 1/3, 1/3])  # Start with equal weights for TSLA, BND, SPY

    def _load_data(self):
        """
        Load forecast data from CSV files and combine them into a single DataFrame.
        """
        try:
            tsla_data = pd.read_csv(self.tsla_csv, index_col=0, parse_dates=True)[self.column]
            bnd_data = pd.read_csv(self.bnd_csv, index_col=0, parse_dates=True)[self.column]
            spy_data = pd.read_csv(self.spy_csv, index_col=0, parse_dates=True)[self.column]
            df = pd.DataFrame({'TSLA': tsla_data, 'BND': bnd_data, 'SPY': spy_data}).dropna()
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")
        return df

    def calculate_annual_returns(self, trading_days=252):
        """
        Calculate annualized returns based on daily forecast data.

        Parameters:
        - trading_days: Number of trading days in a year (default: 252)

        Returns:
        - annualized_returns: Annualized returns for each asset
        """
        daily_returns = self.df.pct_change().dropna()
        print(daily_returns.describe())
        avg_daily_returns = daily_returns.mean()
        annualized_returns = (1 + avg_daily_returns) ** trading_days - 1
        return annualized_returns

    def portfolio_statistics(self, weights):
        """
        Calculate portfolio's return, volatility (risk), and Sharpe ratio.

        Parameters:
        - weights: Portfolio weights

        Returns:
        - portfolio_return: Portfolio's annualized return
        - portfolio_std_dev: Portfolio's annualized standard deviation (risk)
        - sharpe_ratio: Portfolio's Sharpe ratio
        """
        annual_returns = self.calculate_annual_returns()
        portfolio_return = np.dot(weights, annual_returns)

        daily_returns = self.df.pct_change().dropna()
        covariance_matrix = daily_returns.cov() * 252  # Annualize covariance
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        portfolio_std_dev = np.sqrt(portfolio_variance)

        sharpe_ratio = portfolio_return / portfolio_std_dev
        return portfolio_return, portfolio_std_dev, sharpe_ratio

    def optimize_portfolio(self):
        """
        Optimize portfolio weights to maximize the Sharpe Ratio.

        Returns:
        - Dictionary containing optimized weights, return, risk, and Sharpe ratio
        """
        def neg_sharpe_ratio(weights):
            return -self.portfolio_statistics(weights)[2]

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(self.df.columns)))
        initial_weights = self.weights

        optimized = minimize(neg_sharpe_ratio, initial_weights, bounds=bounds, constraints=constraints)
        optimal_weights = optimized.x
        optimal_return, optimal_risk, optimal_sharpe = self.portfolio_statistics(optimal_weights)

        return {
            "weights": optimal_weights,
            "return": optimal_return,
            "risk": optimal_risk,
            "sharpe_ratio": optimal_sharpe
        }

    def risk_metrics(self, confidence_level=0.95):
        """
        Calculate key risk metrics, including volatility and Value at Risk (VaR).

        Parameters:
        - confidence_level: Confidence level for VaR calculation (default: 95%)

        Returns:
        - Dictionary containing volatility and VaR
        """
        daily_returns = self.df.pct_change().dropna()
        portfolio_daily_returns = daily_returns.dot(self.weights)

        volatility = portfolio_daily_returns.std() * np.sqrt(252)
        var_95 = np.percentile(portfolio_daily_returns, (1 - confidence_level) * 100)

        return {"volatility": volatility, "VaR_95": var_95}

    def visualize_portfolio_performance(self):
        """
        Plot both daily returns and cumulative returns for the optimized portfolio and individual assets.
        """
        daily_returns = self.df.pct_change().dropna()
        portfolio_daily_returns = daily_returns.dot(self.weights)

        cumulative_returns = (1 + daily_returns).cumprod()
        cumulative_returns_portfolio = (1 + portfolio_daily_returns).cumprod()

        plt.figure(figsize=(14, 8))

        # Plot daily returns
        plt.subplot(2, 1, 1)
        plt.plot(portfolio_daily_returns, label="Optimized Portfolio (Daily Return)", color="purple", linewidth=2)
        for asset in self.df.columns:
            plt.plot(daily_returns[asset], label=f"{asset} (Daily Return)", linestyle="--")
        plt.title("Daily Returns of Optimized Portfolio vs Individual Assets")
        plt.xlabel("Date")
        plt.ylabel("Daily Return")
        plt.legend()
        plt.grid(True)

        # Plot cumulative returns
        plt.subplot(2, 1, 2)
        plt.plot(cumulative_returns_portfolio, label="Optimized Portfolio (Cumulative Return)", color="purple", linewidth=2)
        for asset in self.df.columns:
            plt.plot(cumulative_returns[asset], label=f"{asset} (Cumulative Return)", linestyle="--")
        plt.title("Cumulative Returns of Optimized Portfolio vs Individual Assets")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def summary(self):
        """
        Generate a summary of the portfolio's performance metrics.

        Returns:
        - Dictionary containing optimized weights, return, risk, Sharpe ratio, volatility, and VaR
        """
        optimal_results = self.optimize_portfolio()
        risk_metrics = self.risk_metrics()
        summary = {
            "Optimized Weights": optimal_results["weights"],
            "Expected Portfolio Return": optimal_results["return"],
            "Expected Portfolio Volatility": optimal_results["risk"],
            "Sharpe Ratio": optimal_results["sharpe_ratio"],
            "Annualized Volatility": risk_metrics["volatility"],
            "Value at Risk (VaR) at 95% Confidence": risk_metrics["VaR_95"]
        }
        return summary
tsla = './Data/tsla_forecast_12_months.csv'
bnd = './Data/bnd_forecast_12_months.csv'
spy = './Data/spy_forecast_12_months.csv'   
t=PortfolioOptimizer(tsla,bnd,spy)
t._load_data()
t.calculate_annual_returns()