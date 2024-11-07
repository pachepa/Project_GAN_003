## Algorithmic Trading with GAN-Generated Scenarios

### Project Overview

This project aims to develop and backtest a simple algorithmic trading strategy using both real and synthetic data. The synthetic data is generated through a Generative Adversarial Network (GAN) trained on historical stock prices. The project includes the following key steps:

1. Download and preprocess 10 years of historical stock data.
2. Train a GAN model to simulate hypothetical stock price scenarios.
3. Implement a rule-based algorithmic trading strategy.
4. Backtest the strategy using both historical data and 100 GAN-generated synthetic scenarios.
5. Evaluate performance using key financial metrics.
6. Benchmark against a passive "buy-and-hold" strategy.

### Project Objectives

- **Historical Data**: Utilize 10 years of stock price data to train the GAN model.
- **Scenario Generation**: Create 100 synthetic price paths to simulate various market conditions.
- **Trading Strategy**: Design a simple algorithmic strategy to evaluate different trading scenarios without hyperparameter optimization.
- **Backtesting**: Test the strategy on both real and synthetic data, evaluating 10 stop-loss/take-profit combinations.
- **Benchmarking**: Compare the active strategy against a passive buy-and-hold benchmark.
- **Performance Metrics**: Measure annual financial metrics including:
  - Sharpe Ratio
  - Calmar Ratio
  - Max Drawdown
  - Profit & Loss (P&L)
  - Win-Loss Ratio
- **Jupyter Notebook Results**: Present results with detailed plots and metrics in a notebook, focusing on insights rather than raw code.
  
### Project Structure

The repository follows a standard Python project structure:
```plaintext
data/  
    ├── cleaned_data_AMZN.csv  # Historical stock data  
    ├── synthetic_data_AMZN.csv  # GAN-generated synthetic scenarios  
models/  
    ├── gan_generator_amzn.h5  # Trained GAN model  
notebooks/  
    ├── results.ipynb  # Jupyter notebook with results & analysis  
src/  
    ├── GAN_model.py  # GAN implementation & training  
    ├── utils.py  # Backtesting & performance metrics  
README.md  # Project overview
