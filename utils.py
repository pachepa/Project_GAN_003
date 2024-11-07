import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class TradingStrategy:
    def __init__(self, data_path=None):
        self.data = None
        self.operations = []
        self.cash = 1_000_000  # Initial cash for backtesting
        self.strategy_value = [self.cash]  # Tracks portfolio value over time
        self.n_shares = 10  # Number of shares per trade
        self.commission = 0.00125  # Commission rate
        self.indicators = {
            'RSI': {'buy': self.rsi_buy_signal, 'sell': self.rsi_sell_signal},
            'SMA': {'buy': self.sma_buy_signal, 'sell': self.sma_sell_signal},
            'MACD': {'buy': self.macd_buy_signal, 'sell': self.macd_sell_signal}
        }
        self.active_indicators = []

    def load_data(self, data):
        self.data = data.dropna().copy()
        self.prepare_data_with_indicators()  # Prepare data by adding indicators

    def prepare_data_with_indicators(self):
        """Calculate and add required technical indicators to self.data."""
        # RSI Calculation
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        # SMA Calculation
        self.data['SHORT_SMA'] = self.data['Close'].rolling(window=50).mean()
        self.data['LONG_SMA'] = self.data['Close'].rolling(window=200).mean()

        # MACD Calculation
        short_ema = self.data['Close'].ewm(span=12, adjust=False).mean()
        long_ema = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = short_ema - long_ema
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()

    def activate_indicator(self, indicator_name):
        """Activate an indicator for trading strategy use."""
        if indicator_name in self.indicators:
            self.active_indicators.append(indicator_name)

    # --- Signal Functions for Buy/Sell based on Indicators ---
    def rsi_buy_signal(self, row):
        return row.RSI < 30  # Buy when RSI < 30

    def rsi_sell_signal(self, row):
        return row.RSI > 70  # Sell when RSI > 70

    def sma_buy_signal(self, row):
        return row.SHORT_SMA > row.LONG_SMA  # Buy on short SMA crossover

    def sma_sell_signal(self, row):
        return row.SHORT_SMA < row.LONG_SMA  # Sell on short SMA cross under

    def macd_buy_signal(self, row):
        return row.MACD > row.Signal_Line

    def macd_sell_signal(self, row):
        return row.MACD < row.Signal_Line

    # --- Backtesting Execution ---
    def execute_trades(self, sl_levels, tp_levels):
        for i, row in self.data.iterrows():
            # Check buy/sell signals for active indicators
            buy_signals = sum([self.indicators[ind]['buy'](row) for ind in self.active_indicators])
            sell_signals = sum([self.indicators[ind]['sell'](row) for ind in self.active_indicators])

            # Execute trade if signals meet criteria
            if buy_signals >= len(self.active_indicators) // 2:
                self._open_operation('long', row)
            elif sell_signals >= len(self.active_indicators) // 2 and self.operations:
                self._close_operations(row)

            # Update portfolio value
            self.strategy_value.append(self.cash + sum(self._operation_value(op, row['Close']) for op in self.operations))

    def _open_operation(self, operation_type, row):
        stop_loss = row['Close'] * 0.95
        take_profit = row['Close'] * 1.05
        self.operations.append({'type': operation_type, 'entry_price': row['Close'], 'stop_loss': stop_loss, 'take_profit': take_profit})
        self.cash -= row['Close'] * self.n_shares * (1 + self.commission)

    def _close_operations(self, row):
        for op in self.operations:
            self.cash += row['Close'] * self.n_shares * (1 - self.commission)
        self.operations.clear()

    def _operation_value(self, op, current_price):
        if op['type'] == 'long':
            return (current_price - op['entry_price']) * self.n_shares
        else:
            return (op['entry_price'] - current_price) * self.n_shares

    # --- Plot Results ---
    def plot_results(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.strategy_value)
        plt.title('Trading Strategy Performance')
        plt.xlabel('Trades')
        plt.ylabel('Portfolio Value')
        plt.show()


class Backtesting:
    def __init__(self, historical_data_path, synthetic_data_path):
        self.historical_data = self.load_data(historical_data_path)
        self.synthetic_data = pd.read_csv(synthetic_data_path)

    def load_data(self, path):
        data = pd.read_csv(path)
        data['Date'] = pd.to_datetime(data['Date'])
        return data

    def run_backtest_with_sl_tp(self, data, sl, tp, shares):
        strategy = TradingStrategy()
        strategy.load_data(data)
        strategy.activate_indicator('RSI')
        #strategy.activate_indicator('SMA')
        #strategy.activate_indicator('MACD')

        # Set strategy parameters
        strategy.n_shares = shares
        print(f"Running backtest with SL: {sl}, TP: {tp}, Shares: {shares}")
        strategy.execute_trades(sl_levels=[sl], tp_levels=[tp])

        # Capture the final cash balance after executing trades
        final_cash_balance = strategy.strategy_value[-1]
        strategy.operations.clear()
        strategy.cash = 1_000_000
        strategy.strategy_value = [strategy.cash]

        return final_cash_balance

    def run_all_sl_tp_variations(self):
        # Define ranges for SL, TP, and n_shares
        sl_range = (0.90, 0.99)
        tp_range = (1.01, 1.10)
        n_shares_range = (1, 100)

        best_final_cash = None
        best_params = {}

        # Generate 10 random combinations of SL, TP, and n_shares
        for i in range(10):
            sl = round(random.uniform(*sl_range), 2)  # Random SL in range 0.90 to 0.99
            tp = round(random.uniform(*tp_range), 2)  # Random TP in range 1.01 to 1.10
            shares = 100  # Random number of shares from 1 to 100

            # Run backtest with the generated combination
            final_cash_balance = self.run_backtest_with_sl_tp(self.historical_data, sl, tp, shares)

            if best_final_cash is None or final_cash_balance > best_final_cash:
                best_final_cash = final_cash_balance
                best_params = {'stop_loss_pct': sl, 'take_profit_pct': tp, 'n_shares': shares}

        print("Best parameters found:")
        print(f"Stop Loss: {best_params['stop_loss_pct']}")
        print(f"Take Profit: {best_params['take_profit_pct']}")
        print(f"Number of Shares: {best_params['n_shares']}")
        print(f"Final Cash Balance: {best_final_cash}")

        return best_final_cash

    def backtest_synthetic_scenarios(self, num_scenarios=10):
        print(f"Starting backtesting on synthetic data ({num_scenarios} scenarios)...")

        for i in range(1, num_scenarios + 1):
            synthetic_scenario_data = self.historical_data.copy()
            synthetic_scenario_data['Close'] = self.synthetic_data[f'Scenario_{i}']

            print(f"\nRunning backtest on Synthetic Scenario {i}/{num_scenarios}")
            strategy = TradingStrategy()
            strategy.load_data(synthetic_scenario_data)
            strategy.cash = 1_000_000
            strategy.strategy_value = [strategy.cash]
            # Run backtest with specific SL, TP, and n_shares values
            final_cash_balance = self.run_backtest_with_sl_tp(
                synthetic_scenario_data,
                sl=0.95,  # Example stop-loss level for synthetic scenario testing
                tp=1.05,  # Example take-profit level for synthetic scenario testing
                shares=50  # Example number of shares for testing
            )

    def calculate_annual_performance(self, initial_cash, final_cash, data):
        p_and_l = final_cash - initial_cash
        total_return = p_and_l / initial_cash

        # Annualized return
        years = (data['Date'].iloc[-1] - data['Date'].iloc[0]).days / 365.0
        annualized_return = (1 + total_return) ** (1 / years) - 1

        # Sharpe Ratio (Risk-Free Rate = 0)
        data['Daily Returns'] = data['Close'].pct_change().fillna(0)
        std_dev = np.std(data['Daily Returns']) * np.sqrt(252)
        sharpe_ratio = annualized_return / std_dev if std_dev != 0 else 0

        # Max Drawdown
        data['Cumulative Value'] = (1 + data['Daily Returns']).cumprod() * initial_cash
        peak = data['Cumulative Value'].cummax()
        drawdown = (data['Cumulative Value'] - peak) / peak
        max_drawdown = drawdown.min()

        # Calmar Ratio
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0

        # Win-Loss Ratio
        data['Trade Result'] = data['Daily Returns'] > 0
        wins = data['Trade Result'].sum()
        losses = len(data['Trade Result']) - wins
        win_loss_ratio = wins / losses if losses > 0 else np.inf

        return {
            "P&L": p_and_l,
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Sharpe Ratio": sharpe_ratio,
            "Calmar Ratio": calmar_ratio,
            "Max Drawdown": max_drawdown,
            "Win-Loss Ratio": win_loss_ratio
        }

def calculate_passive_benchmark(data, initial_cash=1_000_000, strategy_final_cash=None, risk_free_rate=0.0):
    data = data.dropna().sort_values(by='Date')
    first_close = data['Close'].iloc[0]
    n_shares = initial_cash // first_close
    last_close = data['Close'].iloc[-1]

    # Calculate P&L for passive strategy
    passive_final_value = data['Close'].iloc[-1] * n_shares
    passive_return = (passive_final_value - initial_cash) / initial_cash

    # Calculate daily returns for passive strategy
    data['Returns'] = data['Close'].pct_change().fillna(0)
    investment_values = data['Close'] * n_shares

    # Calculate annual Sharpe Ratio for passive strategy
    annual_sharpe_ratio = (data['Returns'].mean() - risk_free_rate) / data['Returns'].std() * np.sqrt(252) if data[
                                                                                                                  'Returns'].std() != 0 else 0

    # Calculate Calmar Ratio for passive strategy
    max_drawdown = (investment_values.min() - investment_values.max()) / investment_values.max()
    calmar_ratio = passive_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Calculate Win-Loss Ratio for passive strategy
    wins = sum(1 for r in data['Returns'] if r > 0)
    losses = sum(1 for r in data['Returns'] if r < 0)
    win_loss_ratio = wins / losses if losses > 0 else float('inf')

    # Plot passive performance
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Closing Price', color='blue')
    plt.title('Asset Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], investment_values, label='Investment Value (Passive)', color='green')
    plt.title('Passive Investment Performance Over Time')
    plt.xlabel('Date')
    plt.ylabel('Investment Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print comparison if active strategy final cash is provided
    if strategy_final_cash is not None:
        strategy_return = (strategy_final_cash - initial_cash) / initial_cash
        passive_vs_strategy = passive_return - strategy_return
        print("Passive vs. Active Strategy Return Difference: {:.2%}".format(passive_vs_strategy))

        return {
            "Passive P&L": passive_final_value - initial_cash,
            "Passive Total Return": passive_return,
            "Passive Annual Sharpe Ratio": annual_sharpe_ratio,
            "Passive Calmar Ratio": calmar_ratio,
            "Passive Max Drawdown": max_drawdown,
            "Passive Win-Loss Ratio": win_loss_ratio,
            "Passive Final Value": passive_final_value,
            "Strategy Final Value": strategy_final_cash,
            "Return Difference": passive_vs_strategy
        }
    else:
        return {
            "Passive P&L": passive_final_value - initial_cash,
            "Passive Total Return": passive_return,
            "Passive Annual Sharpe Ratio": annual_sharpe_ratio,
            "Passive Calmar Ratio": calmar_ratio,
            "Passive Max Drawdown": max_drawdown,
            "Passive Win-Loss Ratio": win_loss_ratio,
            "Passive Final Value": passive_final_value
        }