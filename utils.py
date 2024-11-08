import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates


class TradingStrategy:
    def __init__(self):
        self.data = None
        self.operations = []
        self.cash = 1_000_000
        self.portfolio_value = [self.cash]
        self.n_shares = 100
        self.commission = 0.00125
        self.active_indicators = []

    def load_data(self, data):
        self.data = data.dropna().copy()
        self.prepare_data_with_indicators()

    def prepare_data_with_indicators(self):
        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        self.data['SHORT_SMA'] = self.data['Close'].rolling(window=50).mean()
        self.data['LONG_SMA'] = self.data['Close'].rolling(window=200).mean()
        short_ema = self.data['Close'].ewm(span=12, adjust=False).mean()
        long_ema = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = short_ema - long_ema
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()

    def activate_indicator(self, indicator_name):
        if indicator_name in ['RSI', 'SMA', 'MACD']:
            self.active_indicators.append(indicator_name)

    def execute_trades(self, sl, tp):
        for i, row in self.data.iterrows():
            buy_signal, sell_signal = self.evaluate_signals(row)
            if buy_signal:
                self._open_operation(row, sl, tp)
            elif sell_signal and self.operations:
                self._close_operations(row)
            self._update_portfolio_value(row)

    def evaluate_signals(self, row):
        buy_signals = sum([self.indicator_signal(ind, row, 'buy') for ind in self.active_indicators])
        sell_signals = sum([self.indicator_signal(ind, row, 'sell') for ind in self.active_indicators])
        buy_signal = buy_signals >= len(self.active_indicators) // 2
        sell_signal = sell_signals >= len(self.active_indicators) // 2
        return buy_signal, sell_signal

    def indicator_signal(self, indicator, row, signal_type):
        if indicator == 'RSI':
            return row['RSI'] < 30 if signal_type == 'buy' else row['RSI'] > 70
        elif indicator == 'SMA':
            return row['SHORT_SMA'] > row['LONG_SMA'] if signal_type == 'buy' else row['SHORT_SMA'] < row['LONG_SMA']
        elif indicator == 'MACD':
            return row['MACD'] > row['Signal_Line'] if signal_type == 'buy' else row['MACD'] < row['Signal_Line']
        return False

    def _open_operation(self, row, sl, tp):
        entry_price = row['Close']
        stop_loss = entry_price * sl
        take_profit = entry_price * tp
        self.operations.append({
            'type': 'long',
            'Date': row['Date'],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'shares': self.n_shares
        })
        self.cash -= entry_price * self.n_shares * (1 + self.commission)

    def _close_operations(self, row):
        for op in self.operations:
            exit_price = row['Close']
            self.cash += exit_price * self.n_shares * (1 - self.commission)
        self.operations.clear()

    def _update_portfolio_value(self, row):
        total_value = self.cash + sum(
            (row['Close'] - op['entry_price']) * op['shares'] for op in self.operations
        )
        self.portfolio_value.append(total_value)


class Backtesting:
    def __init__(self, historical_data_path, synthetic_data_path):
        self.historical_data = self.load_data(historical_data_path)
        self.synthetic_data = pd.read_csv(synthetic_data_path)
        self.operations = []

    def load_data(self, path):
        data = pd.read_csv(path)
        data['Date'] = pd.to_datetime(data['Date'])
        return data

    def run_backtest_with_sl_tp(self, data, sl, tp, shares):
        strategy = TradingStrategy()
        strategy.load_data(data)
        strategy.activate_indicator('RSI')
        strategy.n_shares = shares
        strategy.execute_trades(sl=sl, tp=tp)
        self.operations = strategy.operations  # Guardar operaciones
        return strategy.portfolio_value, strategy.operations

    def run_all_sl_tp_variations(self):
        best_final_cash = None
        best_params = {}
        best_portfolio_values = []

        for i in range(10):
            sl = round(random.uniform(0.90, 0.99), 2)
            tp = round(random.uniform(1.01, 1.10), 2)
            shares = 100

            portfolio_value, operations = self.run_backtest_with_sl_tp(self.historical_data, sl, tp, shares)

            final_cash_balance = portfolio_value[-1]
            if best_final_cash is None or final_cash_balance > best_final_cash:
                best_final_cash = final_cash_balance
                best_params = {'stop_loss_pct': sl, 'take_profit_pct': tp, 'n_shares': shares}
                best_portfolio_values = portfolio_value

        return best_final_cash, best_portfolio_values

    def backtest_synthetic_scenarios(self, num_scenarios=100, save_every_n=10):
        print(f"Starting backtesting on synthetic data ({num_scenarios} scenarios)...")

        for i in range(1, num_scenarios + 1):
            synthetic_scenario_data = self.historical_data.copy()
            synthetic_scenario_data['Close'] = self.synthetic_data[f'Scenario_{i}']

            print(f"\nRunning backtest on Synthetic Scenario {i}/{num_scenarios}")
            strategy = TradingStrategy()
            strategy.load_data(synthetic_scenario_data)
            strategy.cash = 1_000_000
            strategy.strategy_value = [strategy.cash]

            final_cash_balance, portfolio_values = self.run_backtest_with_sl_tp(
                synthetic_scenario_data,
                sl=0.95,  # Example stop-loss level for synthetic scenario testing
                tp=1.05,  # Example take-profit level for synthetic scenario testing
                shares=100  # Example number of shares for testing
            )

            # Save operations for only every nth scenario
            if i % save_every_n == 0:
                self.operations.extend(strategy.operations)

    def backtest_synthetic_scenarios(self, num_scenarios=10):
        for i in range(1, num_scenarios + 1):
            synthetic_data = self.historical_data.copy()
            synthetic_data['Close'] = self.synthetic_data[f'Scenario_{i}']
            portfolio_value, operations = self.run_backtest_with_sl_tp(synthetic_data, sl=0.95, tp=1.05, shares=50)

    def calculate_annual_performance(self, initial_cash, final_cash, data):
        p_and_l = final_cash - initial_cash
        total_return = p_and_l / initial_cash

        # Annualized return
        years = (data['Date'].iloc[-1] - data['Date'].iloc[0]).days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1

        # Sharpe Ratio (Risk-Free Rate = 0)
        data['Daily Returns'] = data['Close'].pct_change().fillna(0)
        std_dev = np.std(data['Daily Returns']) * np.sqrt(252)
        sharpe_ratio = (annualized_return / std_dev)**0.5 if std_dev != 0 else 0

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

    def save_operations(self, filename='results/operations_list.csv'):
        if not os.path.exists('results'):
            os.makedirs('results')

        if self.operations:
            operations_df = pd.DataFrame(self.operations)
            if os.path.exists(filename):
                existing_df = pd.read_csv(filename)
                operations_df = pd.concat([existing_df, operations_df], ignore_index=True)

            operations_df.to_csv(filename, index=False)
            print(f"Operations saved to {filename}")

    def plot_strategy_values(self, portfolio_values, dates):
        if not os.path.exists('plots'):
            os.makedirs('plots')
        portfolio_values = portfolio_values[:-1]
        # Graficar valor del portafolio
        plt.figure(figsize=(12, 6))
        plt.plot(dates, portfolio_values, label="Portfolio Value", color="blue")
        plt.title("Trading Strategy Value Over Time")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True)
        plt.savefig('plots/trading_strat.png')
        plt.show()

    def plot_candlestick_chart(self, data):
        if not os.path.exists('plots'):
            os.makedirs('plots')

        data = data.copy()
        data['Date_Num'] = mdates.date2num(data['Date'])

        ohlc = data[['Date_Num', 'Open', 'High', 'Low', 'Close']].values
        fig, ax = plt.subplots(figsize=(12, 6))
        candlestick_ohlc(ax, ohlc, width=0.6, colorup='green', colordown='red')
        ax.xaxis_date()
        ax.set_title('Candlestick Chart')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        plt.grid(True)
        plt.savefig('plots/candlestick_chart.png')
        plt.show()


def calculate_passive_benchmark(data, initial_cash=1_000_000, strategy_final_cash=None, risk_free_rate=0.0):
    data = data.dropna().sort_values(by='Date')
    first_close = data['Close'].iloc[0]
    last_close = data['Close'].iloc[-1]

    # Calculate number of shares bought initially
    n_shares = initial_cash // first_close
    passive_final_value = last_close * n_shares
    passive_return = (passive_final_value - initial_cash) / initial_cash

    passive_annualized_return = (1 + passive_return) ** (1 / 252) - 1

    # Calculate daily returns for passive strategy
    data['Returns'] = data['Close'].pct_change().fillna(0)
    investment_values = data['Close'] * n_shares

    # Plot investment value over time
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], investment_values, label='Investment Value (Passive)', color='green')
    plt.title('Passive Investment Performance Over Time')
    plt.xlabel('Date')
    plt.ylabel('Investment Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/passive_performance_overtime.png')
    plt.show()

    # Calculate performance metrics
    annual_sharpe_ratio = ((data['Returns'].mean() - risk_free_rate) / (data['Returns'].std() * np.sqrt(252)))**0.5
    max_drawdown = (investment_values.min() - investment_values.max()) / investment_values.max()
    calmar_ratio = passive_return / abs(max_drawdown) if max_drawdown != 0 else 0
    wins = sum(1 for r in data['Returns'] if r > 0)
    losses = sum(1 for r in data['Returns'] if r < 0)
    win_loss_ratio = wins / losses if losses > 0 else float('inf')

    # Compare to active strategy
    if strategy_final_cash is not None:
        strategy_return = (strategy_final_cash - initial_cash) / initial_cash
        passive_vs_strategy = passive_return - strategy_return
        print("Passive vs. Active Strategy Return Difference: {:.2%}".format(passive_vs_strategy))

        return {
            "Passive P&L": passive_final_value - initial_cash,
            "Passive Total Return": passive_return,
            "Annualized Passive Return": passive_annualized_return,
            "Passive Annual Sharpe Ratio": annual_sharpe_ratio,
            "Passive Calmar Ratio": calmar_ratio,
            "Passive Max Drawdown": max_drawdown,
            "Passive Win-Loss Ratio": win_loss_ratio,
        }
    else:
        return {
            "Passive P&L": passive_final_value - initial_cash,
            "Passive Total Return": passive_return,
            "Passive Annual Sharpe Ratio": annual_sharpe_ratio,
            "Passive Calmar Ratio": calmar_ratio,
            "Passive Max Drawdown": max_drawdown,
            "Passive Win-Loss Ratio": win_loss_ratio,
            "Passive Final Value": passive_final_value,
        }