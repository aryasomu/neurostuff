import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt

class SimpleStrategy(bt.Strategy):
    params = dict(short_period=5, long_period=15)

    def __init__(self):
        self.sma_short = bt.indicators.EMA(self.data.close, period=self.params.short_period)
        self.sma_long = bt.indicators.EMA(self.data.close, period=self.params.long_period)
        self.trades = []
        self.current_trade = None
        self.order = None
        self.profit_loss = []

    def next(self):
        if self.order:
            return

        if self.sma_short[0] > self.sma_long[0] and self.sma_short[-1] <= self.sma_long[-1]:
            if self.position.size <= 0:
                self.order = self.buy()
                entry_price = self.data.close[0]
                self.current_trade = {
                    'type': 'BUY',
                    'date': self.data.datetime.date(),
                    'price': entry_price,
                    'reason': f'Short-term EMA ({self.params.short_period} days) crossed above Long-term EMA ({self.params.long_period} days)',
                    'short_ema': self.sma_short[0],
                    'long_ema': self.sma_long[0],
                    'ema_difference': self.sma_short[0] - self.sma_long[0]
                }
                self.trades.append(self.current_trade)

        elif self.sma_short[0] < self.sma_long[0] and self.sma_short[-1] >= self.sma_long[-1]:
            if self.position.size >= 0:
                self.order = self.sell()
                exit_price = self.data.close[0]
                self.current_trade = {
                    'type': 'SELL',
                    'date': self.data.datetime.date(),
                    'price': exit_price,
                    'reason': f'Short-term EMA ({self.params.short_period} days) crossed below Long-term EMA ({self.params.long_period} days)',
                    'short_ema': self.sma_short[0],
                    'long_ema': self.sma_long[0],
                    'ema_difference': self.sma_short[0] - self.sma_long[0]
                }
                self.trades.append(self.current_trade)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
            self.order = None

    def log(self, txt, dt=None):
        dt = dt or self.data.datetime.date()
        print(f'{dt}: {txt}')

def backtest_strategy(data):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SimpleStrategy)
    data_bt = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data_bt)
    cerebro.broker.setcash(10000)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)
    
    print("\n🔄 Running backtest...\n")
    results = cerebro.run()
    
    strategy = results[0]
    trades = strategy.trades
    
    print("\n📊 Trade Analysis and Reasoning:")
    print("=" * 80)
    
    total_trades = len(trades)
    if total_trades > 0:
        print(f"\nTotal Trades Made: {total_trades}")
        print("\nDetailed Trade Analysis:")
        print("-" * 80)
        
        for i, trade in enumerate(trades, 1):
            print(f"\n🔄 Trade #{i}:")
            print(f"Type: {trade['type']}")
            print(f"Date: {trade['date']}")
            print(f"Price: ${trade['price']:.2f}")
            print("\n📝 Trade Reasoning:")
            print(f"- {trade['reason']}")
            print("\n📈 Technical Indicators at Trade:")
            print(f"- Short-term EMA: ${trade['short_ema']:.2f}")
            print(f"- Long-term EMA: ${trade['long_ema']:.2f}")
            print(f"- EMA Difference: ${trade['ema_difference']:.2f}")
            print("-" * 40)
    else:
        print("\n⚠️ No trades were executed during this period.")
        print("This could be due to:")
        print("- No clear trading signals")
        print("- Market conditions not meeting strategy criteria")
        print("- Insufficient price movement to trigger trades")
    
    print("\n" + "=" * 80)
    final_portfolio = cerebro.broker.getvalue()
    print(f"\n💰 Final Portfolio Value: ${final_portfolio:.2f}")
    print(f"📈 Return: {((final_portfolio - 10000) / 10000 * 100):.2f}%")
    print("=" * 80)
    
    # Store the trades and data before plotting
    trades_to_return = trades
    data_to_return = data
    
    # Plot after storing the data
    cerebro.plot()
    
    # Return the stored data
    return trades_to_return, data_to_return
