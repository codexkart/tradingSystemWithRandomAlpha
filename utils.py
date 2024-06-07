import lzma
import dill as pickle
import pandas as pd
import numpy as np
import random
from datetime import timedelta

def load_pickle(path):
    with lzma.open(path, "rb") as fp:
        file = pickle.load(fp)
    return file

def save_pickle(path, obj):
    with lzma.open(path, "wb") as fp:
        pickle.dump(obj, fp)

def get_pnl_stats(date, prev, portfolio_df, insts, idx, dfs):
    day_pnl = 0
    nominal_ret = 0
    for inst in insts:
        units = portfolio_df.loc[idx-1, "{} units".format(inst)]
        if units != 0:
            delta = dfs[inst].loc[date, "open"] - dfs[inst].loc[prev, "close"]
            inst_pnl = delta * units
            day_pnl += inst_pnl
            dfs[inst].loc[date, 'ret'] = day_pnl
            nominal_ret += portfolio_df.loc[idx-1, "{} w".format(inst)] * dfs[inst].loc[date, "ret"]
    capital_ret = nominal_ret * portfolio_df.loc[idx-1, 'leverage']
    portfolio_df.loc[idx, 'capital'] = portfolio_df.loc[idx-1, 'capital'] + day_pnl
    portfolio_df.loc[idx, 'day_pnl'] = day_pnl
    portfolio_df.loc[idx, 'nominal_ret'] = nominal_ret
    portfolio_df.loc[idx, 'capital_ret'] = capital_ret
    return day_pnl, capital_ret

class Alpha():
    def __init__(self, insts, dfs, start, end):
        self.insts = insts
        self.dfs = dfs
        self.start = start
        self.end = end

    def init_portfolio_settings(self, trade_range):
        portfolio_df = pd.DataFrame(index=trade_range).reset_index().rename(columns={'index': 'datetime'})
        portfolio_df.loc[0, 'capital'] = 10000
        portfolio_df['capital'] = portfolio_df['capital'].ffill()  # Ensure capital is forward filled
        return portfolio_df

    def compute_meta_info(self, trade_range):
        for inst in self.insts:
            df = pd.DataFrame(index=trade_range)
            self.dfs[inst] = df.join(self.dfs[inst], how='outer').ffill().bfill()
            self.dfs[inst]['ret'] = -1 + self.dfs[inst]['close'] / self.dfs[inst]['close'].shift(1)
            sampled = self.dfs[inst]['close'] != self.dfs[inst]['close'].shift(1).bfill()
            eligible = sampled.rolling(5).apply(lambda x: int(np.any(x))).fillna(0)
            self.dfs[inst]['eligible'] = eligible.astype(int) & (self.dfs[inst]['close'] > 0).astype(int)
        return

    def run_simulation(self):
        print("Running backtest")
        date_range = pd.date_range(start=self.start - timedelta(hours=12.0), end=self.end + timedelta(hours=12.0), freq='D')
        self.compute_meta_info(trade_range=date_range)
        portfolio_df = self.init_portfolio_settings(trade_range=date_range)

        for i in portfolio_df.index:
            date = portfolio_df.loc[i, "datetime"]

            eligibles = [inst for inst in self.insts if self.dfs[inst].loc[date, "eligible"]]
            non_eligibles = [inst for inst in self.insts if inst not in eligibles]

            if i != 0:
                date_prev = portfolio_df.loc[i-1, "datetime"]
                day_pnl, capital_ret = get_pnl_stats(date=date, prev=date_prev, portfolio_df=portfolio_df, insts=self.insts, idx=i, dfs=self.dfs)
            else:
                day_pnl = 0
                capital_ret = 0

            alpha_scores = {}
            for inst in eligibles:
                alpha_scores[inst] = random.uniform(0, 1)

            alpha_scores = {k: v for k, v in sorted(alpha_scores.items(), key=lambda pair: pair[1])}
            alpha_long = list(alpha_scores.keys())[-int(len(eligibles) / 4):]
            alpha_short = list(alpha_scores.keys())[:int(len(eligibles) / 4)]

            for inst in non_eligibles:
                portfolio_df.loc[i, "{} w".format(inst)] = 0
                portfolio_df.loc[i, "{} units".format(inst)] = 0

            nominal_tot = 0

            for inst in eligibles:
                if inst in alpha_long:
                    forecast = 1
                elif inst in alpha_short:
                    forecast = -1
                else:
                    forecast = 0

                # Ensure capital is not NaN
                capital = portfolio_df.loc[i, 'capital']
                if pd.isna(capital) or capital == 0:
                    print(f"Capital is NaN or zero at index {i}.")
                    dollar_allocation = 0
                else:
                    dollar_allocation = capital / (len(alpha_long) + len(alpha_short))

                # Ensure close price is not NaN
                close_price = self.dfs[inst].loc[date, 'close']
                if pd.isna(close_price) or close_price == 0:
                    position = 0
                else:
                    position = (forecast * dollar_allocation) / close_price

                portfolio_df.loc[i, inst + ' units'] = position
                nominal_tot += abs(position * close_price)

            for inst in eligibles:
                units = portfolio_df.loc[i, inst + ' units']
                nominal_inst = units * self.dfs[inst].loc[date, 'close']
                inst_w = nominal_inst / nominal_tot
                portfolio_df.loc[i, inst + ' w'] = inst_w

            portfolio_df.loc[i, 'nominal'] = nominal_tot
            portfolio_df.loc[i, 'leverage'] = nominal_tot / portfolio_df.loc[i, 'capital'] if portfolio_df.loc[i, 'capital'] != 0 else 0

            # Ensure capital is propagated correctly for the next iteration
            if i < len(portfolio_df.index) - 1:
                portfolio_df.loc[i + 1, 'capital'] = portfolio_df.loc[i, 'capital']

            if i % 100 == 0:
                print(portfolio_df.loc[i])
        return portfolio_df
