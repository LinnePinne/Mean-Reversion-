import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from math import sqrt
import os

plt.style.use("default")

# ==========================
# KONFIG: MARKNADER & FILER
# ==========================

markets = [
    {
        "name": "US500",
        "csv": "US500_1D_2012-2025.csv",
    },
    {
        "name": "US30",
        "csv": "US30_1D_2012-2025.csv",
    },
    {
        "name": "USTECH",
        "csv": "USTEC_1D_2012-2025.csv",
    },
]

PORTFOLIO_WEIGHTS = {
    "US500": 0.2,
    "US30":   0.4,
    "USTECH": 0.4,
}

START_CAPITAL = 50_000
EXPOSURE_PCT = 0.5  # 10% av kapitalet per trade

# ==========================
# GEMENSAMMA PARAMETRAR
# ==========================

# Session (intraday-fönster)
# session_start = "08:00:00"
# session_end   = "19:55:00"

# ==========================
# COST MODEL (POINTS)
# ==========================
HALF = 0.5
SLIPPAGE_POINTS = 0.5
# Spread och kommission uttryckt i samma enhet som priset i din CSV (points)
FIXED_SPREAD_POINTS = 0.8
COMM_POINTS_PER_SIDE = 0.05  # per side (entry eller exit)


def commission_round_turn_points():
    """Kommission per round-turn (entry+exit) i points."""
    return 2.0 * COMM_POINTS_PER_SIDE


def run_backtest_for_market(market_name: str, csv_path: str):
    print("\n" + "=" * 70)
    print(f" BACKTEST FÖR MARKNAD: {market_name}")
    print("=" * 70 + "\n")

    # ==========================
    # 1. Ladda data
    # ==========================

    df = pd.read_csv(csv_path)

    # Anpassa kolumnnamn om de skiljer sig
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    else:
        raise ValueError("Hittar ingen 'timestamp' eller 'datetime'-kolumn i CSV.")

    df = df.sort_index()

    required_cols = {'open', 'high', 'low', 'close'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV måste innehålla kolumnerna: {required_cols}")

    # Ta reda på minsta pris-enhet (för att bestämma pip_size)
    '''
    diffs = df["close"].diff().abs()
    tick_est = diffs[diffs > 0].quantile(0.01)  # robust: ignorerar outliers
    print("Estimated min step ~", tick_est)'''

    # ==========================
    # 2. Indikatorer
    # ==========================

    # EMA (snabb)
    df['ema_fast'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=250, adjust=False).mean()

    USE_SPREAD_COLUMN = 'spread_points' in df.columns

    def get_spread_points(row):
        if USE_SPREAD_COLUMN:
            return float(row['spread_points'])
        return FIXED_SPREAD_POINTS

    # ==========================
    # 4. Backtest-loop
    # ==========================
    equity = START_CAPITAL

    trades = []

    in_position = False
    pos_direction = None
    entry_price = None
    entry_time = None

    idx_list = df.index.to_list()

    for i in range(1, len(df) - 1):
        ts = idx_list[i]
        row = df.iloc[i]

        current_time = ts.time()

        # ======================
        # Om vi redan är i trade: kolla SL/TP
        # ======================
        if in_position:
            exit_price = None
            exit_reason = None

            if row["high"] >= row["ema_fast"]:
                spread = get_spread_points(row)
                exit_price = row["ema_fast"] - HALF * spread - SLIPPAGE_POINTS
                exit_reason = 'ema_touch_exit'

            if exit_price is not None:
                exit_time = ts
                if pos_direction == 'LONG':
                    pnl = (exit_price - entry_price) - commission_round_turn_points()

                    pnl_points = (exit_price - entry_price) - commission_round_turn_points()
                    pnl_cash = pnl_points * pos_size

                    equity += pnl_cash

                trades.append({
                    'Entry Time': entry_time,
                    'Exit Time': exit_time,
                    'Direction': pos_direction,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Exit Reason': exit_reason,
                    'pnl': pnl,
                    'PnL (points)': pnl_points,
                    'PnL ($)': pnl_cash,
                    'Equity After': equity,
                    'Exposure ($)': pos_exposure,
                    'Entry Fill Time': entry_fill_time,
                })

                in_position = False
                pos_direction = None
                entry_price = None
                entry_time = None

            # Om vi fortfarande är i position -> hoppa entrylogik
        if in_position:
            continue

        # ======================
        # Sessionfilter
        # ======================
        # if not (session_start <= current_time.strftime("%H:%M:%S") <= session_end):
        # continue

        ema_fast = row['ema_fast']
        ema_slow = row['ema_slow']
        high = row['high']
        low = row['low']
        if np.isnan(ema_slow):
            continue

        close_price = row['close']

        # EMA Crossover-logik
        bullish_trend = close_price < ema_fast and close_price > ema_slow

        # RSI-filter
        deep_pullback = close_price < ((0.2 * (high - low)) + low)

        # Slutlig entry-signal
        long_entry_signal = bullish_trend and deep_pullback

        # EN trade åt gången
        if long_entry_signal:
            pos_direction = 'LONG'
            entry_time = ts

            next_row = df.iloc[i + 1]
            next_open = next_row['open']
            spread = get_spread_points(next_row)
            entry_price = next_open + HALF * spread  # LONG: köp på ask
            entry_fill_time = idx_list[i + 1]  # timestamp för next_row

            # Fixed exposure sizing
            position_value = equity * EXPOSURE_PCT
            position_size = position_value / entry_price
            pos_size = position_size
            pos_exposure = position_value

            in_position = True

            if (pos_direction is None):
                # reset
                pos_direction = None
                entry_price = None
                entry_time = None
                in_position = False
                continue

    # ==========================
    # 5. Resultatsammanställning
    # ==========================
    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        print("Inga trades hittades.")
        return None, trades_df

    trades_df = trades_df.sort_values("Exit Time").reset_index(drop=True)
    trades_df["equity"] = trades_df["pnl"].cumsum()

    # --- Extra statistik ---
    trades_df["is_win"] = trades_df["pnl"] > 0

    gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
    gross_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()  # negativt tal
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf

    avg_win = trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()
    avg_loss = trades_df.loc[trades_df["pnl"] < 0, "pnl"].mean()  # negativt

    winrate = trades_df["is_win"].mean()

    # Expectancy per trade
    expectancy = trades_df["pnl"].mean()

    # Drawdown
    roll_max = trades_df["equity"].cummax()
    dd = trades_df["equity"] - roll_max
    max_dd = dd.min()  # negativt
    max_dd_points = abs(max_dd)  # positivt för rapportering

    # Longest losing streak (räknat i trades)
    loss_streak = 0
    max_loss_streak = 0
    for is_win in trades_df["is_win"]:
        if not is_win:
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
        else:
            loss_streak = 0

    # “Sharpe” på trade-nivå (inte tidsnormaliserad)
    pnl_std = trades_df["pnl"].std(ddof=1)
    sharpe_trade = (expectancy / pnl_std) * sqrt(len(trades_df)) if pnl_std and pnl_std > 0 else np.nan

    stats = {
        "Market": market_name,
        "Trades": int(len(trades_df)),
        "Total PnL (points)": float(trades_df["pnl"].sum()),
        "Gross Profit": float(gross_profit),
        "Gross Loss": float(gross_loss),
        "Profit Factor": float(profit_factor),
        "Winrate": float(winrate),
        "Avg Win": float(avg_win) if not np.isnan(avg_win) else np.nan,
        "Avg Loss": float(avg_loss) if not np.isnan(avg_loss) else np.nan,
        "Expectancy (avg/trade)": float(expectancy),
        "Max Drawdown (points)": float(max_dd_points),
        "Max Losing Streak (trades)": int(max_loss_streak),
        "Sharpe (trade-level)": float(sharpe_trade) if not np.isnan(sharpe_trade) else np.nan,
        'Spread (points)': float(spread),
        'Commission RT (points)': float(commission_round_turn_points()),
    }

    print("\n--- STATS ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    trades_df["Equity"] = trades_df["Equity After"]

    '''
    # PLOT (som du redan får)
    plt.figure(figsize=(12, 5))
    plt.plot(trades_df["Exit Time"], trades_df["equity"])
    plt.title(f"Equity curve - {market_name}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL (points)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(trades_df["Exit Time"], trades_df["Equity"])
    plt.title("Capital Equity Curve (Fixed Exposure)")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    '''
    roll_max = trades_df["Equity"].cummax()
    dd_pct = (trades_df["Equity"] - roll_max) / roll_max

    max_dd_pct = dd_pct.min()
    print(f"Max Drawdown (%): {max_dd_pct * 100:.2f}%")
    return stats, trades_df, df


# ==========================
# KÖR BACKTEST + SLUTSUMMERING + COMBINED EQUITY & STATS
# ==========================

all_results = []
all_trades = []
all_dfs = {}
all_trades = {}
for m in markets:
    stats, trades_df, df = run_backtest_for_market(
        m["name"],
        m["csv"],
    )

    all_dfs[m["name"]] = df
    all_trades[m["name"]] = trades_df

def build_mtm_equity_and_metrics(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    start_capital: float,
    exposure_pct: float,
    trading_days_per_year: int = 252,
    use_trade_size_if_present: bool = True,
):
    """
    Bygger mark-to-market equity på daily bars:
      equity = cash + position_size * close (när position öppen)

    Antaganden:
    - Ingen hävstång (du investerar exposure_pct av aktuell equity i positionen).
    - Entry/exit prices i trades_df är redan "filled" priser (inkl spread/slippage om du byggt så).
    - Commission RT (points) dras vid exit (round-turn). (Entry-spread ligger i entry_price om du modellerar så.)

    Returnerar:
    - metrics (dict)
    - equity_series (pd.Series)
    - daily_returns (pd.Series)
    """

    if trades_df is None or trades_df.empty:
        raise ValueError("trades_df är tom – kan inte bygga equity.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index måste vara DatetimeIndex.")

    # Säkerställ sortering
    df = df.sort_index()
    trades = trades_df.copy()
    trades["Entry Time"] = pd.to_datetime(trades["Entry Time"])
    trades["Exit Time"]  = pd.to_datetime(trades["Exit Time"])
    trades = trades.sort_values("Exit Time").reset_index(drop=True)

    # Inferera faktisk entry-fill-tid om du sparar signal-tid men fyller på next bar
    # (Du kan förbättra detta permanent genom att spara 'Entry Fill Time' i backtesten.)
    if "Entry Fill Time" in trades.columns:
        trades["Entry Fill Time"] = pd.to_datetime(trades["Entry Fill Time"])
    else:
        # Försök hitta nästa index i df för varje entry time
        entry_fill_times = []
        idx = df.index
        for t in trades["Entry Time"]:
            # hitta position i index (om exakt match saknas -> hitta närmaste föregående)
            pos = idx.searchsorted(t)
            # Om t är exakt en bar-timestamp i df och vi gick in på next bar open,
            # så ska fill vara idx[pos+1] när idx[pos] == t.
            # Om searchsorted pekar på nästa större, vill vi använda den som "current bar"
            # och ta nästa därifrån.
            if pos < len(idx) and idx[pos] == t:
                fill_pos = pos + 1
            else:
                # t låg mellan bars -> anta att signalen inträffade på baren precis innan pos,
                # och fill sker på baren vid pos (nästa bar).
                fill_pos = pos

            if fill_pos >= len(idx):
                entry_fill_times.append(pd.NaT)
            else:
                entry_fill_times.append(idx[fill_pos])

        trades["Entry Fill Time"] = entry_fill_times

    # Filtrera bort trades som inte går att mappa till equity-index
    trades = trades.dropna(subset=["Entry Fill Time", "Exit Time"]).reset_index(drop=True)

    # Bygg event-mappar: flera trades samma dag hanteras i ordning (men du kör 1 åt gången)
    entries_by_time = {t: i for i, t in enumerate(trades["Entry Fill Time"])}
    exits_by_time   = {t: i for i, t in enumerate(trades["Exit Time"])}

    equity = start_capital
    cash = start_capital
    pos_size = 0.0
    pos_dir = None  # "LONG" / "SHORT"
    entry_price = None

    equity_path = []

    for ts, row in df.iterrows():
        # 1) Entry event
        if ts in entries_by_time:
            tr = trades.iloc[entries_by_time[ts]]

            if pos_size != 0.0:
                raise RuntimeError("Entry när position redan är öppen (förväntade 1 trade åt gången).")

            pos_dir = tr.get("Direction", "LONG")
            entry_price = float(tr["Entry Price"])

            # position sizing (fixed exposure)
            if use_trade_size_if_present and "Size" in tr and pd.notna(tr["Size"]):
                pos_size = float(tr["Size"])
                # cash justeras så att equity = cash + pos_value vid entry
                pos_value = pos_size * entry_price
                cash = equity - pos_value
            else:
                pos_value = equity * exposure_pct
                pos_size = pos_value / entry_price
                cash -= pos_value

        # 2) Mark-to-market equity på close
        close = float(row["close"])
        if pos_size != 0.0:
            if pos_dir == "LONG":
                equity = cash + pos_size * close
            elif pos_dir == "SHORT":
                # Om du implementerar short senare:
                # equity = cash + pos_size * (entry_price - close) ??? (beror på modell)
                # Här håller vi det enkelt.
                raise NotImplementedError("SHORT MTM ej implementerad i denna funktion.")
            else:
                raise ValueError(f"Okänd direction: {pos_dir}")
        else:
            equity = cash

        # 3) Exit event (efter MTM på close eller före? – vi använder exit price på exit-timestamp)
        # Om din exit_time är samma ts som baren där exit triggas och du fyller på EMA-nivå,
        # är det rimligt att bokföra exit samma dag.
        if ts in exits_by_time:
            tr = trades.iloc[exits_by_time[ts]]

            if pos_size == 0.0:
                raise RuntimeError("Exit utan öppen position.")

            exit_price = float(tr["Exit Price"])
            comm_rt_points = float(tr.get("Commission RT (points)", 0.0))

            # Realisera position till cash på exit_price
            cash = cash + pos_size * exit_price

            # Dra kommission (points * size => cash)
            cash -= comm_rt_points * pos_size

            # Stäng position
            pos_size = 0.0
            pos_dir = None
            entry_price = None

            # Efter exit är equity = cash
            equity = cash

        equity_path.append((ts, equity))

    equity_series = pd.Series(
        data=[v for _, v in equity_path],
        index=pd.DatetimeIndex([t for t, _ in equity_path]),
        name="Equity_MTM"
    )

    # Dagliga returns
    daily_returns = equity_series.pct_change().dropna()

    # Metrics
    ret_mean = daily_returns.mean()
    ret_std = daily_returns.std(ddof=1)

    sharpe = np.nan
    if ret_std and ret_std > 0:
        sharpe = (ret_mean / ret_std) * np.sqrt(trading_days_per_year)

    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std(ddof=1)

    sortino = np.nan
    if downside_std and downside_std > 0:
        sortino = (ret_mean / downside_std) * np.sqrt(trading_days_per_year)

    roll_max = equity_series.cummax()
    dd = equity_series / roll_max - 1.0
    max_dd = dd.min()  # negativ

    n_days = (equity_series.index[-1] - equity_series.index[0]).days
    cagr = np.nan
    if n_days > 0:
        years = n_days / 365.25
        cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1.0 / years) - 1.0

    calmar = np.nan
    if pd.notna(cagr) and pd.notna(max_dd) and max_dd < 0:
        calmar = cagr / abs(max_dd)

    metrics = {
        "Equity Start": float(equity_series.iloc[0]),
        "Equity End": float(equity_series.iloc[-1]),
        "CAGR": float(cagr) if pd.notna(cagr) else np.nan,
        "Max Drawdown (%)": float(max_dd * 100.0) if pd.notna(max_dd) else np.nan,
        "Sharpe (ann.)": float(sharpe) if pd.notna(sharpe) else np.nan,
        "Sortino (ann.)": float(sortino) if pd.notna(sortino) else np.nan,
        "Calmar": float(calmar) if pd.notna(calmar) else np.nan,
    }

    return metrics, equity_series, daily_returns

metrics_mtm, equity_mtm, rets_mtm = build_mtm_equity_and_metrics(
    df=df,
    trades_df=trades_df,
    start_capital=START_CAPITAL,
    exposure_pct=EXPOSURE_PCT,
    trading_days_per_year=252
)

print("\n--- EQUITY METRICS (MTM, DAILY) ---")
for k, v in metrics_mtm.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

'''
plt.figure(figsize=(12, 5))
plt.plot(equity_mtm.index, equity_mtm.values)
plt.title("Equity Curve (MTM, Fixed Exposure)")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.grid(True)
plt.tight_layout()
plt.show()
'''

def build_portfolio_mtm_equity(
    market_data: dict,
    trades_data: dict,
    weights: dict,
    start_capital: float,
    trading_days_per_year: int = 252,
):
    """
    market_data: {"US500": df, ...}
    trades_data: {"US500": trades_df, ...}
    weights:     {"US500": 0.3, ...}
    """

    equity_curves = {}

    for market, weight in weights.items():
        capital_slice = start_capital * weight

        metrics, equity_mtm, _ = build_mtm_equity_and_metrics(
            df=market_data[market],
            trades_df=trades_data[market],
            start_capital=capital_slice,
            exposure_pct=1.0,  # full exposure inside slice
            trading_days_per_year=trading_days_per_year,
        )

        equity_curves[market] = equity_mtm

    # Align alla equity-serier på gemensam kalender
    equity_df = pd.concat(equity_curves.values(), axis=1)
    equity_df.columns = equity_curves.keys()
    equity_df = equity_df.fillna(method="ffill").fillna(method="bfill")

    # Portfölj-equity = summa av sleeves
    portfolio_equity = equity_df.sum(axis=1)
    portfolio_equity.name = "Portfolio_Equity"

    return portfolio_equity

def portfolio_metrics_from_equity(
    equity: pd.Series,
    trading_days_per_year: int = 252,
):
    returns = equity.pct_change().dropna()

    mean = returns.mean()
    std = returns.std(ddof=1)

    sharpe = (mean / std) * np.sqrt(trading_days_per_year) if std > 0 else np.nan

    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1)
    sortino = (mean / downside_std) * np.sqrt(trading_days_per_year) if downside_std > 0 else np.nan

    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = dd.min()

    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0 if years > 0 else np.nan

    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    return {
        "Equity Start": equity.iloc[0],
        "Equity End": equity.iloc[-1],
        "CAGR": cagr,
        "Max DD (%)": max_dd * 100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
    }

market_data = {
    "US30": all_dfs["US30"],
    "US500": all_dfs["US500"],
    "USTECH": all_dfs["USTECH"],
}

trades_data = {
    "US30": all_trades["US30"],
    "US500": all_trades["US500"],
    "USTECH": all_trades["USTECH"],
}

portfolio_equity = build_portfolio_mtm_equity(
    market_data=market_data,
    trades_data=trades_data,
    weights=PORTFOLIO_WEIGHTS,
    start_capital=START_CAPITAL,
)

metrics = portfolio_metrics_from_equity(portfolio_equity)

print("\n--- PORTFOLIO METRICS ---")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

plt.figure(figsize=(12,5))
plt.plot(portfolio_equity.index, portfolio_equity.values)
plt.title("Multi-Market MTM Portfolio Equity")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.grid(True)
plt.tight_layout()
plt.show()



def monte_carlo_portfolio(
    returns: pd.Series,
    start_capital: float,
    n_sims: int = 10_000,
    seed: int = 42,
):
    """
    Monte Carlo via permutation (bootstrap without replacement per sim).
    """
    rng = np.random.default_rng(seed)
    returns = returns.dropna().values
    n = len(returns)

    max_dds = []
    end_equities = []

    for _ in range(n_sims):
        perm = rng.permutation(returns)

        equity = start_capital
        peak = start_capital
        max_dd = 0.0

        for r in perm:
            equity *= (1.0 + r)
            peak = max(peak, equity)
            dd = (equity - peak) / peak
            max_dd = min(max_dd, dd)

        max_dds.append(max_dd)
        end_equities.append(equity)

    return {
        "max_dds": np.array(max_dds),          # negativa tal
        "end_equities": np.array(end_equities)
    }

mc = monte_carlo_portfolio(
    returns=rets_mtm,          # dagliga MTM portfolio-returns
    start_capital=50_000,
    n_sims=10_000
)

max_dds = mc["max_dds"]
end_eq  = mc["end_equities"]

for p in [50, 75, 90, 95, 99]:
    print(f"{p}% DD: {np.percentile(max_dds, p)*100:.1f}%")

for level in [-0.15, -0.20, -0.25, -0.30]:
    prob = np.mean(max_dds <= level)
    print(f"P(DD ≤ {level*100:.0f}%) = {prob*100:.1f}%")

print("Median End Equity:", np.median(end_eq))
print("5% worst End Equity:", np.percentile(end_eq, 5))