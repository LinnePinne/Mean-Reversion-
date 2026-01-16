import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import ta  # finns i din miljö, men används inte här (behåller för kompatibilitet)

plt.style.use("default")

# ==========================
# KONFIG: MARKNADER & FILER
# ==========================

markets = [
    {"name": "US500", "csv": "US500_1D_2012-2025.csv"},
    {"name": "US100", "csv": "USTEC_1D_2012-2025.csv"},
    {"name": "US30",  "csv": "US30_1D_2012-2025.csv"},
]

# ==========================
# PORTFÖLJ
# ==========================
START_CAPITAL = 50_000
MAX_GROSS_EXPOSURE = 2.0         # cap: max 200% av equity i öppna positionsvärden
TARGET_GROSS_EXPOSURE = 2.0      # target: försök investera upp till 200% när möjligt

# ==========================
# INSTRUMENT: $ per indexpunkt per kontrakt
# (FTMO CFD $1/point enligt er)
# ==========================
POINT_VALUE = {
    "US500": 1.0,
    "US100": 1.0,
    "US30":  1.0,
}

# ==========================
# COST MODEL (POINTS)
# ==========================
HALF = 0.5
SLIPPAGE_POINTS = 0.5
FIXED_SPREAD_POINTS = 0.8
COMM_POINTS_PER_SIDE = 0.05  # per side

def commission_round_turn_points():
    return 2.0 * COMM_POINTS_PER_SIDE

# ============================================================
# SINGLE-MARKET BACKTEST HELPERS
# ============================================================

def load_market_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    else:
        raise ValueError("Hittar ingen 'timestamp' eller 'datetime' i CSV.")

    df = df.sort_index()

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV måste innehålla: {required_cols}")

    return df


def generate_trades_for_market(
    market_name: str,
    df: pd.DataFrame,
    ema_fast_len: int = 20,
    ema_slow_len: int = 250,
    pullback_frac: float = 0.20,     # deep pullback: close < low + frac*(high-low)
) -> pd.DataFrame:
    """
    Mean reversion-strategi (LONG only):
      Regim: close < ema_fast och close > ema_slow   (pullback i bullish regim)
      Trigger: "deep pullback" i baren
      Entry: nästa bars open + cost (ask)
      Exit: om barens high >= ema_fast -> exit nästa open - cost (bid)   (ema-touch exit)
      Forced exit vid slut.
    """

    df = df.copy()

    # Indicators
    df["ema_fast"] = df["close"].ewm(span=ema_fast_len, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow_len, adjust=False).mean()

    use_spread_col = "spread_points" in df.columns

    def spread_points(row) -> float:
        return float(row["spread_points"]) if use_spread_col else FIXED_SPREAD_POINTS

    trades = []

    in_position = False
    entry_signal_time = None
    entry_fill_time = None
    entry_price = None

    idx = df.index.to_list()

    for i in range(1, len(df) - 1):
        ts = idx[i]
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        # Indicators ready?
        if np.isnan(row["ema_slow"]) or np.isnan(row["ema_fast"]):
            continue

        # --- Manage exit (if in position) ---
        if in_position:
            # Exit trigger: touch ema_fast within bar
            if float(row["high"]) >= float(row["ema_fast"]):
                spr = spread_points(next_row)
                exit_fill_price = float(next_row["open"] - HALF * spr - SLIPPAGE_POINTS)  # sell on bid
                exit_fill_time = idx[i + 1]

                trades.append({
                    "Market": market_name,
                    "Direction": "LONG",
                    "Entry Signal Time": entry_signal_time,
                    "Entry Fill Time": entry_fill_time,
                    "Exit Fill Time": exit_fill_time,
                    "Entry Price": float(entry_price),
                    "Exit Price": float(exit_fill_price),
                    "Exit Reason": "ema_fast_touch",
                    "Comm RT (points)": float(commission_round_turn_points()),
                })

                in_position = False
                entry_signal_time = None
                entry_fill_time = None
                entry_price = None

            if in_position:
                continue

        # --- Entry logic (only if flat) ---
        close_px = float(row["close"])
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        high = float(row["high"])
        low = float(row["low"])

        # bullish regim + pullback
        bullish_pullback_regime = (close_px < ema_fast) and (close_px > ema_slow)

        # deep pullback in the candle range
        deep_pullback = close_px < (low + pullback_frac * (high - low))

        if bullish_pullback_regime and deep_pullback:
            spr = spread_points(next_row)
            entry_fill_price = float(next_row["open"] + HALF * spr + SLIPPAGE_POINTS)  # buy on ask
            entry_fill_time = idx[i + 1]

            in_position = True
            entry_signal_time = ts
            entry_price = entry_fill_price

            # Note: fill time is next bar index (i+1)
            # This is consistent with your structure.
            # keep it stored:
            # (Important: store entry fill time, not signal time)
            entry_fill_time = idx[i + 1]

    # --- Forced exit at end ---
    if in_position:
        last_row = df.iloc[-1]
        spr = float(last_row["spread_points"]) if "spread_points" in df.columns else FIXED_SPREAD_POINTS
        exit_fill_price = float(last_row["close"] - HALF * spr - SLIPPAGE_POINTS)
        exit_fill_time = df.index[-1]

        trades.append({
            "Market": market_name,
            "Direction": "LONG",
            "Entry Signal Time": entry_signal_time,
            "Entry Fill Time": entry_fill_time,
            "Exit Fill Time": exit_fill_time,
            "Entry Price": float(entry_price),
            "Exit Price": float(exit_fill_price),
            "Exit Reason": "forced_exit_end_of_test",
            "Comm RT (points)": float(commission_round_turn_points()),
        })

    return pd.DataFrame(trades)

# ============================================================
# PORTFÖLJ MTM (Samma motor/struktur som din fungerande kod)
# ============================================================

def build_portfolio_mtm_cash(
    market_dfs: dict,            # {"US500": df, ...}
    trades_df: pd.DataFrame,     # combined trades from all markets
    start_capital: float,
    max_gross_exposure: float = 1.0,
    target_gross_exposure: float = 1.0,
    weights: dict | None = None,
) -> tuple:

    tr = trades_df.copy()
    tr["Entry Fill Time"] = pd.to_datetime(tr["Entry Fill Time"])
    tr["Exit Fill Time"] = pd.to_datetime(tr["Exit Fill Time"])

    # Union time index
    all_index = None
    for mkt, df in market_dfs.items():
        dfx = df.sort_index()
        if dfx.index.has_duplicates:
            dfx = dfx[~dfx.index.duplicated(keep="last")]
        all_index = dfx.index if all_index is None else all_index.union(dfx.index)

    all_index = pd.DatetimeIndex(all_index.sort_values().unique())

    # Close matrix (ffill)
    closes = pd.DataFrame(index=all_index)
    for mkt, df in market_dfs.items():
        dfx = df.sort_index()
        if dfx.index.has_duplicates:
            dfx = dfx[~dfx.index.duplicated(keep="last")]
        closes[mkt] = dfx["close"].reindex(all_index).ffill()

    mkts = sorted(market_dfs.keys())
    n = len(mkts)

    if weights is None:
        weights = {m: 1.0 / n for m in mkts}
    else:
        missing = set(mkts) - set(weights.keys())
        if missing:
            raise ValueError(f"weights saknar marknader: {missing}")
        s = sum(weights[m] for m in mkts)
        if s <= 0:
            raise ValueError("weights summerar till 0 eller mindre.")
        weights = {m: weights[m] / s for m in mkts}

    print("\n[Portfolio] Using weights:")
    for k in sorted(weights.keys()):
        print(f"  {k}: {weights[k]:.4f}")

    entries = tr.sort_values(["Entry Fill Time", "Market"]).groupby("Entry Fill Time")
    exits = tr.sort_values(["Exit Fill Time", "Market"]).groupby("Exit Fill Time")

    positions = {}  # mkt -> dict(contracts, entry_price)

    cash = float(start_capital)
    realized_equity = float(start_capital)  # uppdateras endast vid exits (stängda trades)
    realized_equity_path = []
    equity_path = []
    gross_exposure_path = []
    open_positions_path = []

    for ts in all_index:
        # --- 1) Exits first ---
        if ts in exits.groups:
            block = exits.get_group(ts)
            for _, t in block.iterrows():
                mkt = t["Market"]
                if mkt not in positions:
                    continue

                pos = positions[mkt]
                contracts = float(pos["contracts"])
                entry_price = float(pos["entry_price"])
                exit_price = float(t["Exit Price"])
                pv = float(POINT_VALUE[mkt])

                exit_notional = contracts * exit_price * pv
                pnl_gross = (exit_price - entry_price) * contracts * pv

                comm_points = float(t.get("Comm RT (points)", 0.0))
                pnl_comm = comm_points * pv * contracts
                pnl_net = pnl_gross - pnl_comm
                realized_equity += pnl_net

                # Realize cash: receive exit notional, pay commission
                cash += exit_notional
                cash -= pnl_comm

                del positions[mkt]

        # --- 2) Entries ---
        if ts in entries.groups:
            block = entries.get_group(ts)

            # Compute MTM equity and gross exposure before placing new entries
            mtm_value = 0.0
            gross_exposure_value = 0.0
            for pmkt, pos in positions.items():
                px = float(closes.loc[ts, pmkt])
                pv = float(POINT_VALUE[pmkt])
                notional = float(pos["contracts"]) * px * pv
                mtm_value += notional
                gross_exposure_value += notional

            equity_now = cash + mtm_value
            gross_pct_now = (gross_exposure_value / equity_now) if equity_now > 0 else 0.0

            for _, t in block.iterrows():
                mkt = t["Market"]
                if mkt in positions:
                    continue

                entry_price = float(t["Entry Price"])
                pv = float(POINT_VALUE[mkt])

                # recompute after prior entries
                mtm_value = 0.0
                gross_exposure_value = 0.0
                for pmkt, pos in positions.items():
                    px = float(closes.loc[ts, pmkt])
                    pv2 = float(POINT_VALUE[pmkt])
                    notional = float(pos["contracts"]) * px * pv2
                    mtm_value += notional
                    gross_exposure_value += notional

                equity_now = cash + mtm_value
                gross_pct_now = (gross_exposure_value / equity_now) if equity_now > 0 else 0.0
                remaining_capacity_pct = max(0.0, max_gross_exposure - gross_pct_now)

                desired_notional = equity_now * target_gross_exposure * float(weights.get(mkt, 0.0))
                cap_notional = equity_now * remaining_capacity_pct
                position_notional = min(desired_notional, cap_notional)

                # No leverage beyond cash in this model (same as your code)
                # Tillåt "lånad" finansiering för att nå target exposure (proxy för CFD margin)
                # OBS: cash kan bli negativt, det är avsiktligt i denna modell.
                # position_notional = min(position_notional, cash)  # <-- ta bort
                pass

                denom = entry_price * pv
                contracts = (position_notional / denom) if denom > 0 else 0.0

                if contracts <= 0:
                    continue

                cash -= contracts * entry_price * pv
                positions[mkt] = {"contracts": contracts, "entry_price": entry_price}

        # --- 3) MTM at close ---
        mtm_value = 0.0
        gross_exposure_value = 0.0
        for pmkt, pos in positions.items():
            px = float(closes.loc[ts, pmkt])
            pv = float(POINT_VALUE[pmkt])
            notional = float(pos["contracts"]) * px * pv
            mtm_value += notional
            gross_exposure_value += notional

        equity = cash + mtm_value
        equity_path.append((ts, equity))
        open_positions_path.append((ts, len(positions)))
        gross_exposure_path.append((ts, gross_exposure_value / equity if equity > 0 else 0.0))
        realized_equity_path.append((ts, realized_equity))

    equity_series = pd.Series(
        [v for _, v in equity_path],
        index=pd.DatetimeIndex([t for t, _ in equity_path]),
        name="PortfolioEquity",
    )

    open_pos_series = pd.Series(
        [v for _, v in open_positions_path],
        index=pd.DatetimeIndex([t for t, _ in open_positions_path]),
        name="OpenPositions",
    )

    gross_exposure_series = pd.Series(
        [v for _, v in gross_exposure_path],
        index=pd.DatetimeIndex([t for t, _ in gross_exposure_path]),
        name="GrossExposurePct",
    )

    realized_equity_series = pd.Series(
        [v for _, v in realized_equity_path],
        index=pd.DatetimeIndex([t for t, _ in realized_equity_path]),
        name="RealizedEquity"
    )

    daily_equity = equity_series.resample("1D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()

    return equity_series, realized_equity_series, daily_returns, open_pos_series, gross_exposure_series

# ============================================================
# METRICS + ERC + BOOTSTRAP (samma struktur som din kod)
# ============================================================

def portfolio_metrics_from_equity(equity_series: pd.Series, daily_returns: pd.Series, trading_days=252) -> dict:
    n_days = (equity_series.index[-1] - equity_series.index[0]).days
    years = n_days / 365.25 if n_days > 0 else np.nan
    cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1.0 / years) - 1.0 if years and years > 0 else np.nan

    roll_max = equity_series.cummax()
    dd = equity_series / roll_max - 1.0
    max_dd = dd.min()

    mu = daily_returns.mean()
    sd = daily_returns.std(ddof=1)
    sharpe = (mu / sd) * np.sqrt(trading_days) if sd and sd > 0 else np.nan

    calmar = (cagr / abs(max_dd)) if pd.notna(cagr) and pd.notna(max_dd) and max_dd < 0 else np.nan

    return {
        "Equity Start": float(equity_series.iloc[0]),
        "Equity End": float(equity_series.iloc[-1]),
        "CAGR": float(cagr) if pd.notna(cagr) else np.nan,
        "Max Drawdown %": float(max_dd * 100.0) if pd.notna(max_dd) else np.nan,
        "Sharpe (ann.)": float(sharpe) if pd.notna(sharpe) else np.nan,
        "Calmar": float(calmar) if pd.notna(calmar) else np.nan,
        "Avg Daily Return": float(mu) if pd.notna(mu) else np.nan,
        "Daily Vol": float(sd) if pd.notna(sd) else np.nan,
    }

def erc_weights(cov: np.ndarray, tol=1e-10, max_iter=50_000):
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        port_var = w @ cov @ w
        mrc = cov @ w
        rc = w * mrc
        target = port_var / n
        if np.max(np.abs(rc - target)) < tol:
            break
        w = w * (target / np.maximum(rc, 1e-16))
        w = w / w.sum()
    return w

def build_daily_close_returns(market_dfs: dict) -> pd.DataFrame:
    daily_closes = []
    for mkt, df in market_dfs.items():
        s = df["close"].copy()
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        daily = s.resample("1D").last().dropna().rename(mkt)
        daily_closes.append(daily)
    closes_df = pd.concat(daily_closes, axis=1).dropna()
    returns_df = closes_df.pct_change().dropna()
    return returns_df

def build_strategy_daily_returns_per_market(
    market_dfs: dict,
    portfolio_trades: pd.DataFrame,
    start_capital: float,
) -> pd.DataFrame:
    rets = {}
    for mkt, df in market_dfs.items():
        trades_mkt = portfolio_trades[portfolio_trades["Market"] == mkt].copy()
        if trades_mkt.empty:
            continue

        eq, realized_eq, daily_ret, _, _ = build_portfolio_mtm_cash(
            market_dfs={mkt: df},
            trades_df=trades_mkt,
            start_capital=start_capital,
            max_gross_exposure=MAX_GROSS_EXPOSURE,
            target_gross_exposure=TARGET_GROSS_EXPOSURE,
            weights={mkt: 1.0},
        )
        rets[mkt] = daily_ret.rename(mkt)

    returns_df = pd.concat(rets.values(), axis=1).dropna()
    return returns_df

def block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    if block_len < 1:
        raise ValueError("block_len måste vara >= 1")
    out = []
    while len(out) < n:
        start = rng.integers(0, n)
        block = [(start + j) % n for j in range(block_len)]
        out.extend(block)
    return np.array(out[:n], dtype=int)

def metrics_from_daily_returns(daily_returns: pd.Series, start_equity: float = 1.0, trading_days: int = 252) -> dict:
    equity = (1.0 + daily_returns).cumprod() * start_equity
    n_days = (equity.index[-1] - equity.index[0]).days
    years = n_days / 365.25 if n_days > 0 else np.nan
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years and years > 0 else np.nan

    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = dd.min()

    mu = daily_returns.mean()
    sd = daily_returns.std(ddof=1)
    sharpe = (mu / sd) * np.sqrt(trading_days) if sd and sd > 0 else np.nan

    calmar = (cagr / abs(max_dd)) if pd.notna(cagr) and pd.notna(max_dd) and max_dd < 0 else np.nan

    return {
        "CAGR": float(cagr) if pd.notna(cagr) else np.nan,
        "MaxDD": float(max_dd) if pd.notna(max_dd) else np.nan,
        "Sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "Calmar": float(calmar) if pd.notna(calmar) else np.nan,
        "EndEquity": float(equity.iloc[-1]),
    }

def bootstrap_erc_portfolio(
    strategy_returns_df: pd.DataFrame,
    n_iter: int = 5000,
    block_len: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    mkts = list(strategy_returns_df.columns)
    n = len(strategy_returns_df)
    if n < 2 * block_len:
        raise ValueError("För få datapunkter relativt block_len. Sänk block_len eller använd mer data.")

    results = []
    base_index = strategy_returns_df.index

    for k in range(n_iter):
        idx = block_bootstrap_indices(n=n, block_len=block_len, rng=rng)
        sample = strategy_returns_df.iloc[idx].copy()
        sample.index = base_index[:len(sample)]

        cov = sample.cov().values
        w = erc_weights(cov)
        w = np.array(w, dtype=float)
        w = w / w.sum()

        port_ret = pd.Series(sample.values @ w, index=sample.index, name="port_ret")
        m = metrics_from_daily_returns(port_ret)

        row = {"iter": k, "Sharpe": m["Sharpe"], "CAGR": m["CAGR"], "MaxDD": m["MaxDD"], "Calmar": m["Calmar"], "EndEquity": m["EndEquity"]}
        for i, mk in enumerate(mkts):
            row[f"w_{mk}"] = float(w[i])
        results.append(row)

    return pd.DataFrame(results)

def bootstrap_fixed_weights_portfolio(
    strategy_returns_df: pd.DataFrame,   # columns = markets, rows = daily returns
    weights: dict,                       # {"US500":0.33, ...} eller {"US500": w1, ...}
    n_iter: int = 5000,
    block_len: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Moving Block Bootstrap med FASTA vikter.
    För varje iteration:
      1) block-resample av daily strategy returns (per marknad)
      2) portfolio returns = R @ w_fixed
      3) metrics på portfolio returns
    Returnerar DataFrame med metrics + vikter (konstanta) per iteration.
    """
    rng = np.random.default_rng(seed)

    mkts = list(strategy_returns_df.columns)
    n = len(strategy_returns_df)
    if n < 2 * block_len:
        raise ValueError("För få datapunkter relativt block_len. Sänk block_len eller använd mer data.")

    # Bygg viktvektor i kolumnordning + normalisera
    missing = set(mkts) - set(weights.keys())
    if missing:
        raise ValueError(f"weights saknar marknader: {missing}")

    w = np.array([float(weights[m]) for m in mkts], dtype=float)
    s = float(w.sum())
    if s <= 0:
        raise ValueError("weights summerar till 0 eller mindre.")
    w = w / s

    results = []
    base_index = strategy_returns_df.index

    for k in range(n_iter):
        idx = block_bootstrap_indices(n=n, block_len=block_len, rng=rng)

        sample = strategy_returns_df.iloc[idx].copy()
        # behåll datumindex bara för snygg equity; ordningen representerar resamplet
        sample.index = base_index[:len(sample)]

        port_ret = pd.Series(sample.values @ w, index=sample.index, name="port_ret")

        m = metrics_from_daily_returns(port_ret)

        row = {
            "iter": k,
            "Sharpe": m["Sharpe"],
            "CAGR": m["CAGR"],
            "MaxDD": m["MaxDD"],
            "Calmar": m["Calmar"],
            "EndEquity": m["EndEquity"],
        }
        for i, mk in enumerate(mkts):
            row[f"w_{mk}"] = float(w[i])

        results.append(row)

    return pd.DataFrame(results)


def summarize_boot(series: pd.Series) -> dict:
    s = series.dropna()
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p05": float(s.quantile(0.05)),
        "p95": float(s.quantile(0.95)),
    }


def run_fixed_weight_bootstrap_suite(
    strategy_returns_df: pd.DataFrame,
    weights_dicts: dict,   # {"Equal": {...}, "ERC": {...}}
    n_iter: int = 5000,
    block_len: int = 20,
    seed: int = 42,
) -> dict:
    """
    Kör bootstrap för flera fasta vikt-scenarion och printar sammanfattning.
    Returnerar dict: scenario -> bootstrap_df
    """
    out = {}

    for label, w in weights_dicts.items():
        boot = bootstrap_fixed_weights_portfolio(
            strategy_returns_df=strategy_returns_df,
            weights=w,
            n_iter=n_iter,
            block_len=block_len,
            seed=seed,
        )
        out[label] = boot

        print(f"\n--- BOOTSTRAP DISTRIBUTIONS (Fixed Weights: {label}) ---")
        for col in ["Sharpe", "CAGR", "MaxDD", "Calmar", "EndEquity"]:
            print(col, summarize_boot(boot[col]))

        print(f"\n--- WEIGHTS ({label}) ---")
        for mk in strategy_returns_df.columns:
            # konstanta vikter => distribution är degenererad, men vi skriver värdet
            val = float(boot[f"w_{mk}"].iloc[0])
            print(f"{mk}: {val:.4f}")

    return out

def max_drawdown_pct(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())  # negativt tal

# ==========================
# MAIN
# ==========================
market_dfs = {}
all_trades = []

for m in markets:
    name = m["name"]
    df = load_market_df(m["csv"])
    market_dfs[name] = df

    tdf = generate_trades_for_market(
        market_name=name,
        df=df,
        ema_fast_len=20,
        ema_slow_len=250,
        pullback_frac=0.20,
    )

    if not tdf.empty:
        all_trades.append(tdf)

if not all_trades:
    raise RuntimeError("Inga trades genererades för någon marknad.")

portfolio_trades = pd.concat(all_trades, ignore_index=True)

# ==========================
# ERC WEIGHTS (DAILY CLOSE RETURNS)
# ==========================
returns_df = build_daily_close_returns(market_dfs)
cov = returns_df.cov().values
w = erc_weights(cov)
erc_weights_dict = dict(zip(returns_df.columns, w))

print("\n--- ERC weights (based on daily close returns) ---")
for k, v in erc_weights_dict.items():
    print(f"{k}: {v:.4f}")

# 1) Bygg dagliga STRATEGI-returns per marknad (standalone)
strategy_returns_df = build_strategy_daily_returns_per_market(
    market_dfs=market_dfs,
    portfolio_trades=portfolio_trades,
    start_capital=START_CAPITAL,
)

print("\nStrategy returns DF shape:", strategy_returns_df.shape)
print(strategy_returns_df.describe())

# ==========================
# FIXA WEIGHTS: Equal vs ERC
# ==========================
mkts = list(strategy_returns_df.columns)
equal_w = {m: 1.0 / len(mkts) for m in mkts}

# erc_weights_dict har ni redan från er ERC-beräkning
# (om ERC-weights beräknas på andra returns än strategy_returns_df,
#  se till att keys matchar mkts)
erc_w = {m: float(erc_weights_dict[m]) for m in mkts}

boot_results = run_fixed_weight_bootstrap_suite(
    strategy_returns_df=strategy_returns_df,
    weights_dicts={
        "Equal": equal_w,
        "ERC": erc_w,
    },
    n_iter=5000,
    block_len=20,
    seed=42,
)


'''
# ==========================
# EQUAL WEIGHTS (override ERC)
# ==========================
mkts = sorted(market_dfs.keys())  # ["US100", "US30", "US500"] i ditt fall
equal_weights = {m: 1.0 / len(mkts) for m in mkts}

print("\n--- Equal weights ---")
for k in sorted(equal_weights.keys()):
    print(f"{k}: {equal_weights[k]:.4f}")
'''

# ==========================
# FINAL PORTFÖLJ (ERC weights)
# ==========================
equity_series, realized_equity_series, daily_returns, open_pos_series, gross_exposure_series = build_portfolio_mtm_cash(
    market_dfs=market_dfs,
    trades_df=portfolio_trades,
    start_capital=START_CAPITAL,
    max_gross_exposure=MAX_GROSS_EXPOSURE,
    target_gross_exposure=TARGET_GROSS_EXPOSURE,
    weights=erc_weights_dict, #weights=equal_weights,
)

mtm_dd = max_drawdown_pct(equity_series)
closed_dd = max_drawdown_pct(realized_equity_series)

print(f"Max DD% (MTM equity): {mtm_dd*100:.2f}%")
print(f"Max DD% (Closed trades only): {closed_dd*100:.2f}%")

metrics = portfolio_metrics_from_equity(equity_series, daily_returns)

print("\n--- PORTFÖLJ METRICS (CASH, MTM, ERC) ---")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

print("\nSanity: Open positions at end:", int(open_pos_series.iloc[-1]))
print("Avg gross exposure %:", float(gross_exposure_series.mean() * 100.0))
print("Max gross exposure %:", float(gross_exposure_series.max() * 100.0))

# Plots
plt.figure(figsize=(12, 5))
plt.plot(equity_series.index, equity_series.values)
plt.title("Portfolio Equity (Cash MTM, ERC weights) - Mean Reversion")
plt.xlabel("Date")
plt.ylabel("Equity ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(open_pos_series.index, open_pos_series.values)
plt.title("Open Positions")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(gross_exposure_series.index, gross_exposure_series.values)
plt.title("Gross Exposure %")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(equity_series.index, equity_series.values, label="MTM Equity")
plt.plot(realized_equity_series.index, realized_equity_series.values, label="Realized (Closed Trades Only)")
plt.title("Equity Curves: MTM vs Closed Trades Only")
plt.xlabel("Date")
plt.ylabel("Equity ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

boot_equal = boot_results["Equal"]
boot_erc   = boot_results["ERC"]

plt.figure(figsize=(10,4))
plt.hist(boot_equal["Sharpe"].dropna(), bins=50, alpha=0.6, label="Equal")
plt.hist(boot_erc["Sharpe"].dropna(), bins=50, alpha=0.6, label="ERC")
plt.title("Bootstrap distribution of Sharpe (Fixed Weights)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
plt.hist(boot_equal["MaxDD"].dropna(), bins=50, alpha=0.6, label="Equal")
plt.hist(boot_erc["MaxDD"].dropna(), bins=50, alpha=0.6, label="ERC")
plt.title("Bootstrap distribution of Max Drawdown (Fixed Weights)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()