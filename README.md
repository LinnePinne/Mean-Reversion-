# Mean-Reversion-

Idé - Gå long när pris på något sätt är undervärderat och ha ett jämnviktspris som target, vice versa för shorts.
Vi kan använda RSI, t.ex. köpsignal: RSI < 30 och säljsignal > 70, då kan RSI = 50 vara ett target. Som bekräftelse kommer vi behöva använda en till indikator, EMA(25). Test på lägre timeframes kommer även behöva sessionsfilter, speciellt
om vi handlar index. In sample data på US500 2012 - 2020. Som out of sample data använder vi 2020 - 2025 samt andra marknader. I resultaten separeras longs och shorts eftersom index har en underliggande bullish struktur. Vi kommer också kombinera båda sidorna.
