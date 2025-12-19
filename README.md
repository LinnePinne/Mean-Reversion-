# Mean-Reversion-

Idé - Gå long när Close < EMA_fast(5,10,15,20) och Close > EMA_slow(100,200) samt en deep bullback där Close < ((0.2 * (high-low)) + low), alltså pris stängde i botten av 20% daily range. Exit när Close > EMA-fast In sample data på US500 2012 - 2020. Som out of sample data använder vi 2020 - 2025 samt andra marknader. I resultaten separeras longs och shorts eftersom index har en underliggande bullish struktur. Vi kommer också kombinera båda sidorna.
