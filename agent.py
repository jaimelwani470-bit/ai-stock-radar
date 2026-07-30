import os
import requests
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
from groq import Groq
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. SETUP FREE AI BRAIN (GROQ)
# ==========================================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "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")
client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Nifty 500 4-Pillar & Pattern Radar", layout="wide")

# ==========================================
# 2. DYNAMIC NIFTY 500 TICKER FETCH
# ==========================================
@st.cache_data(ttl=86400)
def get_nifty500_tickers():
    """Fetches official live list of Nifty 500 companies."""
    try:
        url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            lines = response.text.split('\n')
            tickers = []
            for line in lines[1:]:
                cols = line.split(',')
                if len(cols) >= 3:
                    symbol = cols[2].replace('"', '').strip()
                    if symbol and symbol != "Symbol":
                        tickers.append(f"{symbol}.NS")
            if len(tickers) > 100:
                return sorted(tickers)
    except Exception:
        pass
    
    # Robust Fallback Nifty Liquid Basket
    return [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "TATAMOTORS.NS", 
        "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "AXISBANK.NS", "KOTAKBANK.NS", 
        "HINDUNILVR.NS", "BAJFINANCE.NS", "MARUTI.NS", "ASIANPAINT.NS", "HCLTECH.NS", 
        "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", 
        "TATASTEEL.NS", "COALINDIA.NS", "M&M.NS", "ADANIENT.NS", "ADANIPORTS.NS", 
        "BPCL.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "DIVISLAB.NS", "DRREDDY.NS", 
        "CIPLA.NS", "APOLLOHOSP.NS", "GRASIM.NS", "JSWSTEEL.NS", "TECHM.NS", 
        "WIPRO.NS", "HDFCLIFE.NS", "SBILIFE.NS", "BRITANNIA.NS", "TATACONSUM.NS", 
        "INDUSINDBK.NS", "BAJAJ-AUTO.NS", "SHRIRAMFIN.NS", "TRENT.NS", "BEL.NS", 
        "HAL.NS", "VBL.NS", "DLF.NS", "ZOMATO.NS", "JIOFIN.NS", "PFC.NS", "RECLTD.NS"
    ]

# ==========================================
# 3. PATTERN DETECTION ENGINE
# ==========================================
def detect_candlestick_and_chart_patterns(df):
    """Detects classic candlestick formations and multi-bar price breakouts."""
    patterns = []
    
    if len(df) < 5:
        return patterns
        
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    c_open, c_close, c_high, c_low = curr['Open'], curr['Close'], curr['High'], curr['Low']
    p_open, p_close = prev['Open'], prev['Close']
    
    body = abs(c_close - c_open)
    lower_shadow = min(c_open, c_close) - c_low
    upper_shadow = c_high - max(c_open, c_close)
    
    # 1. Bullish Engulfing
    if p_close < p_open and c_close > c_open and c_close > p_open and c_open < p_close:
        patterns.append("Bullish Engulfing")
        
    # 2. Bearish Engulfing
    if p_close > p_open and c_close < c_open and c_close < p_open and c_open > p_close:
        patterns.append("Bearish Engulfing")
        
    # 3. Hammer (Bullish Reversal)
    if body > 0 and lower_shadow >= 2 * body and upper_shadow <= body * 0.5:
        patterns.append("Hammer")
        
    # 4. Shooting Star (Bearish Reversal)
    if body > 0 and upper_shadow >= 2 * body and lower_shadow <= body * 0.5:
        patterns.append("Shooting Star")
        
    # 5. 20-Day High Breakout (Range Breakout Pattern)
    high_20 = df['High'].iloc[-21:-1].max()
    if c_close > high_20:
        patterns.append("20-Day High Breakout")
        
    # 6. 20-Day Low Breakdown
    low_20 = df['Low'].iloc[-21:-1].min()
    if c_close < low_20:
        patterns.append("20-Day Low Breakdown")
        
    return patterns

# ==========================================
# 4. FULL 4-PILLAR & PATTERN SCANNER
# ==========================================
def analyze_single_stock(ticker):
    """Calculates Trend, Momentum, Volatility, Volume & Chart Patterns."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo")
        
        if df.empty or len(df) < 50:
            return None

        df.columns = [col.capitalize() for col in df.columns]
        
        # --- PILLAR 1: TREND ---
        df.ta.ema(length=10, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        
        # --- PILLAR 2: MOMENTUM ---
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # --- PILLAR 3: VOLATILITY & RISK ---
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.atr(length=14, append=True)
        
        latest = df.iloc[-1]
        price = float(latest['Close'])
        
        ema10 = float(latest.get('EMA_10', price))
        ema20 = float(latest.get('EMA_20', price))
        ema50 = float(latest.get('EMA_50', price))
        rsi = float(latest.get('RSI_14', 50))
        
        # MACD
        macd_col = [c for c in df.columns if 'MACD_' in c][0]
        sig_col = [c for c in df.columns if 'MACDs_' in c][0]
        macd_val = float(latest[macd_col])
        sig_val = float(latest[sig_col])
        macd_bullish = macd_val > sig_val
        
        # ATR & Volatility
        atr_col = [c for c in df.columns if 'ATRr_' in c or 'ATR_' in c][0]
        atr = float(latest[atr_col])
        
        # --- PILLAR 4: VOLUME CONVICTION ---
        vol_avg_20 = df['Volume'].tail(20).mean()
        current_vol = float(latest['Volume'])
        volume_spike = current_vol > (vol_avg_20 * 1.2)
        
        # PATTERN RECOGNITION
        detected_patterns = detect_candlestick_and_chart_patterns(df)
        
        # --- BUY SCORE & SIGNAL ---
        buy_score = 0
        if price > ema20 and ema20 > ema50: buy_score += 1
        if 45 <= rsi <= 68:                  buy_score += 1
        if macd_bullish:                     buy_score += 1
        if volume_spike:                     buy_score += 1
        if any(p in detected_patterns for p in ["Bullish Engulfing", "Hammer", "20-Day High Breakout"]): 
            buy_score += 1

        # --- SELL SCORE & SIGNAL ---
        sell_score = 0
        if price < ema20 and ema20 < ema50: sell_score += 1
        if rsi >= 70 or rsi <= 35:            sell_score += 1
        if not macd_bullish:                 sell_score += 1
        if volume_spike and price < df.iloc[-2]['Close']: sell_score += 1
        if any(p in detected_patterns for p in ["Bearish Engulfing", "Shooting Star", "20-Day Low Breakdown"]): 
            sell_score += 1

        # Determine Signal Classification
        signal = "NEUTRAL"
        if buy_score >= 3 and buy_score > sell_score:
            signal = "BUY"
        elif sell_score >= 3 and sell_score > buy_score:
            signal = "SELL"

        return {
            "ticker": ticker,
            "price": price,
            "rsi": rsi,
            "signal": signal,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "patterns": detected_patterns if detected_patterns else ["No Major Pattern"],
            "trend": "Bullish" if price > ema20 > ema50 else ("Bearish" if price < ema20 < ema50 else "Sideways"),
            "volume": "High (Volume Spike)" if volume_spike else "Normal",
            "atr": atr
        }
    except Exception:
        return None

# ==========================================
# 5. AI QUANT STRATEGIST DESK
# ==========================================
def generate_ai_signal_brief(ticker, stats):
    prompt = f"""
    You are the Senior Trading Desk Manager at a hedge fund.
    Evaluate technical matrix for {ticker}:
    - Signal Type: {stats['signal']}
    - Price: INR {stats['price']:.2f}
    - 14-Day RSI: {stats['rsi']:.2f}
    - Trend Alignment: {stats['trend']}
    - Identified Patterns: {', '.join(stats['patterns'])}
    - Volume: {stats['volume']}
    - 14-Day ATR: {stats['atr']:.2f}
    
    Write a 3-part execution brief:
    1. **Signal Rationale:** Explain why this combination of indicators & patterns triggered a {stats['signal']} alert.
    2. **Stop-Loss Level:** Provide exact Stop-Loss price using 1.5x ATR.
    3. **Target Price:** Provide a short-term target price level based on current volatility.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception:
        return "AI panel brief unavailable."

# ==========================================
# 6. STREAMLIT APPLICATION VIEW
# ==========================================
st.title("🛰️ Nifty 500 AI Pattern & Technical Scanner")
st.write("Scans Nifty 500 equities for Trend, Momentum, Volatility, Volume, and Candlestick/Breakout patterns to emit automated **BUY** and **SELL** signals.")

nifty500_list = get_nifty500_tickers()
st.sidebar.metric("Loaded Nifty Universe", f"{len(nifty500_list)} Stocks")

scan_limit = st.sidebar.slider("Scan Depth (Number of Nifty 500 stocks to scan):", min_value=20, max_value=len(nifty500_list), value=100, step=20)
signal_filter = st.sidebar.radio("Display Signals:", ["ALL", "BUY ONLY", "SELL ONLY"])

if st.button("🚀 Execute Nifty 500 Scan"):
    target_pool = nifty500_list[:scan_limit]
    
    progress_bar = st.progress(0)
    status = st.empty()
    all_results = []
    
    with ThreadPoolExecutor(max_workers=12) as executor:
        future_to_ticker = {executor.submit(analyze_single_stock, ticker): ticker for ticker in target_pool}
        
        completed_count = 0
        for future in as_completed(future_to_ticker):
            completed_count += 1
            progress_bar.progress(completed_count / len(target_pool))
            status.text(f"Analyzing Nifty 500 stocks: {completed_count}/{len(target_pool)} processed...")
            
            res = future.result()
            if res:
                all_results.append(res)
                
    status.empty()
    progress_bar.empty()
    
    st.write("---")
    
    # Filter output based on user selection
    if signal_filter == "BUY ONLY":
        filtered_results = [s for s in all_results if s['signal'] == "BUY"]
    elif signal_filter == "SELL ONLY":
        filtered_results = [s for s in all_results if s['signal'] == "SELL"]
    else:
        filtered_results = [s for s in all_results if s['signal'] in ["BUY", "SELL"]]
        
    st.header("🎯 Live Actionable Trade Signals")
    
    if not filtered_results:
        st.info("No stocks currently match the strict 4-Pillar + Pattern requirements for the selected filter.")
    else:
        # Sort by signal conviction
        filtered_results.sort(key=lambda x: max(x['buy_score'], x['sell_score']), reverse=True)
        st.success(f"🔥 Found **{len(filtered_results)}** high-conviction signal setups across Nifty 500!")
        
        for asset in filtered_results:
            with st.container():
                c1, c2 = st.columns([1, 3])
                
                with c1:
                    st.markdown(f"### 📈 **{asset['ticker']}**")
                    if asset['signal'] == "BUY":
                        st.success("🟢 SIGNAL: BUY")
                    else:
                        st.error("🔴 SIGNAL: SELL")
                        
                    st.metric("Price", f"₹{asset['price']:.2f}")
                    st.metric("RSI (14)", f"{asset['rsi']:.2f}")
                    st.caption(f"Patterns: {', '.join(asset['patterns'])}")
                    st.caption(f"Trend: {asset['trend']} | Volume: {asset['volume']}")
                    
                with c2:
                    st.markdown("**AI Quant Risk & Target Brief:**")
                    with st.spinner(f"Compiling brief for {asset['ticker']}..."):
                        brief = generate_ai_signal_brief(asset['ticker'], asset)
                        st.info(brief)
                st.markdown("---")