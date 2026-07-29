import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st
from groq import Groq
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. SETUP FREE AI BRAIN (GROQ)
# ==========================================
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Institutional 4-Pillar AI Scanner", layout="wide")

# ==========================================
# 2. RELIABLE MASTER STOCK LIST
# ==========================================
NSE_MASTER_LIST = [
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
# 3. COMPLETE 4-PILLAR TECHNICAL MATRIX ENGINE
# ==========================================
def analyze_single_stock(ticker):
    """Calculates all 4 Pillars: Trend, Momentum, Volatility, and Volume Conviction."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo")
        
        if df.empty or len(df) < 50: # Need at least 50 bars for EMA 50
            return None

        df.columns = [col.capitalize() for col in df.columns]
        
        # --- PILLAR 1: TREND (10, 20, 50 EMAs) ---
        df.ta.ema(length=10, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        
        # --- PILLAR 2: MOMENTUM (RSI & MACD) ---
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # --- PILLAR 3: VOLATILITY & RISK (Bollinger Bands & ATR) ---
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.atr(length=14, append=True)
        
        # Extract latest indicators
        latest = df.iloc[-1]
        price = float(latest['Close'])
        
        ema10 = float(latest.get('EMA_10', price))
        ema20 = float(latest.get('EMA_20', price))
        ema50 = float(latest.get('EMA_50', price))
        
        rsi = float(latest.get('RSI_14', 50))
        
        # Safely extract MACD values
        macd_col = [c for c in df.columns if 'MACD_' in c][0]
        sig_col = [c for c in df.columns if 'MACDs_' in c][0]
        macd_val = float(latest[macd_col])
        sig_val = float(latest[sig_col])
        macd_bullish = macd_val > sig_val
        
        # Bollinger Bands & ATR
        bbl_col = [c for c in df.columns if 'BBL_' in c][0]
        bbu_col = [c for c in df.columns if 'BBU_' in c][0]
        atr_col = [c for c in df.columns if 'ATRr_' in c or 'ATR_' in c][0]
        
        bbl = float(latest[bbl_col])
        bbu = float(latest[bbu_col])
        atr = float(latest[atr_col])
        
        # --- PILLAR 4: VOLUME CONVICTION ---
        vol_avg_20 = df['Volume'].tail(20).mean()
        current_vol = float(latest['Volume'])
        volume_spike = current_vol > (vol_avg_20 * 1.2) # 20% higher than 20-day average
        
        # ==========================================
        # CONFLUENCE SCORING SYSTEM (Out of 5)
        # ==========================================
        score = 0
        if price > ema20 and ema20 > ema50: score += 1  # Strong Macro Uptrend
        if 45 <= rsi <= 68:                  score += 1  # Healthy Momentum Zone
        if macd_bullish:                     score += 1  # Positive MACD Crossover
        if volume_spike:                     score += 1  # Institutional Volume Backing
        if bbl <= price <= bbu:              score += 1  # Trading within healthy volatility bounds

        return {
            "ticker": ticker,
            "price": price,
            "rsi": rsi,
            "score": score,
            "trend": "Strong Uptrend" if price > ema20 > ema50 else "Consolidating/Neutral",
            "macd": "Bullish" if macd_bullish else "Bearish",
            "volume": "High (Institutional Interest)" if volume_spike else "Normal Retail",
            "atr": atr
        }
    except Exception:
        return None

# ==========================================
# 4. AI QUANT STRATEGIST DESK
# ==========================================
def generate_ai_radar_brief(ticker, stats):
    prompt = f"""
    You are the Chief Risk Officer of an algorithmic trading firm. 
    Evaluate the full 4-pillar technical matrix for {ticker}:
    - Market Price: INR {stats['price']:.2f}
    - Trend Alignment: {stats['trend']}
    - RSI Momentum (14): {stats['rsi']:.2f}
    - MACD Crossover: {stats['macd']}
    - Volume Conviction: {stats['volume']}
    - ATR (14-day volatility measure): {stats['atr']:.2f}
    - Confluence Score: {stats['score']}/5
    
    Provide a professional 3-part institutional trade setup brief:
    1. **Confluence Thesis:** Explain why this 4-pillar alignment signals a high-probability BUY entry.
    2. **ATR Stop-Loss Rule:** Suggest an exact Stop-Loss price using 1.5x ATR below the current price (Current Price - 1.5 * ATR).
    3. **Target Projection:** Provide a logical short-term profit target price.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception:
        return "AI panel unavailable for this specific token."

# ==========================================
# 5. STREAMLIT APPLICATION VIEW
# ==========================================
st.title("🛡️ Institutional 4-Pillar AI Stock Radar")
st.write("Calculates Trend (EMAs), Momentum (RSI & MACD), Volatility (B-Bands & ATR), and Volume Conviction across the market.")

st.sidebar.metric("Master Stock List", f"{len(NSE_MASTER_LIST)} Equities")
scan_limit = st.sidebar.slider("Number of stocks to scan:", min_value=10, max_value=len(NSE_MASTER_LIST), value=30, step=5)

if st.button("🚀 Execute 4-Pillar Confluence Scan"):
    target_pool = NSE_MASTER_LIST[:scan_limit]
    
    progress_bar = st.progress(0)
    status = st.empty()
    all_results = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(analyze_single_stock, ticker): ticker for ticker in target_pool}
        
        completed_count = 0
        for future in as_completed(future_to_ticker):
            completed_count += 1
            progress_bar.progress(completed_count / len(target_pool))
            status.text(f"Calculating 4-Pillar Matrices: {completed_count}/{len(target_pool)} completed...")
            
            res = future.result()
            if res:
                all_results.append(res)
                
    status.empty()
    progress_bar.empty()
    
    st.write("---")
    st.header("🎯 Live 4-Pillar Confluence Buy Alerts")
    
    if not all_results:
        st.error("Could not fetch market data. Check your connection.")
    else:
        # Sort by highest confluence score (5/5 down to 0/5)
        all_results.sort(key=lambda x: x['score'], reverse=True)
        top_buys = [s for s in all_results if s['score'] >= 3]
        
        display_list = top_buys if top_buys else all_results[:5]
        
        if not top_buys:
            st.warning("⚠️ Overall market momentum is currently choppy. Showing stocks with the strongest relative 4-pillar scores:")
        else:
            st.success(f"🔥 Flagged **{len(display_list)}** high-confluence setups!")
        
        for asset in display_list:
            with st.container():
                c1, c2 = st.columns([1, 3])
                
                with c1:
                    st.markdown(f"### 📈 **{asset['ticker']}**")
                    st.metric("Price", f"₹{asset['price']:.2f}")
                    st.metric("RSI (14)", f"{asset['rsi']:.2f}")
                    st.caption(f"Trend: {asset['trend']}")
                    st.caption(f"Volume: {asset['volume']}")
                    
                    stars = "⭐" * asset['score']
                    st.subheader(f"Score: {stars} ({asset['score']}/5)")
                    
                with c2:
                    st.markdown("**AI Institutional Trade Brief & Risk Rules:**")
                    with st.spinner(f"Compiling ATR risk rules for {asset['ticker']}..."):
                        brief = generate_ai_radar_brief(asset['ticker'], asset)
                        st.info(brief)
                st.markdown("---")