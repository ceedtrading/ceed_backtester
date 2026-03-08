import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime, time
import io

# --- SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Ceed Trading: Order Flow Physics Engine", layout="wide")

# AI API SETUP (Alpha Preservation)
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    # Standard production string for the Flash engine
    model = genai.GenerativeModel('gemini-1.5-flash') 
except Exception as e:
    st.error("API Key Error: Check your Streamlit Secrets.")

def run_simulation(df, df_lead, stop_pts, t1_pts, trail_pts, be_trigger_pts, point_val):
    trades = []
    # Physics Gates (Entry Criteria) [cite: 2026-02-17]
    signals = df[(df['F_Buy'] == True) | (df['F_Sell'] == True)].index.tolist()
    
    for idx in signals:
        side = 'Buy' if df.loc[idx, 'F_Buy'] else 'Sell'
        entry_p = df.loc[idx, 'Last']
        entry_time = df.loc[idx, 'dt']
        future = df.loc[idx+1 : idx+200]
        if future.empty: continue
        
        u1_hit, be_activated, peak, exit_p, status = False, False, entry_p, None, "Loss"
        max_favorable, max_adverse = 0, 0
        current_stop = entry_p - stop_pts if side == 'Buy' else entry_p + stop_pts
        
        for i, bar in future.iterrows():
            if side == 'Buy':
                max_favorable = max(max_favorable, bar['High'] - entry_p)
                max_adverse = max(max_adverse, entry_p - bar['Low'])
                if not be_activated and max_favorable >= be_trigger_pts:
                    be_activated, current_stop = True, entry_p
                if bar['Low'] <= current_stop:
                    exit_p, status = current_stop, "BE" if be_activated else "Loss"
                    break
                if not u1_hit and bar['High'] >= entry_p + t1_pts:
                    u1_hit, peak = True, bar['High']
                elif u1_hit:
                    peak = max(peak, bar['High'])
                    if bar['Low'] <= peak - trail_pts:
                        exit_p, status = peak - trail_pts, "Win"
                        break
            else: # Sell Side
                max_favorable = max(max_favorable, entry_p - bar['Low'])
                max_adverse = max(max_adverse, bar['High'] - entry_p)
                if not be_activated and max_favorable >= be_trigger_pts:
                    be_activated, current_stop = True, entry_p
                if bar['High'] >= current_stop:
                    exit_p, status = current_stop, "BE" if be_activated else "Loss"
                    break
                if not u1_hit and bar['Low'] <= entry_p - t1_pts:
                    u1_hit, peak = True, bar['Low']
                elif u1_hit:
                    peak = min(peak, bar['Low'])
                    if bar['High'] >= peak + trail_pts:
                        exit_p, status = peak + trail_pts, "Win"
                        break
        
        if exit_p is not None:
            # Lead Alignment Logic [cite: 2026-03-08]
            lead_context = "N/A"
            if df_lead is not None:
                lead_snap = df_lead[df_lead['dt'] <= entry_time].tail(5)
                if not lead_snap.empty:
                    lead_move = lead_snap['Last'].iloc[-1] - lead_snap['Last'].iloc[0]
                    lead_context = f"{'Bullish' if lead_move > 0 else 'Bearish'} Lead Drift"

            trades.append({
                "Date": df.loc[idx, 'Date'],
                "Weekday": entry_time.strftime('%A'),
                "Hour_Block": entry_time.hour,
                "Timestamp": entry_time, 
                "Side": side, 
                "Status": status, 
                "Lead_Alignment": lead_context,
                "MAE_Pts": max_adverse, 
                "MFE_Pts": max_favorable, 
                "Net_Pts": (exit_p - entry_p if side == 'Buy' else entry_p - exit_p), 
                "Total_$": (exit_p - entry_p if side == 'Buy' else entry_p - exit_p) * point_val
            })
    return pd.DataFrame(trades)

# --- UI FRONTEND ---
st.sidebar.header("Mechanical Controls")
stop_ticks = st.sidebar.number_input("Initial Stop (Ticks)", value=30)
be_trigger = st.sidebar.number_input("BE Trigger (Pts)", value=6.0)
t1_pts = st.sidebar.number_input("Target 1 (Pts)", value=12.0)
trail_pts = st.sidebar.number_input("T2 Trail (Pts)", value=5.0)
point_value = st.sidebar.selectbox("Point Value", options=[50.0, 20.0, 5.0, 2.0])

st.title("Antigravity: Gemini 3 Clean-Pipe Optimizer")
f_file = st.file_uploader("1. Upload Futures Baseline (ES/NQ)", type=['txt', 'csv'])
o_file = st.file_uploader("2. Upload Lead Engine Overlay (NVDA/AAPL)", type=['txt', 'csv'])

df_lead = None
if o_file:
    # Sanitizing lead file [cite: 2026-03-08]
    df_lead = pd.read_csv(o_file, skipinitialspace=True)
    df_lead['dt'] = pd.to_datetime(df_lead['Date'] + ' ' + df_lead['Time'])

if f_file:
    # Explicit mapping for Sierra Chart headers [cite: 2026-03-08]
    raw_data = f_file.getvalue().decode("utf-8").replace("\r", "")
    df = pd.read_csv(io.StringIO(raw_data), skipinitialspace=True)
    df['dt'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Sensors [cite: 2026-02-17]
    df['VWAP_0930'] = (df['Last'] * df['Volume']).cumsum() / df['Volume'].cumsum() 
    df['Range_Pos'] = (df['Last'] - df['LOD']) / (df['HOD'] - df['LOD']).replace(0, np.nan)
    df['Sum_Prev'] = df['Sum'].shift(1)
    
    # Entry Gates [cite: 2026-02-17]
    df['F_Buy'] = (df['Sum_Prev'] < 0) & (df['Sum'] > 0) & (df['dt'].dt.time <= time(15, 45)) & (df['Last'] < df['VWAP_0930']) & (df['Range_Pos'] < 0.25)
    df['F_Sell'] = (df['Sum_Prev'] > 0) & (df['Sum'] < 0) & (df['dt'].dt.time <= time(15, 45)) & (df['Last'] > df['VWAP_0930']) & (df['Range_Pos'] > 0.75)

    if st.button("Run Multi-Asset Synthetic Review"):
        results = run_simulation(df, df_lead, stop_ticks * 0.25, t1_pts, trail_pts, be_trigger, point_value)
        if not results.empty:
            results['Cumulative_Profit'] = results['Total_$'].cumsum()
            st.subheader("Statistical Dashboard")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Profit", f"${results['Total_$'].sum():,.2f}")
            m2.metric("Signals", len(results))
            m3.metric("Win %", f"{(len(results[results['Status']=='Win'])/len(results))*100:.1f}%")
            m4.metric("Avg MAE", f"{results['MAE_Pts'].mean():.2f}")

            st.line_chart(results, x="Timestamp", y="Cumulative_Profit")

            # --- SANITIZED AI BRIDGE ---
            st.divider()
            st.subheader("Antigravity: Gemini 3 Flash Synthetic Review")
            with st.spinner("Analyzing Multi-Asset Order Flow Physics..."):
                # Scrub data to remove all Sierra Chart special characters [cite: 2026-03-08]
                alignment_summary = results.groupby(['Lead_Alignment', 'Status']).size().to_string()
                clean_payload = "".join([c for c in alignment_summary if ord(c) < 128]).replace("\t", " ")
                
                prompt = f"""
                Analyze ES Futures Chassis vs Lead Engine.
                Data Summary: {clean_payload}
                Mission: Identify out-of-sync losses and suggest a 'Refusal to Trade' window.
                """
                
                try:
                    response = model.generate_content(prompt)
                    st.info(response.text)
                except Exception as e:
                    # Capture unredacted error for diagnosis [cite: 2026-03-08]
                    st.error(f"Handshake Logic Failed: {str(e)}")
            
            st.dataframe(results.style.background_gradient(subset=['MAE_Pts'], cmap='Reds'))