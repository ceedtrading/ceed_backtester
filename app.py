import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from datetime import datetime, time
import io

# --- SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Ceed Trading: Gemini 3 Physics Engine", layout="wide")

# --- MODERN SDK INITIALIZATION ---
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("API Authentication Friction: Verify your Streamlit Secrets.")

def run_simulation(df, df_lead, stop_pts, t1_pts, trail_pts, be_trigger_pts, point_val):
    trades = []
    # Physics Gates: Using Sierra 'Sum' column  3000 Volume_GraphData.txt]
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
            lead_sync = "N/A"
            if df_lead is not None:
                lead_snap = df_lead[df_lead['dt'] <= entry_time].tail(5)
                if not lead_snap.empty:
                    drift = lead_snap['Last'].iloc[-1] - lead_snap['Last'].iloc[0]
                    lead_sync = "Aligned" if (side == 'Buy' and drift > 0) or (side == 'Sell' and drift < 0) else "Friction"

            trades.append({
                "Hour": entry_time.hour,
                "Weekday": entry_time.strftime('%A'),
                "Side": side, 
                "Status": status, 
                "Lead_Sync": lead_sync,
                "MAE": round(max_adverse, 2), 
                "MFE": round(max_favorable, 2),
                "Net": (exit_p - entry_p if side == 'Buy' else entry_p - exit_p) * point_val
            })
    return pd.DataFrame(trades)

# --- UI FRONTEND (RESTORED INPUTS) ---
st.title("Antigravity: Gemini 3 Production Optimizer")

st.sidebar.header("Mechanical Controls")
# Restoring the sensors for stop, target, and trail
stop_ticks = st.sidebar.number_input("Initial Stop (Ticks)", value=30)
t1_pts = st.sidebar.number_input("Target 1 (Pts)", value=12.0)
trail_pts = st.sidebar.number_input("T2 Trail (Pts)", value=5.0)
be_trigger = st.sidebar.number_input("BE Trigger (Pts)", value=6.0)
point_value = st.sidebar.selectbox("Point Value", options=[50.0, 20.0, 5.0, 2.0])

# PRESERVED OVERLAY INPUT FIELDS
f_file = st.file_uploader("1. Upload Futures Baseline (ES/NQ)", type=['txt', 'csv'])
o_file = st.file_uploader("2. Upload Lead Engine Overlay (NVDA/AAPL)", type=['txt', 'csv'])

df_lead = None
if o_file:
    df_lead = pd.read_csv(o_file, skipinitialspace=True)
    df_lead.columns = [c.strip() for c in df_lead.columns]
    df_lead['dt'] = pd.to_datetime(df_lead['Date'] + ' ' + df_lead['Time'])

if f_file:
    raw_data = f_file.getvalue().decode("utf-8").replace("\r", "")
    df = pd.read_csv(io.StringIO(raw_data), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df['dt'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Physics Gates: Cumulative Delta 'Sum'  3000 Volume_GraphData.txt]
    df['Sum_Prev'] = df['Sum'].shift(1)
    df['F_Buy'] = (df['Sum_Prev'] < 0) & (df['Sum'] > 0) & (df['dt'].dt.time <= time(15, 45))
    df['F_Sell'] = (df['Sum_Prev'] > 0) & (df['Sum'] < 0) & (df['dt'].dt.time <= time(15, 45))

    if st.button("Run Gemini 3 Synthetic Review"):
        # Corrected variable passing to the simulation engine
        results = run_simulation(df, df_lead, stop_ticks * 0.25, t1_pts, trail_pts, be_trigger, point_value)
        
        if not results.empty:
            st.dataframe(results)
            
            # --- GEMINI 3 FLASH HANDSHAKE ---
            with st.spinner("Engaging Gemini 3 Flash Engine..."):
                summary = results.groupby(['Hour', 'Lead_Sync', 'Status']).size().to_string()
                clean_payload = "".join(i for i in summary if ord(i) < 128)
                
                try:
                    response = client.models.generate_content(
                        model='gemini-3-flash-preview',
                        contents=f"Analyze these trading physics for Alpha Friction: {clean_payload}"
                    )
                    st.info(response.text)
                except Exception as e:
                    st.error(f"SDK Handshake Failed: {str(e)}")