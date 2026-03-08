import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from datetime import datetime, time
import io

# --- SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Ceed Trading: Gemini 3 Physics Engine", layout="wide")

# --- AI SDK INITIALIZATION ---
try:
    # Modern GA SDK
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("API Key missing in Secrets. Simulation will run locally only.")

def run_simulation(df, df_lead, stop_pts, t1_pts, trail_pts, be_trigger_pts, point_val):
    trades = []
    # Identify signals based on Sierra Chart 'Sum'
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
                "Timestamp": entry_time,
                "Hour": entry_time.hour,
                "Side": side, 
                "Status": status, 
                "Lead_Sync": lead_sync,
                "MAE": round(max_adverse, 2), 
                "MFE": round(max_favorable, 2),
                "Net": round((exit_p - entry_p if side == 'Buy' else entry_p - exit_p) * point_val, 2)
            })
    return pd.DataFrame(trades)

# --- UI FRONTEND ---
st.title("Antigravity: Gemini 3 Production Optimizer")

st.sidebar.header("Mechanical Controls")
stop_ticks = st.sidebar.number_input("Initial Stop (Ticks)", value=30)
t1_pts = st.sidebar.number_input("Target 1 (Pts)", value=12.0)
trail_pts = st.sidebar.number_input("T2 Trail (Pts)", value=5.0)
be_trigger = st.sidebar.number_input("BE Trigger (Pts)", value=6.0)
point_value = st.sidebar.selectbox("Point Value", options=[50.0, 20.0, 5.0, 2.0])

f_file = st.file_uploader("1. Upload Futures Baseline (ES/NQ)", type=['txt', 'csv'])
o_file = st.file_uploader("2. Upload Lead Engine Overlay (NVDA/AAPL)", type=['txt', 'csv'])

# SESSION STATE INITIALIZATION
if 'results' not in st.session_state:
    st.session_state.results = None

if f_file:
    # Sierra Chart Scrubbing
    raw_data = f_file.getvalue().decode("utf-8").replace("\r", "")
    df = pd.read_csv(io.StringIO(raw_data), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df['dt'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Physics Gates: Sum
    df['Sum_Prev'] = df['Sum'].shift(1)
    df['F_Buy'] = (df['Sum_Prev'] < 0) & (df['Sum'] > 0) & (df['dt'].dt.time <= time(15, 45))
    df['F_Sell'] = (df['Sum_Prev'] > 0) & (df['Sum'] < 0) & (df['dt'].dt.time <= time(15, 45))

    # PHASE 1: LOCAL RUN
    if st.button("Run Mechanical Simulation"):
        df_lead = None
        if o_file:
            df_lead = pd.read_csv(o_file, skipinitialspace=True)
            df_lead.columns = [c.strip() for c in df_lead.columns]
            df_lead['dt'] = pd.to_datetime(df_lead['Date'] + ' ' + df_lead['Time'])
            
        st.session_state.results = run_simulation(df, df_lead, stop_ticks * 0.25, t1_pts, trail_pts, be_trigger, point_value)

    # DISPLAY SUMMARY BEFORE AI
    if st.session_state.results is not None:
        res = st.session_state.results
        st.subheader("Statistical Performance Summary")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total P/L", f"${res['Net'].sum():,.2f}")
        m2.metric("Trades Generated", len(res))
        win_rate = (len(res[res['Status'] == 'Win']) / len(res)) * 100 if len(res) > 0 else 0
        m3.metric("Win Rate %", f"{win_rate:.1f}%")
        m4.metric("Avg MAE (Heat)", f"{res['MAE'].mean():.2f}")

        st.line_chart(res, x="Timestamp", y="Net")
        st.dataframe(res.style.background_gradient(subset=['MAE'], cmap='Reds'))

        # PHASE 2: OPTIONAL AI OPTIMIZATION
        st.divider()
        if st.button("Request AI Synthetic Review (Gemini 3 Flash)"):
            with st.spinner("Engaging Gemini 3 Flash Engine..."):
                summary = res.groupby(['Hour', 'Lead_Sync', 'Status']).size().to_string()
                clean_payload = "".join(i for i in summary if ord(i) < 128)
                
                try:
                    response = client.models.generate_content(
                        model='gemini-3-flash-preview',
                        contents=f"Analyze these trading physics for Alpha Friction: {clean_payload}. Suggest a 'Refusal to Trade' window."
                    )
                    st.info(response.text)
                except Exception as e:
                    st.error(f"Handshake Failed: {str(e)}")