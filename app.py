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
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("API Handshake Offline.")

def run_simulation(df, df_lead, stop_pts, t1_pts, trail_pts, be_trigger_pts, point_val, active_hours, active_days, slippage_ticks):
    trades = []
    signals = df[(df['F_Buy'] == True) | (df['F_Sell'] == True)].index.tolist()
    slippage_cost = slippage_ticks * 0.25 * point_val
    
    for idx in signals:
        entry_time = df.loc[idx, 'dt']
        if entry_time.hour not in active_hours or entry_time.strftime('%A') not in active_days:
            continue

        side = 'Buy' if df.loc[idx, 'F_Buy'] else 'Sell'
        entry_p = df.loc[idx, 'Last']
        
        # Look ahead up to 500 bars for execution
        future = df.loc[idx+1 : idx+500]
        if future.empty: continue
        
        u1_hit, be_activated, peak, exit_p, status = False, False, entry_p, None, "Loss"
        max_favorable, max_adverse = 0, 0
        current_stop = entry_p - stop_pts if side == 'Buy' else entry_p + stop_pts
        
        for i, bar in future.iterrows():
            # --- PESSIMISTIC EXECUTION GATE ---
            # If a bar hits both the stop AND the target, we assume the LOSS happened first.
            
            if side == 'Buy':
                # 1. Check for Stop/Loss first (Pessimism)
                if bar['Low'] <= current_stop:
                    exit_p = current_stop
                    status = "BE" if be_activated else "Loss"
                    break
                
                # 2. Update Heat/MFE
                max_favorable = max(max_favorable, bar['High'] - entry_p)
                max_adverse = max(max_adverse, entry_p - bar['Low'])
                
                # 3. Check for BE Trigger
                if not be_activated and max_favorable >= be_trigger_pts:
                    be_activated, current_stop = True, entry_p
                
                # 4. Check for Target (T1)
                if not u1_hit and bar['High'] >= entry_p + t1_pts:
                    u1_hit, peak = True, bar['High']
                    if trail_pts == 0:
                        exit_p, status = entry_p + t1_pts, "Win"
                        break
                elif u1_hit:
                    peak = max(peak, bar['High'])
                    if bar['Low'] <= peak - trail_pts:
                        exit_p, status = peak - trail_pts, "Win"
                        break
            
            else: # Sell Side
                # 1. Check for Stop/Loss first
                if bar['High'] >= current_stop:
                    exit_p = current_stop
                    status = "BE" if be_activated else "Loss"
                    break
                
                max_favorable = max(max_favorable, entry_p - bar['Low'])
                max_adverse = max(max_adverse, bar['High'] - entry_p)
                
                if not be_activated and max_favorable >= be_trigger_pts:
                    be_activated, current_stop = True, entry_p
                
                if not u1_hit and bar['Low'] <= entry_p - t1_pts:
                    u1_hit, peak = True, bar['Low']
                    if trail_pts == 0:
                        exit_p, status = entry_p - t1_pts, "Win"
                        break
                elif u1_hit:
                    peak = min(peak, bar['Low'])
                    if bar['High'] >= peak + trail_pts:
                        exit_p, status = peak + trail_pts, "Win"
                        break
        
        if exit_p is not None:
            lead_sync = "No_Overlay"
            if df_lead is not None:
                lead_snap = df_lead[df_lead['dt'] <= entry_time].tail(5)
                if not lead_snap.empty:
                    drift = lead_snap['Last'].iloc[-1] - lead_snap['Last'].iloc[0]
                    lead_sync = "Aligned" if (side == 'Buy' and drift > 0) or (side == 'Sell' and drift < 0) else "Friction"

            raw_net = (exit_p - entry_p if side == 'Buy' else entry_p - exit_p) * point_val
            trades.append({
                "Timestamp": entry_time, "Hour": entry_time.hour, "Weekday": entry_time.strftime('%A'),
                "Side": side, "Status": status, "Lead_Sync": lead_sync,
                "MAE": round(max_adverse, 2), "MFE": round(max_favorable, 2),
                "Net": round(raw_net - slippage_cost, 2)
            })
    return pd.DataFrame(trades)

# --- UI FRONTEND ---
st.title("Antigravity Beta 4.0: Pessimistic Physics")
st.sidebar.header("Mechanical Controls")
stop_ticks = st.sidebar.number_input("Initial Stop (Ticks)", value=30)
t1_pts = st.sidebar.number_input("Target 1 (Pts)", value=12.0, step=0.1)
trail_pts = st.sidebar.number_input("T2 Trail (Pts)", value=5.0, step=0.1)
be_trigger = st.sidebar.number_input("BE Trigger (Pts)", value=6.0)
point_value = st.sidebar.selectbox("Point Value", options=[50.0, 20.0, 5.0, 2.0])
slippage = st.sidebar.number_input("Slippage (Ticks)", value=2)

st.sidebar.divider()
st.sidebar.header("Temporal Filters")
hour_options = list(range(0, 17))
active_hours = st.sidebar.multiselect("Hours", options=hour_options, default=hour_options)
day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
active_days = st.sidebar.multiselect("Days", options=day_options, default=day_options)

f_file = st.file_uploader("Upload Futures Baseline (ES)", type=['txt', 'csv'])
o_file = st.file_uploader("Upload Lead Engine (NVDA)", type=['txt', 'csv'])

if f_file:
    raw_data = f_file.getvalue().decode("utf-8").replace("\r", "")
    df = pd.read_csv(io.StringIO(raw_data), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df['dt'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['Sum_Prev'] = df['Sum'].shift(1)
    df['F_Buy'] = (df['Sum_Prev'] < 0) & (df['Sum'] > 0) & (df['dt'].dt.time <= time(15, 45))
    df['F_Sell'] = (df['Sum_Prev'] > 0) & (df['Sum'] < 0) & (df['dt'].dt.time <= time(15, 45))

    if st.button("Run Simulation"):
        df_lead = None
        if o_file:
            df_lead = pd.read_csv(o_file, skipinitialspace=True)
            df_lead.columns = [c.strip() for c in df_lead.columns]
            df_lead['dt'] = pd.to_datetime(df_lead['Date'] + ' ' + df_lead['Time'])
            
        st.session_state.results = run_simulation(df, df_lead, stop_ticks * 0.25, t1_pts, trail_pts, be_trigger, point_value, active_hours, active_days, slippage)

    if 'results' in st.session_state and st.session_state.results is not None:
        res = st.session_state.results
        st.subheader("Statistical Performance Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Net P/L", f"${res['Net'].sum():,.2f}")
        m2.metric("Total Trades", len(res))
        win_rate = (len(res[res['Status'] == 'Win']) / len(res)) * 100 if len(res) > 0 else 0
        m3.metric("Win Rate %", f"{win_rate:.1f}%")
        st.dataframe(res.sort_values(by="Timestamp", ascending=False))
        
        st.divider()
        if st.button("Request AI Bimodal Optimization"):
            with st.spinner("Analyzing..."):
                h_sum = res.groupby(['Hour', 'Lead_Sync']).agg({'Net': 'sum', 'Status': 'count'}).to_string()
                d_sum = res.groupby(['Weekday', 'Lead_Sync']).agg({'Net': 'sum', 'Status': 'count'}).to_string()
                try:
                    response = client.models.generate_content(model='gemini-3-flash-preview', contents=f"Analyze physics for {h_sum} and {d_sum}. Provide tables sorted by P/L.")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Handshake Failed: {str(e)}")