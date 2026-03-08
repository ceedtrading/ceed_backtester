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
    st.error("API Key missing in Secrets. Simulation will run locally only.")

def run_simulation(df, df_lead, stop_pts, t1_pts, trail_pts, be_trigger_pts, point_val, active_hours, active_days):
    trades = []
    signals = df[(df['F_Buy'] == True) | (df['F_Sell'] == True)].index.tolist()
    
    for idx in signals:
        entry_time = df.loc[idx, 'dt']
        if entry_time.hour not in active_hours: continue
        if entry_time.strftime('%A') not in active_days: continue

        side = 'Buy' if df.loc[idx, 'F_Buy'] else 'Sell'
        entry_p = df.loc[idx, 'Last']
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
                    exit_p = current_stop
                    status = "BE" if be_activated else "Loss"
                    break
                if not u1_hit and bar['High'] >= entry_p + t1_pts:
                    u1_hit, peak = True, bar['High']
                elif u1_hit:
                    peak = max(peak, bar['High'])
                    if bar['Low'] <= peak - trail_pts:
                        exit_p = peak - trail_pts
                        status = "Win"
                        break
            else: # Sell Side
                max_favorable = max(max_favorable, entry_p - bar['Low'])
                max_adverse = max(max_adverse, bar['High'] - entry_p)
                if not be_activated and max_favorable >= be_trigger_pts:
                    be_activated, current_stop = True, entry_p
                if bar['High'] >= current_stop:
                    exit_p = current_stop
                    status = "BE" if be_activated else "Loss"
                    break
                if not u1_hit and bar['Low'] <= entry_p - t1_pts:
                    u1_hit, peak = True, bar['Low']
                elif u1_hit:
                    peak = min(peak, bar['Low'])
                    if bar['High'] >= peak + trail_pts:
                        exit_p = peak + trail_pts
                        status = "Win"
                        break
        
        if exit_p is not None:
            lead_sync = "No_Overlay"
            if df_lead is not None:
                lead_snap = df_lead[df_lead['dt'] <= entry_time].tail(5)
                if not lead_snap.empty:
                    drift = lead_snap['Last'].iloc[-1] - lead_snap['Last'].iloc[0]
                    lead_sync = "Aligned" if (side == 'Buy' and drift > 0) or (side == 'Sell' and drift < 0) else "Friction"

            trades.append({
                "Timestamp": entry_time,
                "Hour": entry_time.hour,
                "Weekday": entry_time.strftime('%A'),
                "Side": side, 
                "Status": status, 
                "Lead_Sync": lead_sync,
                "MAE": round(max_adverse, 2), 
                "MFE": round(max_favorable, 2),
                "Net": round((exit_p - entry_p if side == 'Buy' else entry_p - exit_p) * point_val, 2)
            })
    return pd.DataFrame(trades)

# --- UI FRONTEND ---
st.title("Antigravity: Bimodal Risk Optimizer")

st.sidebar.header("Mechanical Controls")
stop_ticks = st.sidebar.number_input("Initial Stop (Ticks)", value=30)
t1_pts = st.sidebar.number_input("Target 1 (Pts)", value=12.0)
trail_pts = st.sidebar.number_input("T2 Trail (Pts)", value=5.0)
be_trigger = st.sidebar.number_input("BE Trigger (Pts)", value=6.0)
point_value = st.sidebar.selectbox("Point Value", options=[50.0, 20.0, 5.0, 2.0])

st.sidebar.divider()
st.sidebar.header("Temporal Alpha Filters")
hour_options = list(range(0, 17))
active_hours = st.sidebar.multiselect("Select Active Hours", options=hour_options, default=hour_options)
day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
active_days = st.sidebar.multiselect("Select Active Days", options=day_options, default=day_options)

f_file = st.file_uploader("1. Upload Futures Baseline (ES/NQ)", type=['txt', 'csv'])
o_file = st.file_uploader("2. Upload Lead Engine Overlay (NVDA/AAPL)", type=['txt', 'csv'])

if 'results' not in st.session_state:
    st.session_state.results = None

if f_file:
    raw_data = f_file.getvalue().decode("utf-8").replace("\r", "")
    df = pd.read_csv(io.StringIO(raw_data), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df['dt'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    df['Sum_Prev'] = df['Sum'].shift(1)
    df['F_Buy'] = (df['Sum_Prev'] < 0) & (df['Sum'] > 0) & (df['dt'].dt.time <= time(15, 45))
    df['F_Sell'] = (df['Sum_Prev'] > 0) & (df['Sum'] < 0) & (df['dt'].dt.time <= time(15, 45))

    if st.button("Run Mechanical Simulation"):
        df_lead = None
        if o_file:
            df_lead = pd.read_csv(o_file, skipinitialspace=True)
            df_lead.columns = [c.strip() for c in df_lead.columns]
            df_lead['dt'] = pd.to_datetime(df_lead['Date'] + ' ' + df_lead['Time'])
            
        st.session_state.results = run_simulation(df, df_lead, stop_ticks * 0.25, t1_pts, trail_pts, be_trigger, point_value, active_hours, active_days)

    if st.session_state.results is not None:
        res = st.session_state.results
        st.subheader("Statistical Performance Summary")
        
        # --- ENHANCED RISK METRICS ---
        avg_win = res[res['Net'] > 0]['Net'].mean() if not res[res['Net'] > 0].empty else 0
        avg_loss = res[res['Net'] <= 0]['Net'].mean() if not res[res['Net'] <= 0].empty else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total P/L", f"${res['Net'].sum():,.2f}")
        m2.metric("Total Trades", len(res))
        win_rate = (len(res[res['Status'] == 'Win']) / len(res)) * 100 if len(res) > 0 else 0
        m3.metric("Win Rate %", f"{win_rate:.1f}%")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Win $", f"${avg_win:,.2f}", delta_color="normal")
        c2.metric("Avg Loss $", f"${avg_loss:,.2f}", delta_color="inverse")
        c3.metric("Avg MAE", f"{res['MAE'].mean():.2f}")
        c4.metric("Avg MFE", f"{res['MFE'].mean():.2f}")

        st.line_chart(res, x="Timestamp", y="Net")
        st.write("### 📜 Full Trade Log")
        st.dataframe(res.sort_values(by="Timestamp", ascending=False))

        st.divider()
        if st.button("Request AI Bimodal Optimization"):
            with st.spinner("Analyzing Risk Dynamics..."):
                hour_payload = res.groupby(['Hour', 'Lead_Sync']).agg({'Net': 'sum', 'MAE': 'mean', 'Status': 'count'}).reset_index().to_string()
                day_payload = res.groupby(['Weekday', 'Lead_Sync']).agg({'Net': 'sum', 'MAE': 'mean', 'Status': 'count'}).reset_index().to_string()
                
                prompt = f"""
                Act as the Antigravity Synthetic Reviewer using Gemini 3 Flash. 
                Analyze these trading physics. 
                Avg Win: ${avg_win:.2f}, Avg Loss: ${avg_loss:.2f}.
                
                HOURLY DATA:
                {hour_payload}
                
                WEEKDAY DATA:
                {day_payload}

                INSTRUCTIONS:
                1. Provide TABLE 1: HOURLY PERFORMANCE. Sort by Total P/L (Highest to Lowest).
                2. Provide TABLE 2: WEEKDAY PERFORMANCE. Sort by Total P/L (Highest to Lowest).
                3. Columns for both: [Segment, Lead Sync, Win Rate %, Total P/L, Avg MAE, Risk Level].
                4. Compare the 'Avg Win' vs 'Avg Loss' and identify if specific segments are skewing the R:R unfavorably.
                5. Provide one final 'Refusal to Trade' rule.
                """
                
                try:
                    response = client.models.generate_content(model='gemini-3-flash-preview', contents=prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Handshake Failed: {str(e)}")