import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time

# --- SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Ceed Trading: Order Flow Physics Engine", layout="wide")

def run_simulation(df, stop_pts, t1_pts, trail_pts, be_trigger_pts, point_val):
    trades = []
    signals = df[(df['F_Buy'] == True) | (df['F_Sell'] == True)].index.tolist()
    
    for idx in signals:
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
            profit_pts = exit_p - entry_p if side == 'Buy' else entry_p - exit_p
            trades.append({
                "Date": df.loc[idx, 'Date'], "Timestamp": df.loc[idx, 'dt'], 
                "Side": side, "Status": status, "MAE_Pts": max_adverse, 
                "MFE_Pts": max_favorable, "Net_Pts": profit_pts, "Total_$": profit_pts * point_val
            })
    return pd.DataFrame(trades)

# --- UI FRONTEND ---
st.sidebar.header("Mechanical Controls")
stop_ticks = st.sidebar.number_input("Initial Stop (Ticks)", value=30)
be_trigger = st.sidebar.number_input("BE Trigger (Points MFE)", value=6.0)
t1_pts = st.sidebar.number_input("Target 1 (Points)", value=12.0)
trail_pts = st.sidebar.number_input("T2 Trail (Points)", value=5.0)
point_value = st.sidebar.selectbox("Instrument Point Value", options=[50.0, 20.0, 5.0, 2.0], help="ES=50, NQ=20, MES=5, MNQ=2")

# --- DATA INGESTION ---
st.title("Antigravity: Mechanical Backtester")
f_file = st.file_uploader("Upload Futures Baseline (ES/NQ)", type=['txt', 'csv'])
o_file = st.file_uploader("Upload Lead Engine Overlay (NVDA/AAPL)", type=['txt', 'csv'], help="Required for AI-Optimization/Lead-Lag analysis")

if f_file:
    df = pd.read_csv(f_file, skipinitialspace=True)
    df['dt'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Sequence Fix: Define Sensors BEFORE Strategy Application
    df['VWAP_0930'] = (df['Last'] * df['Volume']).cumsum() / df['Volume'].cumsum() 
    df['Range_Pos'] = (df['Last'] - df['LOD']) / (df['HOD'] - df['LOD']).replace(0, np.nan)
    df['Sum_Prev'] = df['Sum'].shift(1)
    
    # Apply Filters (Core Strategy)
    df['Valid_Window'] = df['dt'].dt.time <= time(15, 45)
    df['F_Buy'] = (df['Sum_Prev'] < 0) & (df['Sum'] > 0) & df['Valid_Window'] & (df['Last'] < df['VWAP_0930']) & (df['Range_Pos'] < 0.25)
    df['F_Sell'] = (df['Sum_Prev'] > 0) & (df['Sum'] < 0) & df['Valid_Window'] & (df['Last'] > df['VWAP_0930']) & (df['Range_Pos'] > 0.75)

    if st.button("Run Mechanical Backtest"):
        results = run_simulation(df, stop_ticks * 0.25, t1_pts, trail_pts, be_trigger, point_value)
        
        if not results.empty:
            # --- EQUITY CURVE ---
            results['Cumulative_Profit'] = results['Total_$'].cumsum()
            st.subheader("Equity Curve: Performance Trajectory")
            st.line_chart(results, x="Timestamp", y="Cumulative_Profit")

            # --- FINANCIAL & STATISTICAL DASHBOARD ---
            st.subheader("Statistical Significance & Financial Summary")
            total_signals = len(results)
            unique_days = results['Date'].nunique()
            wins = len(results[results['Status'] == "Win"])
            losses = len(results[results['Status'] == "Loss"])
            total_profit = results['Total_$'].sum()
            win_percent = (wins / total_signals) * 100
            
            # Metrics Grid
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Net Profit", f"${total_profit:,.2f}", delta=f"{win_percent:.1f}% Win Rate")
            m2.metric("Total Signals", total_signals)
            m3.metric("Trading Days", unique_days)
            m4.metric("Win/Loss Ratio", f"{wins/losses:.2f}" if losses > 0 else "∞")

           # --- AI OPTIMIZATION BRIDGE ---
            st.divider()
            st.subheader("Antigravity: Synthetic Optimizer Bridge")
            
            with st.expander("Generate AI Optimization Prompt"):
                # Data for the AI to analyze
                avg_mae = results['MAE_Pts'].mean()
                avg_mfe = results['MFE_Pts'].mean()
                max_drawdown = results['Cumulative_Profit'].diff().min()
                
                ai_prompt = f"""
                Act as a Senior Quant Analyst specializing in Order Flow Physics.
                
                BACKTEST METRICS:
                - Total Profit: ${total_profit:,.2f}
                - Win Rate: {win_percent:.1f}%
                - Total Signals: {total_signals}
                - Avg Heat (MAE): {avg_mae:.2f} pts
                - Avg Expansion (MFE): {avg_mfe:.2f} pts
                - Current Stop: {stop_ticks * 0.25} pts
                - Target 1: {t1_pts} pts
                - BE Trigger: {be_trigger} pts
                
                OBSERVATION: 
                Review the relationship between MAE and MFE. 
                If MAE is consistently high (>70% of Stop), suggest an Entry Offset.
                If MFE is consistently >2x Target 1, suggest a wider T1 or aggressive T2.
                Provide 3 specific structural tuning adjustments to preserve alpha.
                """
                st.text_area("Copy this to Gemini 3 Pro:", ai_prompt, height=300)

 # --- TRADE LOG ---
            st.subheader("Tactical Outcome: Professional Tabular Format")
            st.dataframe(results.style.background_gradient(subset=['MAE_Pts'], cmap='Reds')
                                         .background_gradient(subset=['MFE_Pts'], cmap='Greens')
                                         .format({"Total $": "${:,.2f}", "Net_Pts": "{:.2f}"}))
            
            st.download_button("Download CSV Report", results.to_csv(index=False).encode('utf-8'), "Ceed_Report.csv", "text/csv")
        else:
            st.warning("No signals passed the structural filters. Adjust your constraints.")