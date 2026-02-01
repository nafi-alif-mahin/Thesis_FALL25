import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from web3 import Web3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# -- CONFIG PATHS --
CONFIG_FILE = "logs/config.json"
REPUTATION_FILE = "logs/reputation.json"
ACTIVITY_LOG = "logs/activity_log.csv"
NODES_CONFIG = "logs/nodes_config.json" 
RPC_URL = "http://127.0.0.1:8545"
LEDGER_ADDRESS = "0x923E9b1B9c600Eeaf91A39B3bf405f957B5377E6"

# Ensure logs directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# --- NEW FEATURE: INITIALIZE CONFIG FILE IF MISSING ---
if not os.path.exists(CONFIG_FILE):
    default_config = {
        "block_size": 1,
        "tx_rate": 2.0,
        "noise_level": 0.05,
        "iterations": 50
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(default_config, f)

def load_dynamic_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

# Auto-refresh every 3 seconds
st_autorefresh(interval=3000, key="datarefresh")
st.set_page_config(page_title="Financial Security Dashboard", layout="wide")

# 1. WEB3 CONNECTION
w3 = Web3(Web3.HTTPProvider(RPC_URL))

# --- NEW FEATURE: SIDEBAR TUNING ---
st.sidebar.title("⚙️ System Tuning")
st.sidebar.markdown("Modify DB-BOA & Network parameters here.")

current_cfg = load_dynamic_config()

new_block_size = st.sidebar.slider("Block Size (Batch)", 1, 20, int(current_cfg.get("block_size", 1)))
new_tx_rate = st.sidebar.slider("Tx Processing Rate (sec)", 0.5, 10.0, float(current_cfg.get("tx_rate", 2.0)))
new_noise = st.sidebar.slider("DB-BOA Noise (Exploration)", 0.0, 1.0, float(current_cfg.get("noise_level", 0.05)))
new_iters = st.sidebar.slider("DB-BOA Iterations", 10, 500, int(current_cfg.get("iterations", 50)))

if st.sidebar.button("Apply Parameters"):
    updated_config = {
        "block_size": new_block_size,
        "tx_rate": new_tx_rate,
        "noise_level": new_noise,
        "iterations": new_iters
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(updated_config, f)
    st.sidebar.success("Parameters Sent to Engine!")

st.sidebar.markdown("---")
st.sidebar.title("📡 Network Status")
if w3.is_connected():
    st.sidebar.success("✅ Connected to Ganache")
else:
    st.sidebar.error("❌ Ganache Not Connected")

st.sidebar.info("Dashboard v1.0 – Integrated with DB-BOA Leader Selection")

# --- MAIN UI ---
st.title("🛡️ Financial Security System Dashboard")
st.markdown("### ADTCN Fraud Detection + Private Ethereum Consortium Blockchain")

# 3. TABS
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🧠 ADTCN Metrics", "🎖️ Reputation & DB-BOA", "⛓️ Contract Events"])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.header("System Overview")
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.write("**Status:** Running on Ganache (local private chain)")
        st.write(f"**Current Settings:** Block Size: {new_block_size} | Tx Rate: {new_tx_rate}s")
        st.write(f"**Algorithm:** Dynamic Butterfly-Billiards (Iterations: {new_iters})")
        st.write(f"**Contract:** DataLedger at {LEDGER_ADDRESS}")
    
    with col_b:
        try:
            ledger_abi = [{"inputs":[],"name":"txCount","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]
            contract = w3.eth.contract(address=LEDGER_ADDRESS, abi=ledger_abi)
            total_tx = contract.functions.txCount().call()
            st.metric("Blockchain Tx Count", total_tx)
        except:
            st.metric("Blockchain Tx Count", "Error")

# --- TAB 2: ADTCN METRICS ---
with tab2:
    st.header("ADTCN Performance")
    metrics = {"MCC": 0.4065, "Accuracy": 0.7853, "Precision": 0.3551, "Recall": 0.7451, "F1": 0.4810}
    st.table(pd.DataFrame([metrics]).T.rename(columns={0: "Value"}))

    st.subheader("🔍 Live Detection Logs (DB-BOA Integrated)")
    if os.path.exists(ACTIVITY_LOG):
        logs = pd.read_csv(ACTIVITY_LOG)
        st.dataframe(logs.tail(10), use_container_width=True)
        fig_trend = px.line(logs, x=logs.index, y='fraud_prob', title="Real-time Fraud Probability Trend")
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Activity logs will appear here once main_engine.py records them.")

# --- TAB 3: REPUTATION & DB-BOA (MERGED) ---
with tab3:
    st.header("Consortium Leaderboard & DB-BOA Simulation")
    
    # Part 1: Reputation
    st.subheader("🎖️ Node Trust Scores (Step 5)")
    if os.path.exists(REPUTATION_FILE):
        with open(REPUTATION_FILE, 'r') as f:
            rep = json.load(f)
        df_rep = pd.DataFrame(list(rep.items()), columns=["Address", "Reputation Score"]).sort_values("Reputation Score", ascending=False).reset_index(drop=True)
        
        left, right = st.columns([2, 1])
        with left:
            fig = px.bar(df_rep, x='Address', y='Reputation Score', color='Reputation Score', 
                         color_continuous_scale='RdYlGn', title="Current Reputation Rankings")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.dataframe(df_rep.style.format({"Reputation Score": "{:.0f}"}))
    
    st.markdown("---")
    
    # Part 2: DB-BOA Simulation
    st.subheader("🦋 DB-BOA Leader Selection Strategy & Simulation")
    st.latex(r"Fragrance (f) = c \cdot I^a")
    
    if os.path.exists(NODES_CONFIG):
        with open(NODES_CONFIG, 'r') as f:
            nodes = json.load(f)
        
        df_nodes = pd.DataFrame.from_dict(nodes, orient='index').reset_index().rename(columns={'index': 'Address'})
        df_nodes['Total Cost (Fitness)'] = df_nodes['CT'] + df_nodes['CC'] + df_nodes['MS']
        df_nodes = df_nodes.sort_values('Total Cost (Fitness)')

        # NEW FEATURE: Simulation Plot from Code 2
        np.random.seed(42)
        df_nodes['X'] = [np.random.uniform(0, 10) for _ in range(len(df_nodes))]
        df_nodes['Y'] = [np.random.uniform(0, 10) for _ in range(len(df_nodes))]
        leader_node = df_nodes.iloc[0]

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=[leader_node['X']], y=[leader_node['Y']], 
                                   mode='markers+text', name='Selected Leader',
                                   marker=dict(size=25, color='gold', symbol='star'),
                                   text=["Target Leader"], textposition="top center"))
        fig_sim.add_trace(go.Scatter(x=df_nodes['X'], y=df_nodes['Y'], mode='markers',
                                   name='Nodes (Butterflies)',
                                   marker=dict(size=12, color=df_nodes['Total Cost (Fitness)'], 
                                               colorscale='Viridis', showscale=True,
                                               colorbar=dict(title="Fitness"))))
        fig_sim.update_layout(title="Butterfly Search Space (Live Simulation)", template="plotly_dark")
        st.plotly_chart(fig_sim, use_container_width=True)

        # First Code Feature: Efficiency Comparison
        st.write("### 📊 Node Efficiency Comparison")
        col_metrics, col_chart = st.columns([1, 2])
        with col_metrics:
            st.success(f"**Current Preferred Leader:** \n\n {leader_node['Address']}")
            st.metric("Lowest Fitness Score", round(leader_node['Total Cost (Fitness)'], 2))
        with col_chart:
            fig_fitness = px.bar(df_nodes, x='Address', y='Total Cost (Fitness)', 
                                title="Node Fitness Scores (Lower is Better)",
                                color='Total Cost (Fitness)', color_continuous_scale='Viridis')
            st.plotly_chart(fig_fitness, use_container_width=True)

        # First Code Feature: Detail Parameters
        st.markdown("#### Detailed DB-BOA Parameters")
        c1, c2, c3 = st.columns(3)
        with c1: st.plotly_chart(px.bar(df_nodes, x='Address', y='CT', title="Comp. Time (CT)", color_discrete_sequence=['#FFA07A']), use_container_width=True)
        with c2: st.plotly_chart(px.bar(df_nodes, x='Address', y='CC', title="Comm. Cost (CC)", color_discrete_sequence=['#87CEFA']), use_container_width=True)
        with c3: st.plotly_chart(px.bar(df_nodes, x='Address', y='MS', title="Memory Space (MS)", color_discrete_sequence=['#98FB98']), use_container_width=True)
    else:
        st.info("Nodes configuration file not found.")

# --- TAB 4: CONTRACT EVENTS ---
with tab4:
    st.header("Contract Events & Logs")
    if os.path.exists(ACTIVITY_LOG):
        logs = pd.read_csv(ACTIVITY_LOG)
        blockchain_view = logs[logs['decision'] == 'ACCEPT'].tail(15)
        if not blockchain_view.empty:
            st.table(blockchain_view[['timestamp', 'selected_leader', 'tx_hash']])
        else:
            st.info("No transactions stored on-chain yet.")

st.toast("Dashboard Updated!")
