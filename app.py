import streamlit as st
import numpy as np
import pandas as pd
import random
from ml_inference_sim.simulator import Simulator, Cluster, Request
from ml_inference_sim.hardware import HARDWARE_DB, get_profile

st.set_page_config(page_title="ML Inference Cluster Sim", layout="wide")

st.title("⚡ ML Inference Cluster Simulator")
st.markdown("Simulate cluster capacity, latency, and throughput for Large Language Model workloads.")

# --- Sidebar Configuration ---
st.sidebar.header("1. Hardware Configuration")
hw_choice = st.sidebar.selectbox("Select Hardware Profile", list(HARDWARE_DB.keys()), index=1)
profile = get_profile(hw_choice)

# Allow overrides
ttft = st.sidebar.number_input("Time To First Token (ms)", value=profile.ttft_ms, min_value=1.0)
tps = st.sidebar.number_input("Decode Speed (tokens/s)", value=profile.output_tokens_per_sec, min_value=1.0)
num_devices = st.sidebar.slider("Cluster Size (Number of Devices)", 1, 128, 8)

st.sidebar.header("2. Workload Definition")
rpm = st.sidebar.number_input("Traffic Load (Requests Per Minute)", value=600, step=10)
duration_min = st.sidebar.number_input("Simulation Duration (minutes)", value=1, min_value=1, max_value=60)

col1, col2 = st.sidebar.columns(2)
with col1:
    in_len_mean = st.number_input("Avg Input Tokens", value=512)
    in_len_std = st.number_input("Input StdDev", value=128)
with col2:
    out_len_mean = st.number_input("Avg Output Tokens", value=256)
    out_len_std = st.number_input("Output StdDev", value=64)

st.sidebar.header("3. Quality of Service (SLA)")
max_latency_ms = st.sidebar.number_input("Max Allowed Latency (ms)", value=5000)
enable_drops = st.sidebar.checkbox("Drop Requests exceeding Latency", value=True)

# --- Simulation Logic ---

if st.button("🚀 Run Simulation", type="primary"):
    with st.spinner("Simulating..."):
        # Setup
        cluster = Cluster(num_devices, ttft, tps)
        sim = Simulator(cluster)
        
        # Generate Workload (Poisson)
        num_requests = int(rpm * duration_min)
        rps = rpm / 60.0
        
        requests = []
        current_time = 0.0
        
        for i in range(num_requests):
            inter_arrival = random.expovariate(rps)
            current_time += inter_arrival
            
            # Normal distribution for tokens (clipped to min 1)
            in_tokens = max(1, int(random.gauss(in_len_mean, in_len_std)))
            out_tokens = max(1, int(random.gauss(out_len_mean, out_len_std)))
            
            deadline = None
            if enable_drops:
                deadline = current_time + (max_latency_ms / 1000.0)
                
            req = Request(i, current_time, in_tokens, out_tokens, deadline=deadline)
            requests.append(req)
            
        # Run
        sim.run(requests)
        stats = sim.get_stats()
        
    # --- Results Display ---
    
    # metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Processed Requests", f"{stats['processed']} / {stats['total_requests']}")
    m2.metric("Drop Rate", f"{(stats['dropped'] / stats['total_requests'] * 100):.1f}%", 
              delta_color="inverse" if stats['dropped'] > 0 else "off")
    m3.metric("Throughput (RPM)", f"{stats['throughput_rpm']:.1f}")
    m4.metric("Avg Latency", f"{stats['avg_latency_sec']:.3f} s")
    
    st.divider()
    
    # Detailed Stats
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("latency Distribution")
        st.write(f"**P50**: {stats['p50_latency_sec']:.3f}s | **P90**: {stats['p90_latency_sec']:.3f}s | **P99**: {stats['p99_latency_sec']:.3f}s")
        if stats['processed'] > 0:
            latencies = [r.latency for r in sim.completed_requests if r.status == "COMPLETED"]
            st.bar_chart(pd.DataFrame(latencies, columns=["Latency (s)"]))
        
    with c2:
        st.subheader("Hardware Efficiency")
        utilization = []
        for d in cluster.devices:
            busy_pct = (d.total_busy_time / sim.current_time) * 100 if sim.current_time > 0 else 0
            utilization.append({"Device ID": d.id, "Utilization %": busy_pct})
            
        st.dataframe(pd.DataFrame(utilization), hide_index=True)
        avg_util = sum(u["Utilization %"] for u in utilization) / len(utilization)
        st.info(f"Average Cluster Utilization: **{avg_util:.1f}%**")
        
        if avg_util > 95:
            st.warning("⚠️ High Saturation: Cluster is bottlenecked. Consider adding nodes.")
        elif avg_util < 30:
            st.info("💡 Low Saturation: You might be able to downscale.")

else:
    st.info("Adjust parameters on the left and click Run Simulation.")
