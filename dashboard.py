import streamlit as st
import pandas as pd
import os
from PIL import Image
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Traffic Violation Detection System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="color: white; font-size: 2.5em;">🚦 Traffic Violation Detection System</h1>
    <p style="color: #aaa; font-size: 1.1em;">AI-Powered Enforcement Dashboard | Real-time Monitoring</p>
</div>
""", unsafe_allow_html=True)

# Load data
CSV_MAIN   = "violations_log.csv"
CSV_HELMET = "helmet_violations_log.csv"

dfs = []
if os.path.exists(CSV_MAIN):
    df1 = pd.read_csv(CSV_MAIN)
    dfs.append(df1)
if os.path.exists(CSV_HELMET):
    df2 = pd.read_csv(CSV_HELMET)
    dfs.append(df2)

if not dfs:
    st.warning("⚠️ No violations logged yet.")
    st.stop()

df = pd.concat(dfs, ignore_index=True)

# Parse timestamp — handle format: 20260405_170022_641263
def parse_ts(ts):
    try:
        return datetime.strptime(str(ts), "%Y%m%d_%H%M%S_%f")
    except:
        return None

df['timestamp_dt'] = df['timestamp'].apply(parse_ts)
df['hour'] = df['timestamp_dt'].apply(
    lambda x: x.hour if x else None)

# Sidebar
st.sidebar.title("🔧 Control Panel")
st.sidebar.markdown("---")

violation_filter = st.sidebar.selectbox(
    "Filter by Violation Type",
    ["All", "red_light_jump", "helmetless_riding"]
)

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
st.sidebar.metric("Total Records", len(df))
identified = len(df[~df['plate_number'].isin(['N/A', 'UNREAD', ''])])
st.sidebar.metric("Plates Identified", identified)

# Apply filter
filtered_df = df if violation_filter == "All" else \
    df[df["violation_type"] == violation_filter]

# ── Metrics ─────────────────────────────
st.markdown("## 📊 Overview")
col1, col2, col3, col4 = st.columns(4)

total      = len(df)
red_count  = len(df[df.violation_type == "red_light_jump"])
helm_count = len(df[df.violation_type == "helmetless_riding"])

with col1:
    st.metric("🚨 Total Violations", total)
with col2:
    st.metric("🔴 Red Light Jumps", red_count)
with col3:
    st.metric("⛑️ Helmetless Riders", helm_count)
with col4:
    st.metric("🔍 Plates Identified", identified)

st.markdown("---")

# ── Charts ──────────────────────────────
st.markdown("## 📈 Analytics")
col1, col2 = st.columns(2)

with col1:
    vc = df['violation_type'].value_counts()
    fig_pie = px.pie(
        values=vc.values,
        names=vc.index,
        title="Violation Distribution",
        color_discrete_map={
            'red_light_jump':    '#e94560',
            'helmetless_riding': '#f5a623'
        },
        hole=0.4
    )
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    hourly = df[df['hour'].notna()].groupby(
        ['hour', 'violation_type']).size().reset_index(name='count')
    if not hourly.empty:
        fig_bar = px.bar(
            hourly, x='hour', y='count',
            color='violation_type',
            title="Violations by Hour",
            color_discrete_map={
                'red_light_jump':    '#e94560',
                'helmetless_riding': '#f5a623'
            }
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No hourly data available")

st.markdown("---")

# ── Top Violators ────────────────────────
st.markdown("## 🏆 Top Violators")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔴 Red Light Violators")
    red_df = df[
        (df.violation_type == "red_light_jump") &
        (~df.plate_number.isin(['N/A','UNREAD','']))]
    if not red_df.empty:
        top = red_df['plate_number'].value_counts()\
                    .head(10).reset_index()
        top.columns = ['Plate Number', 'Violations']
        st.dataframe(top, use_container_width=True,
                     hide_index=True)
    else:
        st.info("No identified plates yet")

with col2:
    st.subheader("⛑️ Helmet Violators")
    helm_df = df[
        (df.violation_type == "helmetless_riding") &
        (~df.plate_number.isin(['N/A','UNREAD','']))]
    if not helm_df.empty:
        top = helm_df['plate_number'].value_counts()\
                     .head(10).reset_index()
        top.columns = ['Plate Number', 'Violations']
        st.dataframe(top, use_container_width=True,
                     hide_index=True)
    else:
        st.info("No helmet violations with plates yet")

st.markdown("---")

# ── Violation Log ────────────────────────
st.markdown("## 📋 Violation Log")

search = st.text_input("🔍 Search by plate number", "")
if search:
    display_df = filtered_df[
        filtered_df['plate_number'].str.contains(
            search, case=False, na=False)]
else:
    display_df = filtered_df

st.dataframe(
    display_df[['timestamp','violation_type',
                'plate_number','screenshot_path']]
    .sort_values("timestamp", ascending=False)
    .reset_index(drop=True),
    use_container_width=True
)
st.markdown(f"**Showing {len(display_df)} of {len(df)} violations**")

st.markdown("---")

# ── Evidence Screenshots ──────────────────
st.markdown("## 📸 Evidence Screenshots")

valid = display_df[
    display_df["screenshot_path"].apply(
        lambda x: os.path.exists(str(x)))
]["screenshot_path"].tolist()

if valid:
    col1, col2 = st.columns([1, 2])
    with col1:
        selected = st.selectbox("Select evidence:", valid)
        if selected:
            row = display_df[
                display_df['screenshot_path'] == selected
            ].iloc[0]
            st.markdown("### 📄 Details")
            st.markdown(f"**Type:** {row['violation_type']}")
            st.markdown(f"**Plate:** {row['plate_number']}")
            st.markdown(f"**Time:** {row['timestamp']}")
    with col2:
        if selected and os.path.exists(selected):
            img = Image.open(selected)
            st.image(img, caption=selected,
                    use_column_width=True)
else:
    st.info("No screenshots available.")

st.markdown("---")

# ── Download ──────────────────────────────
st.markdown("## ⬇️ Export Report")
col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="📥 Download Full CSV",
        data=df.to_csv(index=False),
        file_name=f"violations_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
with col2:
    st.download_button(
        label="📥 Download Filtered CSV",
        data=filtered_df.to_csv(index=False),
        file_name=f"violations_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666;'>"
    "Traffic Violation Detection System | "
    "YOLOv8 + EasyOCR + Streamlit"
    "</div>",
    unsafe_allow_html=True
)