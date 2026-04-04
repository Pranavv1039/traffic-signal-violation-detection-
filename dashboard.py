import streamlit as st
import pandas as pd
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Traffic Violation Detection System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #e94560;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .violation-badge {
        background-color: #e94560;
        color: white;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 12px;
    }
    .helmet-badge {
        background-color: #f5a623;
        color: white;
        padding: 3px 8px;
        border-radius: 5px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Header
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
    dfs.append(pd.read_csv(CSV_MAIN))
if os.path.exists(CSV_HELMET):
    dfs.append(pd.read_csv(CSV_HELMET))

if not dfs:
    st.warning("⚠️ No violations logged yet.")
    st.info("Run main.py or helmet_images.py to start detecting violations.")
    st.stop()

df = pd.concat(dfs, ignore_index=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S_%f', errors='coerce')

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/traffic-light.png", width=80)
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
st.sidebar.metric("Unique Plates", df['plate_number'].nunique())

# Apply filter
if violation_filter != "All":
    filtered_df = df[df["violation_type"] == violation_filter]
else:
    filtered_df = df

# ── Top Metrics ────────────────────────────────────────────────────────────
st.markdown("## 📊 Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🚨 Total Violations",
        value=len(df),
        delta=f"+{len(df)} detected"
    )

with col2:
    red_count = len(df[df.violation_type == "red_light_jump"])
    st.metric(
        label="🔴 Red Light Jumps",
        value=red_count,
        delta=f"{round(red_count/len(df)*100)}% of total" if len(df) > 0 else "0%"
    )

with col3:
    helmet_count = len(df[df.violation_type == "helmetless_riding"])
    st.metric(
        label="⛑️ Helmetless Riders",
        value=helmet_count,
        delta=f"{round(helmet_count/len(df)*100)}% of total" if len(df) > 0 else "0%"
    )

with col4:
    na_plates = len(df[df.plate_number == "N/A"])
    identified = len(df) - na_plates
    st.metric(
        label="🔍 Plates Identified",
        value=identified,
        delta=f"{round(identified/len(df)*100)}% success" if len(df) > 0 else "0%"
    )

st.markdown("---")

# ── Charts ─────────────────────────────────────────────────────────────────
st.markdown("## 📈 Analytics")
col1, col2 = st.columns(2)

with col1:
    # Pie chart
    violation_counts = df['violation_type'].value_counts()
    fig_pie = px.pie(
        values=violation_counts.values,
        names=violation_counts.index,
        title="Violation Distribution",
        color_discrete_map={
            'red_light_jump': '#e94560',
            'helmetless_riding': '#f5a623'
        },
        hole=0.4
    )
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Bar chart - violations over time
    if df['timestamp'].notna().any():
        df['hour'] = df['timestamp'].dt.hour
        hourly = df.groupby(['hour', 'violation_type']).size().reset_index(name='count')
        fig_bar = px.bar(
            hourly, x='hour', y='count',
            color='violation_type',
            title="Violations by Hour",
            color_discrete_map={
                'red_light_jump': '#e94560',
                'helmetless_riding': '#f5a623'
            }
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Timestamp data not available for chart")

st.markdown("---")

# ── Top Violators ──────────────────────────────────────────────────────────
st.markdown("## 🏆 Top Violators")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔴 Red Light Violators")
    red_df = df[df.violation_type == "red_light_jump"]
    if not red_df.empty:
        top_red = red_df['plate_number'].value_counts().head(5).reset_index()
        top_red.columns = ['Plate Number', 'Violations']
        st.dataframe(top_red, use_container_width=True, hide_index=True)
    else:
        st.info("No red light violations yet")

with col2:
    st.subheader("⛑️ Helmet Violators")
    helmet_df = df[df.violation_type == "helmetless_riding"]
    if not helmet_df.empty:
        top_helmet = helmet_df['plate_number'].value_counts().head(5).reset_index()
        top_helmet.columns = ['Plate Number', 'Violations']
        st.dataframe(top_helmet, use_container_width=True, hide_index=True)
    else:
        st.info("No helmet violations yet")

st.markdown("---")

# ── Violation Log ──────────────────────────────────────────────────────────
st.markdown("## 📋 Violation Log")

# Search
search = st.text_input("🔍 Search by plate number", "")
if search:
    display_df = filtered_df[
        filtered_df['plate_number'].str.contains(search, case=False, na=False)]
else:
    display_df = filtered_df

# Color code violation types
def color_violation(val):
    if val == 'red_light_jump':
        return 'background-color: #3d0000; color: #ff4444'
    elif val == 'helmetless_riding':
        return 'background-color: #3d2000; color: #ff9900'
    return ''

st.dataframe(
    display_df.sort_values("timestamp", ascending=False)
              .reset_index(drop=True),
    use_container_width=True
)

st.markdown(f"**Showing {len(display_df)} of {len(df)} violations**")

st.markdown("---")

# ── Evidence Screenshots ───────────────────────────────────────────────────
st.markdown("## 📸 Evidence Screenshots")

valid_screenshots = display_df[
    display_df["screenshot_path"].apply(
        lambda x: os.path.exists(str(x)))
]["screenshot_path"].tolist()

if valid_screenshots:
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_img = st.selectbox(
            "Select violation evidence:", valid_screenshots)
        if selected_img:
            row = display_df[
                display_df['screenshot_path'] == selected_img].iloc[0]
            st.markdown("### 📄 Violation Details")
            st.markdown(f"**Type:** {row['violation_type']}")
            st.markdown(f"**Plate:** {row['plate_number']}")
            st.markdown(f"**Time:** {row['timestamp']}")

    with col2:
        if selected_img and os.path.exists(selected_img):
            img = Image.open(selected_img)
            st.image(img, caption=f"Evidence: {selected_img}",
                    use_column_width=True)
else:
    st.info("No screenshots available for selected filter.")

st.markdown("---")

# ── Download ───────────────────────────────────────────────────────────────
st.markdown("## ⬇️ Export Report")
col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="📥 Download Full CSV Report",
        data=df.to_csv(index=False),
        file_name=f"violations_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    filtered_csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Report",
        data=filtered_csv,
        file_name=f"filtered_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666;'>"
    "Traffic Violation Detection System | "
    "Built with YOLOv8 + EasyOCR + Streamlit"
    "</div>",
    unsafe_allow_html=True
)