import streamlit as st
import pandas as pd
import os
import re
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

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #0f0f1a;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    .metric-container {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #e94560;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 3em;
        font-weight: bold;
        color: #e94560;
    }
    .metric-label {
        font-size: 1em;
        color: #aaa;
        margin-top: 5px;
    }
    .section-header {
        color: white;
        border-left: 4px solid #e94560;
        padding-left: 15px;
        margin: 20px 0 10px 0;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }
    .stSelectbox label {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


def clean_plate(plate):
    """Clean and normalize plate numbers."""
    if not isinstance(plate, str):
        return plate
    p = plate.upper().strip()
    # Remove noise words
    for noise in ['MESN','NESO','YAMAHA','VAMAHA','PLATE DETECTED',
                  'DETECTED','UNREAD']:
        p = p.replace(noise, '').strip()
    # Fix OCR errors
    p = re.sub(r'IO(?=\d)', '10', p)
    p = re.sub(r'I0(?=\d)', '10', p)
    # Normalize spaces
    p = ' '.join(p.split())

    # Try to extract valid plate pattern
    # Indian: KL 10 AZ 8471 or KL 10AZ 8471
    m = re.search(
        r'([A-Z]{2})\s*(\d{1,2})\s*([A-Z]{1,3})\s*(\d{3,4})', p)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)}"

    # Simple: 2 letters + 3-4 numbers (YB 6433, NN 773)
    m = re.search(r'([A-Z]{2})\s*([0-9]{3,4})', p)
    if m:
        return f"{m.group(1)} {m.group(2)}"

    # Vietnamese: 30-N1 9500
    m = re.search(r'(\d{2}[-\s][A-Z]\d[\s-]?\d{4})', p)
    if m:
        return m.group(1)

    return p if len(p) > 2 else 'N/A'


# ── Load Data ────────────────────────────────────────────────
CSV_MAIN   = "violations_log.csv"
CSV_HELMET = "helmet_violations_log.csv"

dfs = []
if os.path.exists(CSV_MAIN):
    dfs.append(pd.read_csv(CSV_MAIN))
if os.path.exists(CSV_HELMET):
    dfs.append(pd.read_csv(CSV_HELMET))

if not dfs:
    st.warning("⚠️ No violations logged yet.")
    st.info("Run main.py or helmet_images.py to detect violations.")
    st.stop()

df = pd.concat(dfs, ignore_index=True)

# Clean plates
df['plate_clean'] = df['plate_number'].apply(clean_plate)

# Parse timestamp
def parse_ts(ts):
    try:
        return datetime.strptime(str(ts), "%Y%m%d_%H%M%S_%f")
    except:
        return None

df['timestamp_dt'] = df['timestamp'].apply(parse_ts)
df['hour']         = df['timestamp_dt'].apply(
    lambda x: x.hour if x else None)
df['date']         = df['timestamp_dt'].apply(
    lambda x: x.strftime('%Y-%m-%d') if x else None)


# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/color/96/traffic-light.png", width=70)
st.sidebar.title("🔧 Control Panel")
st.sidebar.markdown("---")

violation_filter = st.sidebar.selectbox(
    "📂 Filter by Violation Type",
    ["All", "red_light_jump", "helmetless_riding"]
)

plate_search = st.sidebar.text_input("🔍 Search Plate Number", "")

if st.sidebar.button("🔄 Refresh Dashboard"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Summary")
st.sidebar.metric("Total Violations", len(df))
st.sidebar.metric("Unique Plates",
                  df[~df['plate_clean'].isin(
                      ['N/A',''])]['plate_clean'].nunique())
st.sidebar.metric("Red Light Jumps",
                  len(df[df.violation_type == "red_light_jump"]))
st.sidebar.metric("Helmet Violations",
                  len(df[df.violation_type == "helmetless_riding"]))


# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
            padding: 25px; border-radius: 12px;
            text-align: center; margin-bottom: 25px;
            border: 1px solid #e94560;">
    <h1 style="color: white; font-size: 2.2em; margin: 0;">
        🚦 Traffic Violation Detection System
    </h1>
    <p style="color: #aaa; margin: 8px 0 0 0;">
        AI-Powered Enforcement Dashboard &nbsp;|&nbsp;
        YOLOv8 + EasyOCR + Streamlit
    </p>
</div>
""", unsafe_allow_html=True)


# ── Metrics ──────────────────────────────────────────────────
st.markdown("## 📊 Overview")
col1, col2, col3, col4 = st.columns(4)

total      = len(df)
red_count  = len(df[df.violation_type == "red_light_jump"])
helm_count = len(df[df.violation_type == "helmetless_riding"])
identified = len(df[~df['plate_clean'].isin(['N/A','','UNREAD'])])

metrics = [
    ("🚨", "Total Violations",    total,      "#e94560"),
    ("🔴", "Red Light Jumps",     red_count,  "#ff4444"),
    ("⛑️", "Helmetless Riders",   helm_count, "#f5a623"),
    ("🔍", "Plates Identified",   identified, "#00b4d8"),
]

for col, (icon, label, value, color) in zip(
        [col1, col2, col3, col4], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-container">
            <div style="font-size:2em">{icon}</div>
            <div class="metric-value" style="color:{color}">
                {value}
            </div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")


# ── Apply Filters ────────────────────────────────────────────
filtered_df = df.copy()
if violation_filter != "All":
    filtered_df = filtered_df[
        filtered_df["violation_type"] == violation_filter]
if plate_search:
    filtered_df = filtered_df[
        filtered_df['plate_clean'].str.contains(
            plate_search, case=False, na=False)]


# ── Charts ───────────────────────────────────────────────────
st.markdown("## 📈 Analytics")
col1, col2 = st.columns(2)

with col1:
    vc = df['violation_type'].value_counts()
    labels = {'red_light_jump': 'Red Light Jump',
              'helmetless_riding': 'Helmetless Riding'}
    fig_pie = px.pie(
        values=vc.values,
        names=[labels.get(n, n) for n in vc.index],
        title="Violation Distribution",
        color_discrete_sequence=['#e94560', '#f5a623'],
        hole=0.45
    )
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_size=16,
        legend=dict(font=dict(color='white'))
    )
    fig_pie.update_traces(textfont_color='white')
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    hourly = df[df['hour'].notna()].groupby(
        ['hour', 'violation_type']).size().reset_index(name='count')
    if not hourly.empty:
        hourly['violation_type'] = hourly['violation_type'].map(
            labels).fillna(hourly['violation_type'])
        fig_bar = px.bar(
            hourly, x='hour', y='count',
            color='violation_type',
            title="Violations by Hour of Day",
            color_discrete_sequence=['#e94560', '#f5a623'],
            barmode='group'
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_size=16,
            xaxis=dict(gridcolor='#333', title='Hour'),
            yaxis=dict(gridcolor='#333', title='Count'),
            legend=dict(font=dict(color='white'))
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No hourly data available")

st.markdown("---")


# ── Top Violators ────────────────────────────────────────────
st.markdown("## 🏆 Top Violators")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔴 Red Light Violators")
    red_df = df[
        (df.violation_type == "red_light_jump") &
        (~df.plate_clean.isin(['N/A', '', 'UNREAD']))]
    if not red_df.empty:
        top = red_df['plate_clean'].value_counts()\
                    .head(10).reset_index()
        top.columns = ['Plate Number', 'Violations']
        top.index   = top.index + 1
        st.dataframe(top, use_container_width=True)
    else:
        st.info("No identified plates yet")

with col2:
    st.markdown("### ⛑️ Helmet Violators")
    helm_df = df[
        (df.violation_type == "helmetless_riding") &
        (~df.plate_clean.isin(['N/A', '', 'UNREAD']))]
    if not helm_df.empty:
        top = helm_df['plate_clean'].value_counts()\
                     .head(10).reset_index()
        top.columns = ['Plate Number', 'Violations']
        top.index   = top.index + 1
        st.dataframe(top, use_container_width=True)
    else:
        st.info("No helmet violations with plates yet")

st.markdown("---")


# ── Violation Log ────────────────────────────────────────────
st.markdown("## 📋 Violation Log")

display_df = filtered_df[
    ['timestamp', 'violation_type',
     'plate_clean', 'screenshot_path']
].copy()
display_df.columns = [
    'Timestamp', 'Violation Type', 'Plate Number', 'Screenshot']
display_df = display_df.sort_values(
    'Timestamp', ascending=False).reset_index(drop=True)
display_df.index = display_df.index + 1

# Color violation type
def color_type(val):
    if val == 'red_light_jump':
        return 'color: #ff4444'
    elif val == 'helmetless_riding':
        return 'color: #f5a623'
    return ''

st.dataframe(
    display_df.style.applymap(
        color_type, subset=['Violation Type']),
    use_container_width=True,
    height=400
)
st.markdown(
    f"**Showing {len(display_df)} of {len(df)} violations**")

st.markdown("---")


# ── Evidence Screenshots ──────────────────────────────────────
st.markdown("## 📸 Evidence Screenshots")

valid = filtered_df[
    filtered_df["screenshot_path"].apply(
        lambda x: os.path.exists(str(x)))
]["screenshot_path"].tolist()

if valid:
    col1, col2 = st.columns([1, 2])
    with col1:
        selected = st.selectbox(
            "Select violation evidence:", valid)
        if selected:
            row = filtered_df[
                filtered_df['screenshot_path'] == selected
            ].iloc[0]
            st.markdown("### 📄 Violation Details")
            st.markdown(f"**🕐 Time:** `{row['timestamp']}`")
            st.markdown(
                f"**⚠️ Type:** `{row['violation_type']}`")
            st.markdown(
                f"**🚗 Plate:** `{row['plate_clean']}`")
            st.markdown(
                f"**📁 File:** `{row['screenshot_path']}`")
    with col2:
        if selected and os.path.exists(selected):
            img = Image.open(selected)
            st.image(img,
                     caption=f"Evidence: {selected}",
                     use_column_width=True)
else:
    st.info(
        "No screenshots found locally. "
        "Run detection to generate evidence.")

st.markdown("---")


# ── Download ──────────────────────────────────────────────────
st.markdown("## ⬇️ Export Reports")
col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        label="📥 Full Report (CSV)",
        data=df.to_csv(index=False),
        file_name=f"all_violations_"
                  f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
with col2:
    st.download_button(
        label="📥 Red Light Only",
        data=df[df.violation_type=='red_light_jump']\
            .to_csv(index=False),
        file_name="red_light_violations.csv",
        mime="text/csv",
        use_container_width=True
    )
with col3:
    st.download_button(
        label="📥 Helmet Only",
        data=df[df.violation_type=='helmetless_riding']\
            .to_csv(index=False),
        file_name="helmet_violations.csv",
        mime="text/csv",
        use_container_width=True
    )

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#555; padding:10px;'>
    🚦 Traffic Violation Detection System &nbsp;|&nbsp;
    Built with YOLOv8 + EasyOCR + Streamlit &nbsp;|&nbsp;
    Pranav Gedela
</div>
""", unsafe_allow_html=True)