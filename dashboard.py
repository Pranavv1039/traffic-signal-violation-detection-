import streamlit as st
import pandas as pd
import os
import re
from PIL import Image
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title='Traffic Violation Detection', page_icon='🚦', layout='wide')

st.markdown('''
<style>
[data-testid="stAppViewContainer"] { background: #0a0a14; }
[data-testid="stSidebar"] { background: #111122; border-right: 1px solid #e94560; }
.card { background: linear-gradient(135deg,#1a1a2e,#16213e); border:1px solid #333; border-radius:12px; padding:20px; text-align:center; }
.card-red { border-color:#e94560 !important; }
.card-orange { border-color:#f5a623 !important; }
.card-blue { border-color:#00b4d8 !important; }
.card-green { border-color:#06d6a0 !important; }
.card-val { font-size:2.8em; font-weight:900; }
.card-lbl { font-size:0.85em; color:#888; margin-top:4px; letter-spacing:1px; text-transform:uppercase; }
h1,h2,h3 { color: white !important; }
div[data-testid="stDataFrame"] { border-radius:10px; }
</style>
''', unsafe_allow_html=True)

def clean_plate(p):
    if not isinstance(p, str): return 'N/A'
    p = p.upper().strip()
    for n in ['MESN','NESO','YAMAHA','VAMAHA','PLATE DETECTED','DETECTED','UNREAD']:
        p = p.replace(n,'').strip()
    p = re.sub(r'IO(?=\d)','10',p)
    p = re.sub(r'I0(?=\d)','10',p)
    p = ' '.join(p.split())
    m = re.search(r'([A-Z]{2})\s*(\d{1,2})\s*([A-Z]{1,3})\s*(\d{3,4})',p)
    if m: return f'{m.group(1)} {m.group(2)} {m.group(3)} {m.group(4)}'
    m = re.search(r'([A-Z]{2})\s*([0-9]{3,4})',p)
    if m: return f'{m.group(1)} {m.group(2)}'
    m = re.search(r'(\d{2}[-\s][A-Z]\d[\s-]?\d{4})',p)
    if m: return m.group(1)
    return p if len(p)>2 else 'N/A'

def parse_ts(ts):
    try: return datetime.strptime(str(ts),'%Y%m%d_%H%M%S_%f')
    except: return None

dfs=[]
for f in ['violations_log.csv','helmet_violations_log.csv']:
    if os.path.exists(f): dfs.append(pd.read_csv(f))

if not dfs:
    st.warning('No violations logged yet. Run detection first.')
    st.stop()

df=pd.concat(dfs,ignore_index=True)
df['plate_clean']=df['plate_number'].apply(clean_plate)
df['timestamp_dt']=df['timestamp'].apply(parse_ts)
df['hour']=df['timestamp_dt'].apply(lambda x: x.hour if x else None)

# Sidebar
with st.sidebar:
    st.markdown('<h2 style="color:#e94560">🚦 TVD System</h2>', unsafe_allow_html=True)
    st.markdown('---')
    vf=st.selectbox('Filter Violation Type',['All','red_light_jump','helmetless_riding'])
    ps=st.text_input('🔍 Search Plate','')
    st.markdown('---')
    st.markdown('<div style="color:#888;font-size:0.8em">SYSTEM STATS</div>', unsafe_allow_html=True)
    st.metric('Total Records', len(df))
    st.metric('Unique Plates', df[~df.plate_clean.isin(['N/A',''])].plate_clean.nunique())
    st.markdown('---')
    if st.button('🔄 Refresh',use_container_width=True): st.rerun()
    st.markdown('<div style="color:#555;font-size:0.75em;margin-top:20px">Built by Pranav Gedela<br>YOLOv8 + EasyOCR + Streamlit</div>', unsafe_allow_html=True)

# Header
st.markdown('''
<div style="background:linear-gradient(135deg,#0f0f1a,#1a1a3e);padding:30px;border-radius:16px;text-align:center;border:1px solid #e94560;margin-bottom:25px">
<h1 style="color:white;font-size:2.4em;margin:0;letter-spacing:2px">🚦 TRAFFIC VIOLATION DETECTION SYSTEM</h1>
<p style="color:#888;margin:10px 0 0 0;font-size:1.05em">AI-Powered Real-Time Enforcement Dashboard &nbsp;•&nbsp; YOLOv8 + EasyOCR</p>
</div>
''', unsafe_allow_html=True)

total=len(df)
red=len(df[df.violation_type=='red_light_jump'])
helm=len(df[df.violation_type=='helmetless_riding'])
ident=len(df[~df.plate_clean.isin(['N/A','','UNREAD'])])

c1,c2,c3,c4=st.columns(4)
for col,icon,val,lbl,cls in zip([c1,c2,c3,c4],
    ['🚨','🔴','⛑️','🔍'],
    [total,red,helm,ident],
    ['TOTAL VIOLATIONS','RED LIGHT JUMPS','HELMET VIOLATIONS','PLATES IDENTIFIED'],
    ['card-red','card-red','card-orange','card-blue']):
    with col:
        st.markdown(f'<div class="card {cls}"><div class="card-val">{val}</div><div class="card-lbl">{icon} {lbl}</div></div>', unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

# Charts
st.markdown('## 📈 Analytics')
c1,c2=st.columns(2)

with c1:
    vc=df.violation_type.value_counts()
    lmap={'red_light_jump':'Red Light Jump','helmetless_riding':'Helmet Violation'}
    fig=px.pie(values=vc.values, names=[lmap.get(n,n) for n in vc.index],
        title='Violation Distribution', hole=0.5,
        color_discrete_sequence=['#e94560','#f5a623'])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',font_color='white',
        title_font_size=16,legend=dict(font=dict(color='white')))
    fig.update_traces(textfont_color='white',textinfo='percent+label')
    st.plotly_chart(fig,use_container_width=True)

with c2:
    hourly=df[df.hour.notna()].groupby(['hour','violation_type']).size().reset_index(name='count')
    if not hourly.empty:
        hourly['violation_type']=hourly.violation_type.map(lmap).fillna(hourly.violation_type)
        fig=px.bar(hourly,x='hour',y='count',color='violation_type',
            title='Violations by Hour of Day',barmode='group',
            color_discrete_sequence=['#e94560','#f5a623'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',title_font_size=16,
            xaxis=dict(gridcolor='#222',title='Hour of Day'),
            yaxis=dict(gridcolor='#222',title='Count'),
            legend=dict(font=dict(color='white')))
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.info('No hourly data available')

st.markdown('---')

# Top Violators
st.markdown('## 🏆 Top Violators')
c1,c2=st.columns(2)

with c1:
    st.markdown('### 🔴 Red Light Violators')
    rdf=df[(df.violation_type=='red_light_jump')&(~df.plate_clean.isin(['N/A','','UNREAD']))]
    if not rdf.empty:
        top=rdf.plate_clean.value_counts().head(10).reset_index()
        top.columns=['Plate Number','Violations']
        top.index=top.index+1
        st.dataframe(top,use_container_width=True)
    else:
        st.info('No identified plates yet')

with c2:
    st.markdown('### ⛑️ Helmet Violators')
    hdf=df[(df.violation_type=='helmetless_riding')&(~df.plate_clean.isin(['N/A','','UNREAD']))]
    if not hdf.empty:
        top=hdf.plate_clean.value_counts().head(10).reset_index()
        top.columns=['Plate Number','Violations']
        top.index=top.index+1
        st.dataframe(top,use_container_width=True)
    else:
        st.info('No helmet violations with plates yet')

st.markdown('---')

# Violation Log
st.markdown('## 📋 Violation Log')
fdf=df.copy()
if vf!='All': fdf=fdf[fdf.violation_type==vf]
if ps: fdf=fdf[fdf.plate_clean.str.contains(ps,case=False,na=False)]

show=fdf[['timestamp','violation_type','plate_clean','screenshot_path']].copy()
show.columns=['Timestamp','Violation Type','Plate Number','Screenshot']
show=show.sort_values('Timestamp',ascending=False).reset_index(drop=True)
show.index=show.index+1
st.dataframe(show,use_container_width=True,height=350)
st.caption(f'Showing {len(show)} of {len(df)} total violations')

st.markdown('---')

# Evidence
st.markdown('## 📸 Evidence Screenshots')
valid=fdf[fdf.screenshot_path.apply(lambda x: os.path.exists(str(x)))].screenshot_path.tolist()
if valid:
    c1,c2=st.columns([1,2])
    with c1:
        sel=st.selectbox('Select evidence:',valid)
        if sel:
            row=fdf[fdf.screenshot_path==sel].iloc[0]
            st.markdown(f'''
            <div class="card" style="text-align:left;margin-top:10px">
            <div style="color:#888;font-size:0.8em">VIOLATION DETAILS</div>
            <hr style="border-color:#333">
            <div>🕐 <b style="color:#aaa">Time:</b><br><code style="color:#e94560">{row["timestamp"]}</code></div><br>
            <div>⚠️ <b style="color:#aaa">Type:</b><br><code style="color:#f5a623">{row["violation_type"]}</code></div><br>
            <div>🚗 <b style="color:#aaa">Plate:</b><br><code style="color:#06d6a0">{row["plate_clean"]}</code></div>
            </div>
            ''', unsafe_allow_html=True)
    with c2:
        if sel and os.path.exists(sel):
            img=Image.open(sel)
            st.image(img,caption=f'Evidence: {sel}',use_column_width=True)
else:
    st.info('No screenshots found. Run detection locally to generate evidence.')

st.markdown('---')

# Downloads
st.markdown('## ⬇️ Export Reports')
c1,c2,c3=st.columns(3)
ts=datetime.now().strftime('%Y%m%d_%H%M%S')
with c1:
    st.download_button('📥 Full Report',df.to_csv(index=False),f'all_violations_{ts}.csv','text/csv',use_container_width=True)
with c2:
    st.download_button('🔴 Red Light Only',df[df.violation_type=='red_light_jump'].to_csv(index=False),'red_light.csv','text/csv',use_container_width=True)
with c3:
    st.download_button('⛑️ Helmet Only',df[df.violation_type=='helmetless_riding'].to_csv(index=False),'helmet.csv','text/csv',use_container_width=True)

st.markdown('<br><div style="text-align:center;color:#444;font-size:0.85em">🚦 Traffic Violation Detection System &nbsp;•&nbsp; Pranav Gedela &nbsp;•&nbsp; GITAM University 2025-26</div>', unsafe_allow_html=True)
