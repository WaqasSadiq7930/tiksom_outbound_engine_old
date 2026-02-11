import streamlit as st
import pandas as pd
from groq import Groq
import io
import zipfile
import os
import joblib
import time

from tiksom_industry_engine import classify_dataframe, generate_campaign_text

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="Tiksom Engine", layout="wide")
st.title("🚀 Tiksom: Balanced Mode (Speed + Accuracy)")

if 'api_status' not in st.session_state:
    st.session_state['api_status'] = "⚪ Ready"

@st.cache_resource
def load_models():
    try:
        clf = joblib.load("models/industry_ml_model.joblib")
        tfidf = joblib.load("models/industry_tfidf.joblib")
        return clf, tfidf
    except:
        return None, None

clf, tfidf = load_models()

# ============================================================
# MAIN APP
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")
    use_ai = st.checkbox("Enable AI (Aggressive Mode)", value=True)
    api_key = st.text_input("Groq API Key", type="password", disabled=not use_ai)
    uploaded_file = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
    
    st.divider()
    if clf: st.success("✅ ML Brain Active")
    else: st.warning("⚠️ ML Brain Missing (Using Keywords + AI)")

if uploaded_file and st.button("🚀 Start Processing"):
    
    # 1. Load Data
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='latin1', on_bad_lines='skip')
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"File Error: {e}")
        st.stop()
        
    df.columns = [str(c).strip().lower() for c in df.columns]
    col_name = next((c for c in df.columns if 'company' in c or 'name' in c), None)
    if not col_name:
        st.error("❌ 'Company Name' column missing.")
        st.stop()

    col_desc = next((c for c in df.columns if 'desc' in c or 'about' in c), None)
    col_ind_src = next((c for c in df.columns if 'industry' in c), None)
    target_text_col = col_desc
    if not col_desc:
        df['synthetic_desc'] = df[col_name].astype(str)
        if col_ind_src:
            df['synthetic_desc'] += " " + df[col_ind_src].astype(str)
        target_text_col = 'synthetic_desc'

    col_head = next((c for c in df.columns if 'headcount' in c or 'employ' in c), None)
    target_head_col = col_head
    if not col_head:
        df['headcount_default'] = "1-10"
        target_head_col = 'headcount_default'

    # ---------------------------------------------------------
    # BALANCED PROCESSING
    # ---------------------------------------------------------
    
    client = Groq(api_key=api_key) if (use_ai and api_key) else None
    
    st.markdown("### 📊 Live Processing Stats")
    status_placeholder = st.empty()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_total = st.empty()
    with c2: metric_local = st.empty()
    with c3: metric_ai = st.empty()
    with c4: metric_general = st.empty()
        
    progress_bar = st.progress(0)
    
    total_rows = len(df)
    processed_count = 0
    count_local = 0 
    count_ai = 0     
    count_general = 0 
    
    # BATCH SIZE 20 = THE SWEET SPOT
    batch_size = 20
    all_results = []
    
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]
        
        # Process
        processed_batch = classify_dataframe(
            df=batch_df,
            clf=clf,
            tfidf=tfidf,
            client=client,
            text_col=target_text_col,
            headcount_col=target_head_col,
            company_col=col_name
        )
        
        all_results.append(processed_batch)
        processed_count += len(batch_df)
        
        # Stats
        local_in_batch = processed_batch['source'].str.contains('ML|Keyword|Strong', case=False, na=False).sum()
        ai_in_batch = processed_batch['source'].str.contains('AI', case=False, na=False).sum()
        general_in_batch = len(processed_batch[processed_batch['final_industry'] == 'General'])
        
        count_local += local_in_batch
        count_ai += ai_in_batch
        count_general += general_in_batch
        
        # UI Refresh
        current_status = st.session_state.get('api_status', "⚪ Ready")
        if "Active" in current_status: color = "✅"
        elif "Rate Limit" in current_status: color = "🟠"
        else: color = "⚪"
        
        status_placeholder.info(f"**API Status:** {color} {current_status}")
        
        metric_total.metric("Total Rows", f"{processed_count} / {total_rows}")
        metric_local.metric("⚡ Local (Fast)", count_local)
        metric_ai.metric("🤖 AI (Accurate)", count_ai)
        metric_general.metric("⚠️ General", count_general)
        
        progress_bar.progress(int((processed_count / total_rows) * 100))
        
    # Finalize
    final_df = pd.concat(all_results, ignore_index=True)
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("full_report.csv", final_df.to_csv(index=False).encode('utf-8'))
        groups = final_df.groupby(['final_industry', 'headcount_bucket'])
        for (ind, hc), group in groups:
            folder_name = ind
            zf.writestr(f"{folder_name}/{hc}_leads.csv", group.to_csv(index=False).encode('utf-8'))
            camp_txt = generate_campaign_text(ind, hc)
            zf.writestr(f"{folder_name}/{hc}_campaign.txt", camp_txt.encode('utf-8'))

    st.success("✅ Complete!")
    st.download_button("📦 Download Results", zip_buffer.getvalue(), "Tiksom_Results.zip", mime="application/zip")