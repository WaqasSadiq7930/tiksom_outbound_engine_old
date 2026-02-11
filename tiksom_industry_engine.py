import pandas as pd
import numpy as np
import re
import time
import streamlit as st
from groq import Groq
from industry_keywords import INDUSTRY_MAP, STRONG_SIGNALS

# ----------------------------
# 1. CONTENT DB
# ----------------------------
CONTENT_DB = {
    "IT_Software": {
        "1-10": {"pain": "developer burnout & backlog", "sol": "AI-Augmented Remote Squads", "val": "ship features 3x faster"},
        "11-50": {"pain": "scaling technical debt", "sol": "Autonomous DevOps Pipelines", "val": "cut deployment costs by 40%"}
    },
    "Healthcare": {
        "1-10": {"pain": "patient admin overload", "sol": "AI Voice Agents for Clinics", "val": "automate 90% of scheduling"},
        "11-50": {"pain": "interoperability silos", "sol": "FHIR-Compliant Data Layers", "val": "unify patient data instantly"}
    },
    "Fintech": {
        "1-10": {"pain": "compliance bottlenecks", "sol": "Automated KYC/AML Engines", "val": "onboard users in seconds"},
        "11-50": {"pain": "legacy infrastructure limits", "sol": "Modular Banking APIs", "val": "launch new products in weeks"}
    },
    "Real_Estate": {
        "1-10": {"pain": "manual lead follow-ups", "sol": "24/7 AI Leasing Agents", "val": "never miss a tenant inquiry"},
        "11-50": {"pain": "property maintenance chaos", "sol": "IoT-Predictive Maintenance", "val": "reduce repair costs drastically"}
    },
    "General": { 
        "1-10": {"pain": "repetitive manual workflows", "sol": "GenAI Process Automation", "val": "reclaim 20 hours/week"},
        "11-50": {"pain": "data silos slowing growth", "sol": "Unified Intelligence Dashboards", "val": "make real-time decisions"}
    }
}

# ----------------------------
# 2. HELPER FUNCTIONS
# ----------------------------

def normalize_industry_name(name):
    if not name: return "General"
    return name.strip().replace(" ", "_")

def clean_company_name(name):
    if pd.isna(name): return ""
    name = str(name)
    name = re.sub(r'\b(inc|ltd|llc|corp|company|co|limited|gmbh|plc|pvt|private)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[^a-zA-Z0-9 ]', '', name)
    return name.strip()

def normalize(text):
    return re.sub(r"\s+", " ", str(text).lower())

def bucket_headcount(val):
    # PURE LOCAL (FAST)
    if pd.isna(val) or str(val).strip() == "": 
        return "1-10", True 
    try:
        val_str = str(val).lower().replace(",", "")
        nums = re.findall(r'\d+', val_str)
        if not nums: return "1-10", True
        
        if len(nums) > 1:
            avg = (int(nums[0]) + int(nums[1])) / 2
            val_int = int(avg)
        else:
            val_int = int(nums[0])

        if val_int <= 10: return "1-10", False
        elif 11 <= val_int <= 50: return "11-50", False
        else: return "51+", False
    except:
        return "1-10", True

# ----------------------------
# 3. CAMPAIGN GENERATOR
# ----------------------------

def generate_campaign_text(industry, headcount):
    lookup_ind = normalize_industry_name(industry)
    lookup_ind = lookup_ind if lookup_ind in CONTENT_DB else "General"
    
    db_entry = CONTENT_DB.get(lookup_ind, CONTENT_DB["General"])
    c = db_entry.get(headcount, db_entry.get("1-10"))
    links = "https://www.tiksom.co.uk/portfolio and https://hashlogics.com/portfolio"

    return f"""=== CAMPAIGN STRATEGY: {industry} | {headcount} ===
PAIN: {c['pain']}
SOLUTION: {c['sol']}
VALUE: {c['val']}

-------------------------------------------------------
[EMAIL 1 - INITIAL]
Subject: Fix {c['pain']} at {{Company Name}}?

Hi {{First Name}},
Dealing with {c['pain']} in 2026? Our {c['sol']} helps teams {c['val']}.
See examples: {links}
10 min call? [link]
Best, Fahad
"""

# ----------------------------
# 4. CLASSIFICATION LOGIC (BALANCED: SPEED + ACCURACY)
# ----------------------------

def ai_classify(client, text):
    # 1. Skip Junk (Speed)
    if len(text) < 15: return "General"
    if not client: return "General"
    
    prompt = f"""Classify into ONE category: {', '.join(INDUSTRY_MAP.keys())}. 
    Return ONLY the category name. Guess the closest one. Do NOT return Unknown.
    Text: {text[:300]}"""

    # 2. Try 1: Normal Call
    try:
        if 'api_status' in st.session_state:
            st.session_state['api_status'] = "🟢 AI Active"
            
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=20
        )
        return normalize_industry_name(resp.choices[0].message.content.strip())
        
    except Exception as e:
        # 3. Try 2: The "One-Shot" Rescue (Accuracy)
        # Agar Rate Limit hai to 1.5 second ruko aur phirse try karo
        # Ye "General" mein phenkne se behtar hai
        try:
            if 'api_status' in st.session_state:
                st.session_state['api_status'] = "🟠 Rate Limit (Retrying in 1.5s)"
            
            time.sleep(1.5) # Small wait
            
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=20
            )
            return normalize_industry_name(resp.choices[0].message.content.strip())
            
        except:
            # Agar ab bhi fail hua, to phir General
            if 'api_status' in st.session_state:
                st.session_state['api_status'] = "🔴 Skip (Local Fallback)"
            return "General"

def classify_row(text, clf, tfidf, client):
    text = normalize(text)
    
    # PRIORITY 1: STRONG SIGNALS (Highest Accuracy + Zero Cost)
    for ind, signals in STRONG_SIGNALS.items():
        for s in signals:
            if re.search(r"\b" + re.escape(s) + r"\b", text):
                return normalize_industry_name(ind), "Strong_Signal", 1.0

    # PRIORITY 2: ML MODEL (High Speed + Good Accuracy)
    if clf and tfidf:
        try:
            X = tfidf.transform([text])
            probs = clf.predict_proba(X)[0]
            best_idx = np.argmax(probs)
            # 25% Threshold is safer than 20%
            if probs[best_idx] >= 0.25:
                return normalize_industry_name(clf.classes_[best_idx]), "ML_Aggressive", probs[best_idx]
        except: pass

    # PRIORITY 3: KEYWORDS (Speed)
    scores = {ind: 0 for ind in INDUSTRY_MAP}
    for ind, kws in INDUSTRY_MAP.items():
        for k in kws:
            if re.search(r"\b" + re.escape(k) + r"\b", text): scores[ind] += 1
            
    best_kw = max(scores, key=scores.get)
    if scores[best_kw] > 0: 
        return normalize_industry_name(best_kw), "Keyword", 0.6

    # PRIORITY 4: API (Last Resort - Accuracy)
    if client:
        return ai_classify(client, text), "AI_Forced", 1.0
        
    return "General", "General_Fallback", 0.0

# ----------------------------
# 5. PIPELINE
# ----------------------------

def classify_dataframe(df, clf, tfidf, client, text_col, headcount_col, company_col):
    df = df.copy()
    
    df['clean_company'] = df[company_col].apply(clean_company_name)
    
    headcount_results = df[headcount_col].apply(bucket_headcount)
    df['headcount_bucket'] = [x[0] for x in headcount_results]
    df['is_estimated_headcount'] = [x[1] for x in headcount_results]
    
    results = []
    sources = []
    confs = []
    
    for row in df.itertuples():
        c_text = str(getattr(row, text_col, ""))
        c_comp = str(getattr(row, company_col, ""))
        full_text = f"{c_comp} {c_text}"
        
        ind, src, conf = classify_row(full_text, clf, tfidf, client)
        
        results.append(ind)
        sources.append(src)
        confs.append(conf)
        
    df['final_industry'] = results
    df['source'] = sources
    df['confidence'] = confs
    return df