import streamlit as st
import requests
from datetime import datetime

# Streamlit Page Config
st.set_page_config(
    page_title="ATM_LLM Report",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Style
st.markdown("""
    <style>
    .stApp {
        background-color: #E6F0FA;
    }
    .stTextArea textarea {
        background-color: white;
        padding: 10px;
        border: 2px solid #3B82F6;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        overflow: visible !important;
        height: auto !important;
        min-height: 200px !important;
        max-height: none !important;
        white-space: pre-wrap !important;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        margin: 5px;
        font-weight: bold;
        height: 40px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
    .stSubheader {
        color: #1E40AF;
    }
    h1 {
        font-size: 36px !important;
        font-weight: bold !important;
        color: #1E3A8A !important;
    }
    .bold-input label {
        font-weight: bold !important;
    }
    .large-input input {
        height: 40px !important;
        font-size: 16px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Date and Time
current_time = datetime(2025, 7, 14, 10, 23)

# Page Title
st.title("ATM Report Generator & Translator")

# Input Field
st.markdown('<div class="bold-input large-input">', unsafe_allow_html=True)
technical_input = st.text_input("Enter technical input (e.g. `HW ISSUE_MTS_Replacement / HW ISSUE_CDM_Replacement`):", key="technical_input")
st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if "english_message" not in st.session_state:
    st.session_state.english_message = ""
if "translated_message" not in st.session_state:
    st.session_state.translated_message = ""

# Fallback fix for missing intervention
def generate_fallback_if_missing(english_message, technical_input):
    lines = english_message.strip().split('\n')
    has_intervention_header = any("intervention report" in line.lower() for line in lines)
    has_intervention_item = any(line.strip().startswith("1.") or "The affected component was" in line for line in lines)
    parts = [p.strip() for p in technical_input.split('/')]

    if len(parts) == 1 and has_intervention_header and not has_intervention_item:
        action = parts[0].split('_')[-1].lower()
        action_past = {
            "replacement": "replaced",
            "repair": "repaired",
            "cleaning": "cleaned",
            "fix": "fixed"
        }.get(action, f"{action}ed")
        fallback_sentence = f"The affected component was {action_past}."
        if fallback_sentence not in english_message:
            return english_message.strip() + "\n" + fallback_sentence
    return english_message

# Buttons layout
col1, col2, col3 = st.columns([1, 1, 1])

# Full Report (English + French)
with col1:
    if st.button("Full Report"):
        technical_input = st.session_state.technical_input.strip()
        if not technical_input:
            st.warning("Please enter a valid technical input.")
        else:
            with st.spinner("Generating both reports..."):
                try:
                    res = requests.post(
                        "http://localhost:5000/generate-message",
                        json={"technical_input": technical_input},
                        timeout=120
                    )
                    if res.status_code == 200:
                        data = res.json()
                        english_message = data.get("english_report", "")
                        if english_message:
                            english_message = generate_fallback_if_missing(english_message, technical_input)
                            st.session_state.english_message = f"Generated on {current_time.strftime('%Y-%m-%d %I:%M %p CET')}:\n\n{english_message}"
                        else:
                            st.error("No English report received.")
                    else:
                        st.error(f"English generation error: {res.text}")

                    if st.session_state.english_message:
                        trans_res = requests.post(
                            "http://localhost:5002/translate",
                            json={"text": st.session_state.english_message},
                            timeout=120
                        )
                        if trans_res.status_code == 200:
                            translation = trans_res.json().get("translated_text", "")
                            if translation:
                                st.session_state.translated_message = f"Translated on {current_time.strftime('%Y-%m-%d %I:%M %p CET')}:\n\n{translation}"
                            else:
                                st.error("No translation received.")
                        else:
                            st.error(f"Translation error: {trans_res.text}")
                    st.success("Both reports generated!")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

# English Only
with col2:
    if st.button("English Only"):
        technical_input = st.session_state.technical_input.strip()
        if not technical_input:
            st.warning("Please enter a valid technical input.")
        else:
            with st.spinner("Generating English report..."):
                try:
                    res = requests.post(
                        "http://localhost:5000/generate-message",
                        json={"technical_input": technical_input},
                        timeout=30
                    )
                    if res.status_code == 200:
                        data = res.json()
                        english_message = data.get("english_report", "")
                        if english_message:
                            english_message = generate_fallback_if_missing(english_message, technical_input)
                            st.session_state.english_message = f"Generated on {current_time.strftime('%Y-%m-%d %I:%M %p CET')}:\n\n{english_message}"
                            st.session_state.translated_message = ""
                            st.success("English report generated!")
                        else:
                            st.error("No English report received.")
                    else:
                        st.error(f"English generation error: {res.text}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

# French Only
with col3:
    if st.button("French Only"):
        if not st.session_state.english_message:
            st.warning("Please generate an English message first.")
        else:
            with st.spinner("Generating French report..."):
                try:
                    trans_res = requests.post(
                        "http://localhost:5002/translate",
                        json={"text": st.session_state.english_message},
                        timeout=120
                    )
                    if trans_res.status_code == 200:
                        translation = trans_res.json().get("translated_text", "")
                        if translation:
                            st.session_state.translated_message = f"Translated on {current_time.strftime('%Y-%m-%d %I:%M %p CET')}:\n\n{translation}"
                            st.success("French report generated!")
                        else:
                            st.error("No translation received.")
                    else:
                        st.error(f"Translation error: {trans_res.text}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

# Output English Report
if st.session_state.english_message:
    st.subheader("English Report")
    st.markdown(f"<div style='white-space: pre-wrap; background-color: white; padding: 15px; border-radius: 10px; border: 2px solid #3B82F6;'>{st.session_state.english_message}</div>", unsafe_allow_html=True)

# Output French Translation
if st.session_state.translated_message:
    st.subheader("French Translation")
    st.markdown(f"<div style='white-space: pre-wrap; background-color: white; padding: 15px; border-radius: 10px; border: 2px solid #3B82F6;'>{st.session_state.translated_message}</div>", unsafe_allow_html=True)
