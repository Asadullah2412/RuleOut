import streamlit as st
from llm import LLM_Chat  

# --- 1. Page Config ---
st.set_page_config(
    page_title="RuleOut | Pediatric Clinical Assistant",
    page_icon="🩺",
    layout="wide"
)

# --- 2. Session State for UI Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 3. Sidebar (Controls & Patient Quick-Context) ---
with st.sidebar:
    st.title("🩺 RuleOut")
    st.caption("Pediatric Clinical Decision Support")
    st.divider()

    st.subheader("Patient Context (Optional)")
    age = st.text_input("Age", placeholder="e.g., 18 months")
    weight = st.text_input("Weight (kg)", placeholder="e.g., 11.2 kg")
    allergies = st.text_input("Allergies", placeholder="e.g., NKDA / Penicillin")

    st.divider()
    if st.button("🗑️ Clear Consultation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.info("⚠️ **Clinical Disclaimer:** Decision-support tool only. Verify all dosages and protocols against institutional guidelines.")

# --- 4. Main Chat Interface ---
st.header("Pediatric Clinical Consultation")

# Render existing messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 5. Handle User Input ---
if prompt := st.chat_input("Enter clinical case details, symptoms, or protocol query..."):
    # Build patient context header if sidebar fields are filled
    patient_meta = []
    if age: patient_meta.append(f"Age: {age}")
    if weight: patient_meta.append(f"Weight: {weight}")
    if allergies: patient_meta.append(f"Allergies: {allergies}")

    full_query = f"[Patient Context: {', '.join(patient_meta)}]\n\n{prompt}" if patient_meta else prompt

    # Display clean user message in UI
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Call your dedicated LLM function
    with st.chat_message("assistant"):
        with st.spinner("Analyzing case & querying knowledge base..."):
            try:
                response_md = LLM_Chat(query=full_query)
                st.markdown(response_md)
                st.session_state.chat_history.append({"role": "assistant", "content": response_md})
            except Exception as e:
                st.error(f"Error generating clinical response: {e}")
