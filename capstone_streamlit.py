"""
capstone_streamlit.py — HR Assist AI Policy Bot
------------------------------------------------
Run: streamlit run capstone_streamlit.py
Requires: GROQ_API_KEY in .env or environment
"""

import streamlit as st
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="HR Assist",
    page_icon="📋",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stChatMessage { background: #1a1a24 !important; border: 1px solid #2a2a3a; border-radius: 12px; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Load agent (cached so it only builds once) ────────────
@st.cache_resource
def load_agent():
    from agent import build_agent
    return build_agent()


try:
    app, embedder, collection = load_agent()
except Exception as e:
    st.error(f"❌ Failed to load agent: {e}")
    st.info("Make sure GROQ_API_KEY is set and agent.py is in the same directory.")
    st.stop()


# ── Session state ─────────────────────────────────────────
if "messages"   not in st.session_state: st.session_state.messages   = []
if "thread_id"  not in st.session_state: st.session_state.thread_id  = str(uuid.uuid4())[:8]
if "stats"      not in st.session_state: st.session_state.stats      = {"queries": 0, "total_faith": 0.0}


# ── Layout: main chat + right sidebar ─────────────────────
col_main, col_side = st.columns([3, 1])

# ── Sidebar ───────────────────────────────────────────────
with col_side:
    st.markdown("### 📋 HR Assist")
    st.caption("Your 24/7 Company HR Policy Assistant")
    st.divider()

    st.success(f"✅ {collection.count()} policy docs loaded")
    st.markdown(f"**Session ID:** `{st.session_state.thread_id}`")

    n = st.session_state.stats["queries"]
    st.markdown(f"**Queries this session:** {n}")
    if n > 0:
        avg = st.session_state.stats["total_faith"] / n
        colour = "🟢" if avg >= 0.8 else "🟡" if avg >= 0.6 else "🔴"
        st.markdown(f"**Avg faithfulness:** {colour} {avg:.2f}")

    st.divider()
    st.markdown("**💡 Try asking:**")
    prompts = [
        "How many annual leave days do I get per year?",
        "What is the notice period for a confirmed employee?",
        "Can I carry forward unused leave to next year?",
        "How does the performance appraisal cycle work?",
        "What expenses can I claim for reimbursement?",
        "What are the rules for working from home?",
    ]
    for p in prompts:
        if st.button(p, key=p, use_container_width=True):
            st.session_state["prefill"] = p
            st.rerun()

    st.divider()
    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.stats     = {"queries": 0, "total_faith": 0.0}
        st.rerun()

    st.divider()
    st.caption("**Policy Topics:**")
    topics = [
        "Annual & Casual Leave", "Sick Leave", "Work From Home",
        "Payroll & Salary", "Reimbursements", "Performance Appraisal",
        "Probation & Onboarding", "Resignation & Notice", "Benefits (PF/Insurance)",
        "Code of Conduct", "POSH Policy", "Public Holidays"
    ]
    for t in topics:
        st.caption(f"• {t}")


# ── Main chat area ────────────────────────────────────────
with col_main:
    st.markdown("## 📋 HR Assist — Company Policy Bot")
    st.caption("Ask about leave, payroll, reimbursements, appraisals, benefits, resignation, and more.")
    st.divider()

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])
            if msg.get("escalate_to_hr"):
                st.warning("🔴 This query may require direct HR assistance — please contact hr@company.com or raise a ticket on the HR portal.")

    # Handle sidebar button prefill
    prefill = st.session_state.pop("prefill", None)

    # Chat input
    user_input = st.chat_input("Ask about leave, payroll, reimbursements, appraisals, WFH policy...") or prefill

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = app.invoke({"question": user_input}, config=config)

            answer = result.get("answer", "Sorry, I could not generate an answer.")
            faith  = result.get("faithfulness", 0.0)
            route  = result.get("route", "")
            sources = result.get("sources", [])
            escalate = result.get("escalate_to_hr", False)

            st.write(answer)

            # Metadata line
            meta_parts = []
            if faith > 0:   meta_parts.append(f"Faithfulness: {faith:.2f}")
            if route:       meta_parts.append(f"Route: {route}")
            if sources:     meta_parts.append(f"Sources: {', '.join(sources[:2])}")
            meta_str = " | ".join(meta_parts)
            if meta_str:
                st.caption(meta_str)

            # Escalation badge
            if escalate:
                st.warning("🔴 This query may require direct HR assistance — please contact hr@company.com or raise a ticket on the HR portal.")

        # Save to session
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "meta": meta_str if meta_parts else "",
            "escalate_to_hr": escalate,
        })

        # Update stats
        st.session_state.stats["queries"] += 1
        st.session_state.stats["total_faith"] += faith