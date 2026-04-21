import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uuid
import streamlit as st

from agent.graph import build_graph, ask
from config import EMERGENCY_NUMBER, HELPLINE

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MediAssist — MediCare General Hospital",
    page_icon="🏥",
    layout="centered",
)

# ── Load graph once ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_graph():
    app, _ = build_graph()
    return app

compiled_app = load_graph()

# ── Session state init ────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏥 MediAssist")
    st.caption("MediCare General Hospital, Bhubaneswar")
    st.divider()

    st.markdown(
        "I can help you with:\n"
        "- OPD timings & appointments\n"
        "- Consultation fees\n"
        "- Insurance & TPA queries\n"
        "- Lab tests & diagnostics\n"
        "- Pharmacy information\n"
        "- Health packages\n"
        "- Admission & discharge\n"
        "- Visitor policy\n"
    )

    st.divider()

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.caption(f"Session ID: `{st.session_state.thread_id[:8]}...`")
    st.caption(f"Emergency: **{EMERGENCY_NUMBER}**")
    st.caption(f"Helpline: **{HELPLINE}**")

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("MediAssist")
st.caption("Patient Assistant — MediCare General Hospital")

st.info(
    f"For medical emergencies call **{EMERGENCY_NUMBER}** immediately. "
    "This assistant provides information only and does not give medical advice.",
    icon="⚠️",
)

# ── Render chat history ───────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "debug" in msg:
            with st.expander("Debug info", expanded=False):
                d = msg["debug"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Route",        d.get("route", "—"))
                col2.metric("Faithfulness", f"{d.get('faithfulness', 0):.2f}")
                col3.metric("Retries",      d.get("eval_retries", 0))
                if d.get("sources"):
                    st.markdown(f"**Sources:** {', '.join(d['sources'])}")

# ── Handle user input ─────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask me anything about MediCare Hospital..."):

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask(compiled_app, prompt, st.session_state.thread_id)
                answer = result.get("answer") or "I'm sorry, I couldn't process that request. Please call our helpline at " + HELPLINE + "."
            except Exception as e:
                answer = f"Something went wrong. Please call our helpline at {HELPLINE}. (Error: {str(e)[:80]})"
                result = {}

        st.markdown(answer)

        debug = {
            "route":        result.get("route", "—"),
            "faithfulness": result.get("faithfulness", 0.0),
            "eval_retries": result.get("eval_retries", 0),
            "sources":      result.get("sources", []),
        }

        with st.expander("Debug info", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("Route",        debug["route"])
            col2.metric("Faithfulness", f"{debug['faithfulness']:.2f}")
            col3.metric("Retries",      debug["eval_retries"])
            if debug["sources"]:
                st.markdown(f"**Sources:** {', '.join(debug['sources'])}")

    # Store assistant message with debug payload
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "debug":   debug,
    })
