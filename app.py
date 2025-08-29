import streamlit as st
from openai import OpenAI

# Load API key from Streamlit Secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="MindCare AI", page_icon="💙")
st.markdown("<h2 style='text-align: center;'>💙 MindCare AI - Confidential Chatbot (Text + Voice)</h2>", unsafe_allow_html=True)

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello 💙 I'm MindCare AI. You can type or talk to me — how are you feeling today?"}
    ]

# Show past chat
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------------- Input Bar ----------------------
col1, col2 = st.columns([8, 1])
with col1:
    user_text = st.text_input("Type your thoughts...", key="input_text", label_visibility="collapsed")
with col2:
    audio = st.audio_input("🎤", label_visibility="collapsed")

# ---------------------- Handle Text ----------------------
if user_text:
    st.session_state["messages"].append({"role": "user", "content": user_text})
    st.chat_message("user").write(user_text)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are MindCare AI, an empathetic wellness assistant. Be supportive, kind, and safe."},
            *st.session_state["messages"]
        ]
    )
    reply = completion.choices[0].message.content
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)

# ---------------------- Handle Voice ----------------------
if audio:
    with open("input.wav", "wb") as f:
        f.write(audio.getvalue())

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=open("input.wav", "rb")
    )
    spoken_text = transcript.text
    st.session_state["messages"].append({"role": "user", "content": spoken_text})
    st.chat_message("user").write(spoken_text)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are MindCare AI, an empathetic wellness assistant. Be supportive, kind, and safe."},
            *st.session_state["messages"]
        ]
    )
    reply = completion.choices[0].message.content
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
