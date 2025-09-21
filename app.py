import streamlit as st
import datetime
import speech_recognition as sr
from transformers import pipeline
import subprocess
import re
import os
import google.generativeai as genai
import html  

# --- Configuration Constants ---
GEMINI_MODEL_NAME = 'gemini-1.5-flash'
DEFAULT_OLLAMA_MODEL = "llama3"
EMOTION_CLASSIFICATION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# --- Page Config ---
st.set_page_config(
    page_title="MindfulYouth - AI Mental Wellness",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
body, .main { background: #0E1117; color: #FAFAFA; font-family: 'Inter', sans-serif; }
.chat-container { display: flex; flex-direction: column; gap: 12px; padding: 20px; max-width: 800px; margin: 0 auto; height: calc(100vh - 180px); overflow-y: auto; }
.message { display: flex; margin-bottom: 18px; }
.message-user { justify-content: flex-end; }
.message-ai { justify-content: flex-start; }
.message-content { max-width: 75%; padding: 15px 20px; border-radius: 12px; }
.user-message { background: #262730; color: #FAFAFA; border-radius: 12px 12px 4px 12px; }
.ai-message { background: #1A1D25; color: #FAFAFA; border-radius: 12px 12px 12px 4px; border-left: 4px solid #25D366; }
.timestamp { font-size: 0.75rem; color: #6B7280; margin-top: 8px; text-align: right; }
.avatar { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px; flex-shrink: 0; font-weight: bold; font-size: 0.8rem; }
.avatar-user { background: #2563EB; color: white; }
.avatar-ai { background: #10B981; color: white; }
.chat-input-container { display: flex; align-items: center; gap: 8px; }
.chat-mic { background: #25D366; border: none; color: white; padding: 10px; border-radius: 50%; cursor: pointer; font-size: 18px; }
.chat-mic:hover { background: #128C7E; }
</style>
""", unsafe_allow_html=True)

# --- Initialize Gemini API ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.warning("GEMINI_API_KEY environment variable not set. Online assistance (Gemini) will be disabled.")
else:
    try:
        genai.configure(api_key=api_key)
        if 'gemini_model' not in st.session_state:
            st.session_state.gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}. Online assistance will be disabled.")
        api_key = None

def generate_online_response(prompt):
    if not api_key or 'gemini_model' not in st.session_state:
        return None

    try:
        model = st.session_state.gemini_model
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.7)
        )
        return response.text
    except Exception as e:
        st.error(f"Gemini error: {e}")
        return None

# --- Emotion Detection ---
@st.cache_resource
def load_emotion_analyzer():
    return pipeline(
        "text-classification",
        model=EMOTION_CLASSIFICATION_MODEL,
        return_all_scores=True
    )

emotion_analyzer = load_emotion_analyzer()

def detect_emotions(text):
    try:
        results = emotion_analyzer(text)[0]
        sorted_emotions = sorted(results, key=lambda x: x['score'], reverse=True)[:2]
        return [(emo['label'].lower(), round(emo['score'], 2)) for emo in sorted_emotions]
    except Exception:
        return [("neutral", 0.7)]

emotion_emoji = {
    "joy": "😊", "sadness": "😢", "anger": "😠", "fear": "😨", "love": "❤️",
    "surprise": "😲", "neutral": "😐", "embarrassment": "😳", "disgust": "🤢",
    "shame": "😔", "guilt": "😥", "admiration": "🤩", "amusement": "😂",
    "annoyance": "😒", "approval": "👍", "caring": "🥰", "confusion": "🤔",
    "curiosity": "🧐", "desire": "😍", "disappointment": "😞", "disapproval": "👎",
    "excitement": "🥳", "gratitude": "🙏", "grief": "😭", "hope": "🤞",
    "indifference": "😑", "optimism": "😀", "pride": "🦁", "realization": "💡",
    "relief": "😌", "remorse": "😔", "resignation": "🤷‍♀️", "terror": "😱",
    "vengeance": "😈", "worry": "😟"
}

def confidence_bar(score, length=5):
    filled = int(round(score * length))
    empty = length - filled
    return "▮" * filled + "▯" * empty

# --- Voice Recognition ---
def recognize_speech():
    try:
        r = sr.Recognizer()
        with st.spinner("Listening..."):
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=1)
                audio = r.listen(source, timeout=5, phrase_time_limit=6)
            return r.recognize_google(audio)
    except sr.UnknownValueError:
        st.warning("Could not understand audio. Please try again.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred during speech recognition: {e}")
        return None

# --- Offline Ollama Functions ---
@st.cache_data(ttl=3600)
def get_installed_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        models = []
        for line in result.stdout.splitlines()[1:]:
            parts = re.split(r"\s{2,}", line.strip())
            if parts and parts[0] and parts[0] != "NAME":
                models.append(parts[0])
        return models if models else [DEFAULT_OLLAMA_MODEL]
    except FileNotFoundError:
        st.warning("Ollama command not found. Please ensure Ollama is installed and in your PATH.")
        return [DEFAULT_OLLAMA_MODEL]
    except Exception as e:
        st.error(f"Error listing Ollama models: {e}")
        return [DEFAULT_OLLAMA_MODEL]

def query_ollama(model, prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            error_msg = result.stderr.strip()
            st.error(f"Ollama error (Model: {model}): {error_msg}")
            return f"I'm having trouble connecting with my offline brain ({model}): {error_msg}. Please check your Ollama setup."
        return result.stdout.strip() or "I'm here to listen. How are you feeling today?"
    except FileNotFoundError:
        return "Ollama command not found. Please ensure Ollama is installed and in your PATH."
    except subprocess.TimeoutExpired:
        return "My offline thoughts are taking too long. Can you rephrase or try again?"
    except Exception as e:
        st.error(f"An unexpected error occurred while querying Ollama: {e}")
        return "I'm here to support you. How are you feeling right now?"

# --- Clean HTML from text ---
def clean_html_tags(text):
    """Remove HTML tags from text and escape any HTML entities"""
    if not text:
        return ""
    
    # First remove any HTML tags including div with timestamp class
    clean_text = re.sub(r'<div class="timestamp">.*?</div>', '', text, flags=re.DOTALL)
    clean_text = re.sub(r'<[^>]*>', '', clean_text)
    
    # Then escape any HTML entities to prevent rendering
    clean_text = html.escape(clean_text)
    
    # Unescape common entities that should remain readable
    clean_text = clean_text.replace('&quot;', '"').replace('&#x27;', "'").replace('&amp;', '&')
    
    return clean_text.strip()

# --- Session State Initialization ---
if "chat" not in st.session_state:
    st.session_state.chat = []
if "pending_user_input" not in st.session_state:
    st.session_state.pending_user_input = None
if "use_online" not in st.session_state:
    st.session_state.use_online = False
if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_OLLAMA_MODEL

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    models = get_installed_models()
    if st.session_state.selected_model not in models and models:
        st.session_state.selected_model = models[0]
    st.session_state.selected_model = st.selectbox(
        "Offline AI Model",
        models,
        index=models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0,
        key="ollama_model_select"
    )

    st.session_state.use_online = st.checkbox("Enable Online Assistance (Gemini)", value=st.session_state.use_online, key="gemini_checkbox")
    if st.session_state.use_online and not api_key:
        st.warning("GEMINI_API_KEY is not set or configured. Online assistance will not work.")

    if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_button"):
        st.session_state.chat = []
        st.session_state.pending_user_input = None
        st.rerun()

# --- Header ---
st.markdown("""
<div style="text-align:center; padding:15px 0;">
<h1 style="margin:0;">🌱 MindfulYouth</h1>
<p style="margin:5px 0 0 0; opacity:0.7; font-size:0.9rem;">AI Mental Wellness Assistant</p>
</div>
""", unsafe_allow_html=True)

# --- Display Chat ---
chat_container_placeholder = st.empty()

with chat_container_placeholder.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for sender, msg, timestamp, emotions in st.session_state.chat:
        # Clean the message to remove any HTML tags
        safe_msg = clean_html_tags(msg)
        
        # User message HTML (includes emotion analysis)
        if sender == "You":
            emo_text = " ".join([f"{emotion_emoji.get(emo,'😐')} {emo.capitalize()} [{confidence_bar(conf)}]" for emo, conf in emotions])
            st.markdown(f"""
            <div class="message message-user">
                <div class="message-content user-message">
                    <div><b>You</b></div>
                    {safe_msg}
                    <div style="margin-top:5px; font-size:0.8rem;">🧠 {emo_text}</div>
                    <div class="timestamp">{timestamp}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        # AI message HTML (simple text response)
        else:
            st.markdown(f"""
            <div class="message message-ai">
                <div class="message-content ai-message">
                    <div><b>MindfulAssistant</b></div>
                    {safe_msg}
                    <div class="timestamp">{timestamp}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Scroll to bottom of chat
st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

# --- Input & Voice Logic ---
current_input_from_widget = None
voice_input_text = None

col1, col2 = st.columns([10, 1])
with col1:
    chat_input_val = st.chat_input("Type a message...", key="main_chat_input")
    if chat_input_val:
        current_input_from_widget = chat_input_val

with col2:
    if st.button("🎙️", key="mic_btn"):
        voice_input_text = recognize_speech()

# --- Process User Input (from either source) ---
user_input_to_process = None
if current_input_from_widget:
    user_input_to_process = current_input_from_widget
elif voice_input_text:
    user_input_to_process = voice_input_text

if user_input_to_process:
    timestamp = datetime.datetime.now().strftime("%H:%M")
    emotions = detect_emotions(user_input_to_process)
    st.session_state.chat.append(("You", user_input_to_process, timestamp, emotions))
    st.session_state.pending_user_input = user_input_to_process
    st.rerun()

# --- AI Reply Processing ---
if st.session_state.pending_user_input:
    with st.spinner("🤔 Thinking..."):
        context_messages = []
        for sender, msg, _, _ in st.session_state.chat[:-1]:
            context_messages.append(f"{sender}: {msg}")
        
        current_user_message = st.session_state.pending_user_input

        prompt = f"""
        You are MindfulYouth, an AI mental wellness assistant. Your goal is to be empathetic, supportive, and helpful.
        Based on the conversation history and the user's current message, provide a kind and encouraging response.
        If the user is asking for advice or expressing a strong emotion, acknowledge it and respond thoughtfully.

        Conversation so far:
        {'\n'.join(context_messages)}

        User's current message: "{current_user_message}"
        Detected emotions for current message: {detect_emotions(current_user_message)}

        Your empathetic response:
        """
        
        reply = None
        if st.session_state.use_online and api_key:
            reply = generate_online_response(prompt)
            if reply:
                st.toast("Online assistance used (Gemini).", icon="🌐")
            else:
                st.warning("Online assistance (Gemini) failed. Falling back to offline model.")
        
        if not reply:
            reply = query_ollama(st.session_state.selected_model, prompt)
            st.toast("Offline assistance used (Ollama).", icon="💻")

    # Clean the reply to remove any HTML tags
    clean_reply = clean_html_tags(reply) if reply else "I'm here to listen. How are you feeling today?"

    st.session_state.chat.append(("AI", clean_reply, datetime.datetime.now().strftime("%H:%M"), []))
    st.session_state.pending_user_input = None
    st.rerun()
