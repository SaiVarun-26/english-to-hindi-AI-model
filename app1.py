import streamlit as st
import os
import getpass

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="AI Translator", page_icon="🤖", layout="centered")

st.title("🌐 AI Translator with Gemini + LangChain")

# -----------------------------
# API Keys setup
# -----------------------------
if "GOOGLE_API_KEY" not in os.environ:
    google_api = st.text_input("🔑 Enter your Google Gemini API Key", type="password")
    if google_api:
        os.environ["GOOGLE_API_KEY"] = google_api

if "LANGSMITH_API_KEY" not in os.environ:
    langsmith_api = st.text_input("🔑 Enter your LangSmith API Key (optional)", type="password")
    if langsmith_api:
        os.environ["LANGSMITH_API_KEY"] = langsmith_api

# -----------------------------
# Initialize model
# -----------------------------
if "GOOGLE_API_KEY" in os.environ:
    model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
else:
    st.warning("⚠ Please enter your Google Gemini API key above to continue.")
    st.stop()

# -----------------------------
# User input
# -----------------------------
st.subheader("✍️ Enter English text to translate into Hindi")
user_input = st.text_area("Your text:", placeholder="Type something in English...")

if st.button("Translate"):
    if user_input.strip():
        with st.spinner("Translating..."):
            # Send message to the model
            messages = [
                SystemMessage(content="Translate the following from English into Hindi"),
                HumanMessage(content=user_input),
            ]
            response = model.invoke(messages)

            # Display response
            st.success("✅ Translation:")
            st.write(response.content)
    else:
        st.error("⚠ Please enter some text.")

# -----------------------------
# Streaming output (optional)
# -----------------------------
if st.checkbox("Enable Streaming Output"):
    if user_input.strip():
        st.write("⏳ Streaming Translation:")
        placeholder = st.empty()
        streamed_text = ""
        for token in model.stream(
            [
                SystemMessage(content="Translate the following from English into Hindi"),
                HumanMessage(content=user_input),
            ]
        ):
            streamed_text += token.content
            placeholder.markdown(f"**{streamed_text}**")
