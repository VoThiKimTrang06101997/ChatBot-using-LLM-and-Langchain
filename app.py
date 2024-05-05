import streamlit as st
from llm_chains import load_normal_chain, load_pdf_chat_chain
from streamlit_mic_recorder import mic_recorder
from utils import get_timestamp, load_config, get_avatar
from image_handler import handle_image
from pdf_handler import add_documents_to_db
from html_templates import css
from database_operations import load_last_k_text_messages, save_text_message, save_image_message, save_audio_message, load_messages, get_all_chat_history_ids, delete_chat_history
import sqlite3

import requests
import json
import openai

config = load_config()

# Configure OpenAI API key
openai.api_key = 'openai-api-key'

def ask_openai(query):
    headers = {
        "Authorization": f"Bearer {openai.api_key}",
        "Content-Type": "application/json",
    }
    
    try:
        # Make a request to the OpenAI ChatCompletion API using the GPT-4 Turbo engine
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",  # Specify the model
            messages=[{"role": "system", "content": "You are an AI trained to politely answer questions."},
                      {"role": "user", "content": query}],
            max_tokens=2000,
            temperature=0.2,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        # Parse the response to extract the text content
        text_response = response['choices'][0]['message']['content']
        print(f"Response content: {text_response}")
        return text_response.strip()

    except Exception as e:
        print(f"An error occurred: {e}")
        return str(e)  # Return the error as a string if there are any issues

# # Example of using the bot function with a test query
# print(ask_openai("Write a Python function to reverse a string."))


@st.cache_resource
def load_chain():
    if st.session_state.pdf_chat:
        print("loading pdf chat chain")
        return load_pdf_chat_chain()
    return load_normal_chain()

def toggle_pdf_chat():
    st.session_state.pdf_chat = True
    clear_cache()

def get_session_key():
    if st.session_state.session_key == "new_session":
        st.session_state.new_session_key = get_timestamp()
        return st.session_state.new_session_key
    return st.session_state.session_key

def delete_chat_session_history():
    delete_chat_history(st.session_state.session_key)
    st.session_state.session_index_tracker = "new_session"

def clear_cache():
    st.cache_resource.clear()

def main():
    st.title("Multimodal Local Chat App")
    st.write(css, unsafe_allow_html=True)
    
    if "db_conn" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
        st.session_state.db_conn = sqlite3.connect(config["chat_sessions_database_path"], check_same_thread=False)
        st.session_state.audio_uploader_key = 0
        st.session_state.pdf_uploader_key = 1
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + get_all_chat_history_ids()

    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index)
    pdf_toggle_col, voice_rec_col = st.sidebar.columns(2)
    pdf_toggle_col.toggle("PDF Chat", key="pdf_chat", value=False)
    with voice_rec_col:
        voice_recording=mic_recorder(start_prompt="Record Audio",stop_prompt="Stop recording", just_once=True)
    delete_chat_col, clear_cache_col = st.sidebar.columns(2)
    delete_chat_col.button("Delete Chat Session", on_click=delete_chat_session_history)
    clear_cache_col.button("Clear Cache", on_click=clear_cache)
    
    chat_container = st.container()
    user_input = st.chat_input("Type your message here", key="user_input")
    
    
    uploaded_image = st.sidebar.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    uploaded_pdf = st.sidebar.file_uploader("Upload a pdf file", accept_multiple_files=True, 
                                            key=st.session_state.pdf_uploader_key, type=["pdf"], on_change=toggle_pdf_chat)
      
    if uploaded_pdf:
        with st.spinner("Processing pdf..."):
            try:
                add_documents_to_db(uploaded_pdf)
                st.session_state.pdf_uploader_key += 1
            except Exception as e:
                st.error(f"Failed to process PDF: {e}")
                
    
    if user_input:
        if uploaded_image:
            with st.spinner("Processing image..."):
                llm_answer = handle_image(uploaded_image.getvalue(), user_input)
                save_text_message(get_session_key(), "human", user_input)
                save_image_message(get_session_key(), "human", uploaded_image.getvalue())
                save_text_message(get_session_key(), "ai", llm_answer)
                user_input = None

        if user_input:
            llm_answer = ask_openai(user_input)
            save_text_message(get_session_key(), "human", user_input)
            save_text_message(get_session_key(), "ai", llm_answer)
            user_input = None


    if (st.session_state.session_key != "new_session") != (st.session_state.new_session_key != None):
        with chat_container:
            chat_history_messages = load_messages(get_session_key())

            for message in chat_history_messages:
                with st.chat_message(name=message["sender_type"], avatar=get_avatar(message["sender_type"])):
                    if message["message_type"] == "text":
                        st.write(message["content"])
                    if message["message_type"] == "image":
                        st.image(message["content"])
                    if message["message_type"] == "audio":
                        st.audio(message["content"], format="audio/wav")

        if (st.session_state.session_key == "new_session") and (st.session_state.new_session_key != None):
            st.rerun()

if __name__ == "__main__":
    main()
    
    