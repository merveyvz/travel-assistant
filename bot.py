import streamlit as st
from chatbot.utils import write_message
from chatbot.agent import generate_response

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# page config
st.set_page_config("Ebert", page_icon="🎙️")

# set up session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the travel assistant Chatbot!  How can I help you?"},
    ]

# submit handler
def handle_submit(message):
    # handle the response
    with st.spinner("Thinking"):
        # call the agent
        response = generate_response(message)
        write_message("assistant", response)

# display messages in session state
for message in st.session_state.messages:
    write_message(message["role"], message["content"], save=False)


# handle any user input
if prompt := st.chat_input("What is up?"):
    # display user message in chat message container
    write_message("user", prompt)

    # generate a response
    handle_submit(prompt)
