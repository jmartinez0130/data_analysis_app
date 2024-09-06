import streamlit as st
from streamlit.web.bootstrap import run
import streamlit_app  # This imports your main app file

def handler(event, context):
    run(streamlit_app, flag_options={"server.headless": True})
