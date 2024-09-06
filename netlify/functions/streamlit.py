from streamlit.web.bootstrap import run
import streamlit as st

def handler(event, context):
    return run(lambda: st.set_page_config(page_title="My Streamlit App") or exec(open('streamlit_app.py').read()), flag_options={"server.address": "0.0.0.0", "server.headless": True})
