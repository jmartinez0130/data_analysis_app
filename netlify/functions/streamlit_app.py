import streamlit as st
from streamlit.web.bootstrap import run

def handler(event, context):
    return run(
        lambda: exec(open('streamlit_app.py').read()),
        flag_options={
            "server.address": "0.0.0.0",
            "server.headless": True,
            "server.enableCORS": False,
            "server.enableXsrfProtection": False
        }
    )
