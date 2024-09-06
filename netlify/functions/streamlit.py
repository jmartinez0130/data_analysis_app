from streamlit.web.bootstrap import run
import streamlit as st

def handler(event, context):
    return run(lambda: exec(open('streamlit_app.py').read()), flag_options={"server.address": "0.0.0.0", "server.headless": True})

if __name__ == "__main__":
    handler(None, None)
