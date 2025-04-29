@echo off
set PYTHONPATH=%CD%\src;%PYTHONPATH%
cd src
streamlit run app.py 