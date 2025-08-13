@echo off
cd /d C:\Users\sunpi\ai\mimamori-ai
py -m uvicorn guardian:app --host 127.0.0.1 --port 8787 --workers 1
