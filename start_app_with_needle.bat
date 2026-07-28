@echo off
echo Starting Local Needle AI Engine...
cd /d "d:\cactus\needle"
start /B .\.venv\Scripts\python.exe needle_api_server.py

echo Local Needle AI Engine started on http://localhost:8000
echo Launching Web Application...
cd /d "d:\Projectai\Project.AI"
npm run dev
