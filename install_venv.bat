@echo off

REM Get the directory of the current batch script
set "SCRIPT_DIR=%~dp0"

echo Creating a new virtual environment...
python -m venv venv

echo Activating the virtual environment...
call .\venv\Scripts\activate

echo Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt

echo Setting up environment variable...
setx IRM_PATH "%SCRIPT_DIR%"

echo Setup completed.
pause
