@echo off
goto :start
-----------------------------------------------------------------------------
This file has been developed within the scope of the
Technical Director course at Filmakademie Baden-Wuerttemberg.
http://technicaldirector.de

Written by Lukas Kapp
Copyright (c) 2023 Animationsinstitut of Filmakademie Baden-Wuerttemberg
-----------------------------------------------------------------------------
:start

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
