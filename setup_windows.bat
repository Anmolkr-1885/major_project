@echo off
setlocal
cd /d "%~dp0"
echo Creating Python 3.11 virtual environment...
py -3.11 -m venv .venv
if errorlevel 1 (
  echo Could not create venv. Install Python 3.11 and ensure the py launcher is available.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto failed
pip install -r alzheimer_mri_cnn\requirements.txt -r alzheimer_device_model\requirements.txt
if errorlevel 1 goto failed
echo.
echo Setup complete. Open a new terminal in this folder and activate with:
echo .\.venv\Scripts\activate
pause
exit /b 0
:failed
echo Dependency installation failed. Review the error above.
pause
exit /b 1
