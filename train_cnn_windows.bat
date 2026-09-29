@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe alzheimer_mri_cnn\alzheimer_mri_cnn.py train --data_dir "..\Alzheimer's dataset" --out results\mri_patient_split
pause
