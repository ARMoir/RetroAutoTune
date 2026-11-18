@echo off
cd /d "%~dp0"
py retro_autotune_gui.py
if errorlevel 1 (
  echo.
  echo If you see a missing module error, run INSTALL.bat first.
  pause
)
