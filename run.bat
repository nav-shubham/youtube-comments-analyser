@echo off
title YouTube Comment Scraper & Analyzer
echo ===================================================
echo   YouTube Comment Scraper & Analyzer (Powered by uv)
echo ===================================================
echo.
echo Launching Streamlit web application...
echo Press Ctrl+C in this window to stop the server.
echo.
uv run streamlit run app.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] The application failed to run or was terminated.
    echo Please make sure dependencies are fully installed.
    echo.
    pause
)
