@echo off
chcp 65001 >nul
title Content Gap Analysis - Server

echo ============================================
echo  Content Gap Analysis V2
echo  Servidor de Analisis Semantico
echo ============================================
echo.

:: Verificar Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no encontrado en PATH
    echo         Instala Python 3.10+ desde python.org
    pause
    exit /b 1
)

cd /d "%~dp0"

:: Verificar entorno virtual
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Creando entorno virtual...
    python -m venv venv
    echo [INFO] Instalando dependencias...
    venv\Scripts\pip install -r requirements.txt
    echo.
)

echo [INFO] Iniciando servidor...
echo        URL: http://localhost:5000
echo        HTML: Abre index.html en tu navegador
echo.
echo [CTRL+C] para detener
echo ============================================
echo.

venv\Scripts\python clustering_server.py

pause
