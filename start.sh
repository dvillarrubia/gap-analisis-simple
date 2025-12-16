#!/bin/bash
# Content Gap Analysis V2 - Start Script

echo "============================================"
echo " Content Gap Analysis V2"
echo " Servidor de Analisis Semantico"
echo "============================================"
echo ""

cd "$(dirname "$0")"

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 no encontrado"
    echo "        Instala Python 3.10+ con: apt install python3 python3-venv python3-pip"
    exit 1
fi

# Crear entorno virtual si no existe
if [ ! -d "venv" ]; then
    echo "[INFO] Creando entorno virtual..."
    python3 -m venv venv
    echo "[INFO] Instalando dependencias..."
    ./venv/bin/pip install -r requirements.txt
    echo ""
fi

echo "[INFO] Iniciando servidor..."
echo "       URL: http://localhost:5000"
echo ""
echo "[CTRL+C] para detener"
echo "============================================"
echo ""

./venv/bin/python clustering_server.py
