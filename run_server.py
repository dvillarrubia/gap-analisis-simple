# ============================================
# SEO Clustering - Servidor de Produccion para Windows
# Usa Waitress como servidor WSGI (alternativa a Gunicorn)
# ============================================
#
# Uso:
#   python run_server.py
#
# Configuracion via variables de entorno o argumentos:
#   --host 0.0.0.0
#   --port 5000
#   --threads 4
#
# ============================================

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='SEO Clustering API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind (default: 5000)')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads (default: 4)')
    args = parser.parse_args()

    # Importar despues de parsear argumentos para mostrar ayuda rapido
    try:
        from waitress import serve
    except ImportError:
        print("Error: waitress no esta instalado.")
        print("Instalar con: pip install waitress")
        sys.exit(1)

    try:
        from clustering_server import app
    except ImportError as e:
        print(f"Error importando clustering_server: {e}")
        sys.exit(1)

    print("=" * 60)
    print("SEO Clustering & Content Gap Analysis API")
    print("=" * 60)
    print(f"  Host:    {args.host}")
    print(f"  Puerto:  {args.port}")
    print(f"  Threads: {args.threads}")
    print(f"  URL:     http://{args.host}:{args.port}")
    print("=" * 60)
    print("Presiona Ctrl+C para detener el servidor")
    print("")

    try:
        serve(
            app,
            host=args.host,
            port=args.port,
            threads=args.threads,
            url_scheme='http',
            ident='SEO-Clustering-API'
        )
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    except Exception as e:
        print(f"Error iniciando servidor: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
