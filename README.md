# Content Gap Analysis V2

Herramienta de analisis semantico de brechas de contenido. Compara las keywords de tu sitio con las de tus competidores para descubrir oportunidades de contenido usando clustering semantico con embeddings.

## Caracteristicas

- **Analisis semantico**: Usa embeddings de Sentence Transformers para agrupar keywords por significado
- **Comparacion competitiva**: Identifica keywords que cubren competidores y tu no
- **Visualizacion interactiva**: Interfaz web para explorar resultados
- **Filtrado semantico**: Filtra keywords por similitud con una query
- **Persistencia**: Guarda y carga analisis anteriores

## Requisitos

- Python 3.10 o 3.11
- 4GB+ RAM (8GB recomendado)
- GPU opcional (acelera el procesamiento)

## Instalacion Local

### Windows

1. **Opcion rapida**: Doble clic en `start.bat`
   - Crea el entorno virtual automaticamente
   - Instala dependencias
   - Inicia el servidor

2. **Manual**:
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python clustering_server.py
```

### Linux/Mac

```bash
chmod +x start.sh
./start.sh
```

O manualmente:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python clustering_server.py
```

## Uso

1. Inicia el servidor (ver arriba)
2. Abre `index.html` en tu navegador
3. Clic en "Abrir Aplicacion"
4. Carga tu archivo Excel (keywords propias)
5. Carga archivos de competidores
6. Clic en "Analizar Content Gap"

## Estructura de Archivos Excel

### Columnas requeridas

| Columna | Tipo | Descripcion |
|---------|------|-------------|
| Keyword | texto | Palabra clave |
| Topic | texto | Tema/categoria |

### Columnas opcionales

| Columna | Tipo | Descripcion |
|---------|------|-------------|
| Subtopic | texto | Subcategoria |
| Traffic | numero | Trafico organico |
| Volume | numero | Volumen de busqueda |
| Position | numero | Posicion en SERP |
| KD | numero | Dificultad (0-100) |
| URL | texto | URL posicionada |

## Despliegue en VPS con Docker

### 1. Subir archivos al servidor

```bash
scp -r ./* usuario@tu-servidor:/opt/content-gap/
```

### 2. Construir y ejecutar

```bash
cd /opt/content-gap
docker-compose up -d --build
```

### 3. Verificar

```bash
curl http://localhost:5000/health
```

### 4. Configurar Nginx (opcional)

```nginx
server {
    listen 80;
    server_name tu-dominio.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
    }
}
```

## Endpoints API

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/health` | GET | Estado del servidor |
| `/analyze_cluster_reinforcement_v2` | POST | Analisis principal |
| `/semantic_filter` | POST | Filtrar por similitud |
| `/list_analyses` | GET | Listar analisis guardados |
| `/load_analysis/{id}` | GET | Cargar un analisis |
| `/delete_analysis/{id}` | DELETE | Eliminar un analisis |

## Configuracion LLM (Opcional)

Copia `.env.example` a `.env` y configura:

```env
# Usar Ollama (local, gratis)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# O usar OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-tu-api-key
OPENAI_MODEL=gpt-4o-mini
```

## Estructura del Proyecto

```
content-gap-Analisis/
├── clustering_server.py   # Backend Flask
├── requirements.txt       # Dependencias Python
├── run_server.py          # Servidor produccion (Waitress)
├── cluster-reinforcement-v2.html  # Frontend principal
├── index.html             # Pagina de inicio
├── Dockerfile             # Imagen Docker
├── docker-compose.yml     # Orquestacion
├── .env.example           # Configuracion ejemplo
├── start.bat              # Inicio Windows
└── start.sh               # Inicio Linux/Mac
```

## Solucion de Problemas

### El servidor tarda mucho en iniciar

- La primera ejecucion descarga modelos (~1.5GB)
- Espera 1-2 minutos para que cargue el modelo

### Error de memoria

- Requiere minimo 4GB RAM libres
- Reduce `BATCH_SIZE` en clustering_server.py si hay problemas

### Puerto 5000 ocupado

```bash
# Linux
lsof -i :5000
kill -9 <PID>

# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

## Licencia

Uso interno.
