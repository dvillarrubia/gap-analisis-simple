#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Servidor Flask para Clustering Semántico con UMAP
Calcula centroides de topics y posiciones UMAP para visualización
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import umap
import sys
import io
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import threading
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

# Cargar variables de entorno
load_dotenv()

# Configurar salida UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = Flask(__name__)
CORS(app)  # Permitir requests desde el navegador

# ============================================
# OPTIMIZACIONES DE RENDIMIENTO
# ============================================
# 1. FP16 (half precision) - 1.5-2x más rápido
# 2. Batch size optimizado - mejor throughput
# 3. UMAP parámetros rápidos - menos iteraciones
# ============================================

# Configuración de optimización
USE_FP16 = True  # Half precision para velocidad
BATCH_SIZE = 64  # Batch size para encode (default: 32)
UMAP_N_EPOCHS = 200  # Menos épocas para UMAP (default: 500)

# Detectar dispositivo (GPU si disponible)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Dispositivo: {device.upper()}")

# Cargar modelo de embeddings (solo una vez)
# NOTA: Se usa exclusivamente el modelo español 768D por mayor precisión semántica
print("🔄 Cargando modelo de embeddings...")
print("   - Modelo español (768D): hiiamsid/sentence_similarity_spanish_es")
model_768 = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es', device=device)

# Aplicar FP16 si está habilitado (1.5-2x más rápido)
if USE_FP16 and device == 'cuda':
    model_768 = model_768.half()
    print("   - FP16 (half precision): ACTIVADO")
elif USE_FP16:
    # En CPU, usar torch.float16 es más lento, pero podemos optimizar de otra forma
    print("   - FP16: No disponible en CPU (requiere GPU)")

print(f"   - Batch size: {BATCH_SIZE}")
print(f"   - UMAP epochs: {UMAP_N_EPOCHS}")
print("✅ Modelo cargado (768 dimensiones) - OPTIMIZADO\n")

# Por compatibilidad, mantener 'model' como alias
model = model_768


def encode_optimized(texts, show_progress_bar=True):
    """Wrapper optimizado para encode con batch size grande"""
    if isinstance(texts, str):
        texts = [texts]
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True
    )


# Alias para compatibilidad - redirige al método optimizado
_original_encode = model_768.encode
def _optimized_encode_wrapper(texts, show_progress_bar=True, **kwargs):
    """Wrapper que intercepta llamadas a model.encode y aplica batch_size optimizado"""
    if isinstance(texts, str):
        texts = [texts]
    return _original_encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        **kwargs
    )

# Reemplazar el método encode del modelo con la versión optimizada
model_768.encode = _optimized_encode_wrapper
model.encode = _optimized_encode_wrapper
print("🚀 Encode optimizado con batch_size=64 aplicado globalmente")


def create_umap_fast(n_neighbors=15, min_dist=0.1, metric='cosine'):
    """Crear UMAP con parámetros optimizados para velocidad"""
    return umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        n_epochs=UMAP_N_EPOCHS,  # Menos épocas = más rápido
        low_memory=True,  # Usar menos RAM
        random_state=42
    )


# Monkey-patch UMAP para aplicar optimizaciones globalmente
_OriginalUMAP = umap.UMAP

class OptimizedUMAP(_OriginalUMAP):
    """UMAP optimizado con menos épocas y low_memory por defecto"""
    def __init__(self, *args, n_epochs=None, low_memory=None, **kwargs):
        # Aplicar optimizaciones si no se especifican
        if n_epochs is None:
            n_epochs = UMAP_N_EPOCHS
        if low_memory is None:
            low_memory = True
        super().__init__(*args, n_epochs=n_epochs, low_memory=low_memory, **kwargs)

# Reemplazar UMAP globalmente
umap.UMAP = OptimizedUMAP
print(f"🚀 UMAP optimizado con n_epochs={UMAP_N_EPOCHS} aplicado globalmente")


# ============================================
# 4. PRE-CÁLCULO EN SEGUNDO PLANO + CACHE SQLite
# ============================================
# Cache persistente en SQLite para embeddings
# - Persiste entre reinicios del servidor
# - No consume RAM (disco)
# - Escala a miles de archivos
# - Casi instantáneo para archivos repetidos
# ============================================
import sqlite3
import pickle
import zlib

# Configuración del cache
CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), 'embeddings_cache.db')
CACHE_TTL = 86400 * 7  # 7 días de vida
CACHE_MAX_ENTRIES = 1000  # Máximo número de entradas

# Estado de tareas de pre-cálculo en progreso (en memoria, temporal)
precalc_tasks = {}

# Thread pool para tareas en background
executor = ThreadPoolExecutor(max_workers=2)


def init_cache_db():
    """Inicializa la base de datos SQLite para cache"""
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embedding_cache (
            file_hash TEXT PRIMARY KEY,
            texts_compressed BLOB,
            embeddings_compressed BLOB,
            keyword_column TEXT,
            num_texts INTEGER,
            created_at REAL,
            last_accessed REAL,
            access_count INTEGER DEFAULT 1
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON embedding_cache(last_accessed)')

    # Tabla para cache de análisis completos (Refuerzo V2)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_cache (
            id TEXT PRIMARY KEY,
            name TEXT,
            own_file_name TEXT,
            own_file_hash TEXT,
            competitor_names TEXT,
            competitor_hashes TEXT,
            params TEXT,
            clusters_compressed BLOB,
            unassigned_compressed BLOB,
            summary_json TEXT,
            created_at REAL,
            last_accessed REAL
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_created ON analysis_cache(created_at DESC)')

    conn.commit()
    conn.close()
    print(f"💾 Cache SQLite inicializado: {CACHE_DB_PATH}")


def get_file_hash(file_content):
    """Genera hash único para el contenido del archivo"""
    if isinstance(file_content, bytes):
        return hashlib.md5(file_content).hexdigest()
    return hashlib.md5(str(file_content).encode()).hexdigest()


# === FUNCIONES DE CACHE PARA ANÁLISIS ===

def save_analysis_to_cache(analysis_id, name, own_file_name, own_file_hash,
                           competitor_names, competitor_hashes, params,
                           clusters, unassigned_gaps, summary):
    """Guarda un análisis completo en cache"""
    try:
        clusters_compressed = zlib.compress(pickle.dumps(clusters))
        unassigned_compressed = zlib.compress(pickle.dumps(unassigned_gaps))
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        now = time.time()
        cursor.execute('''
            INSERT OR REPLACE INTO analysis_cache
            (id, name, own_file_name, own_file_hash, competitor_names, competitor_hashes,
             params, clusters_compressed, unassigned_compressed, summary_json, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (analysis_id, name, own_file_name, own_file_hash,
              json.dumps(competitor_names), json.dumps(competitor_hashes),
              json.dumps(params), clusters_compressed, unassigned_compressed,
              json.dumps(summary), now, now))
        conn.commit()
        conn.close()
        print(f"💾 Análisis guardado: {analysis_id[:8]}...")
        return True
    except Exception as e:
        print(f"❌ Error guardando análisis: {e}")
        return False


def load_analysis_from_cache(analysis_id):
    """Carga un análisis desde cache"""
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT clusters_compressed, unassigned_compressed, summary_json,
                   name, own_file_name, competitor_names, created_at
            FROM analysis_cache WHERE id = ?
        ''', (analysis_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None
        cursor.execute('UPDATE analysis_cache SET last_accessed = ? WHERE id = ?',
                      (time.time(), analysis_id))
        conn.commit()
        conn.close()
        clusters = pickle.loads(zlib.decompress(row[0]))
        unassigned = pickle.loads(zlib.decompress(row[1]))
        summary = json.loads(row[2])
        return {
            'clusters': clusters,
            'unassigned_gaps': unassigned,
            'summary': summary,
            'meta': {
                'name': row[3],
                'own_file': row[4],
                'competitors': json.loads(row[5]),
                'created_at': row[6]
            }
        }
    except Exception as e:
        print(f"❌ Error cargando análisis: {e}")
        return None


def list_cached_analyses(limit=20):
    """Lista los análisis guardados en cache"""
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, name, own_file_name, competitor_names, summary_json, created_at
            FROM analysis_cache ORDER BY created_at DESC LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [{
            'id': row[0], 'name': row[1], 'own_file': row[2],
            'competitors': json.loads(row[3]), 'summary': json.loads(row[4]),
            'created_at': row[5]
        } for row in rows]
    except Exception as e:
        print(f"Error listando analisis: {e}")
        return []


def delete_analysis_from_cache(analysis_id):
    """Elimina un análisis del cache"""
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM analysis_cache WHERE id = ?', (analysis_id,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted > 0
    except Exception as e:
        print(f"❌ Error eliminando: {e}")
        return False




# ============================================================================
# CLASIFICADOR DE SEARCH INTENT POR EMBEDDINGS
# ============================================================================

# Ejemplos de referencia para cada tipo de intención (español)
INTENT_EXAMPLES = {
    'informational': [
        'qué es', 'cómo funciona', 'cómo hacer', 'por qué', 'guía de', 'tutorial',
        'definición de', 'significado de', 'para qué sirve', 'cuándo', 'dónde',
        'tipos de', 'características de', 'beneficios de', 'ventajas de',
        'ejemplos de', 'diferencia entre', 'historia de', 'origen de'
    ],
    'transactional': [
        'comprar', 'precio', 'oferta', 'descuento', 'barato', 'económico',
        'tienda', 'envío gratis', 'cupón', 'promoción', 'venta', 'pedido',
        'contratar', 'suscripción', 'tarifa', 'presupuesto', 'cotización'
    ],
    'commercial': [
        'mejor', 'mejores', 'top', 'comparativa', 'comparar', 'vs', 'versus',
        'review', 'reseña', 'opiniones', 'análisis', 'ranking', 'alternativas',
        'recomendaciones', 'cual elegir', 'merece la pena', 'vale la pena'
    ],
    'navigational': [
        'login', 'acceso', 'iniciar sesión', 'registro', 'contacto', 'teléfono',
        'dirección', 'horario', 'ubicación', 'oficial', 'web', 'app', 'descargar'
    ]
}

# ============================================================================
# CONTENT PATTERNS - Tipos de contenido para Semantic Gaps
# ============================================================================
CONTENT_PATTERNS = {
    'definition': ['qué es', 'que es', 'definición de', 'definicion de', 'significado de'],
    'how_it_works': ['cómo funciona', 'como funciona', 'cómo se hace', 'como se hace', 'funcionamiento de'],
    'tutorial': ['cómo hacer', 'como hacer', 'tutorial', 'paso a paso', 'guía de', 'guia de', 'manual de'],
    'comparison': ['vs', 'versus', 'diferencia entre', 'diferencias entre', 'comparativa', 'comparar', 'mejor que'],
    'list': ['tipos de', 'mejores', 'top', 'lista de', 'ejemplos de', 'clases de'],
    'review': ['opiniones', 'review', 'reseña', 'resena', 'experiencia con', 'análisis de', 'analisis de'],
    'price': ['precio', 'precios', 'cuánto cuesta', 'cuanto cuesta', 'tarifas', 'coste', 'costo'],
    'benefits': ['beneficios', 'ventajas', 'para qué sirve', 'para que sirve', 'por qué usar', 'por que usar']
}

PATTERN_LABELS = {
    'definition': 'Definicion',
    'how_it_works': 'Funcionamiento',
    'tutorial': 'Tutorial',
    'comparison': 'Comparativa',
    'list': 'Listado',
    'review': 'Review',
    'price': 'Precio',
    'benefits': 'Beneficios',
    'generic': 'Generico'
}

def classify_content_pattern(keyword):
    """Clasifica keyword por tipo de contenido"""
    kw_lower = keyword.lower()
    for pattern_type, triggers in CONTENT_PATTERNS.items():
        for trigger in triggers:
            if trigger in kw_lower:
                return pattern_type
    return 'generic'

def extract_core_concept(keyword):
    """Extrae el concepto base de una keyword quitando prefijos de patron"""
    kw_lower = keyword.lower()
    prefixes = [
        'qué es ', 'que es ', 'cómo hacer ', 'como hacer ',
        'cómo funciona ', 'como funciona ', 'mejores ', 'mejor ',
        'tipos de ', 'beneficios de ', 'ventajas de ',
        'precio de ', 'precios de ', 'opiniones de ', 'opiniones sobre ',
        'guía de ', 'guia de ', 'tutorial de ', 'tutorial para ',
        'diferencia entre ', 'comparativa de ', 'para qué sirve ', 'para que sirve '
    ]
    for prefix in prefixes:
        if kw_lower.startswith(prefix):
            return kw_lower[len(prefix):].strip()
    suffixes = [' vs ', ' versus ', ' o ']
    for suffix in suffixes:
        if suffix in kw_lower:
            return kw_lower.split(suffix)[0].strip()
    return kw_lower


def cluster_with_louvain(embeddings, similarity_threshold=0.5, resolution=1.0):
    """
    Clustering con algoritmo Louvain (detección de comunidades).

    Args:
        embeddings: np.array de embeddings (n_samples, n_features)
        similarity_threshold: umbral mínimo de similitud para crear arista
        resolution: parámetro de resolución (mayor = más clusters)

    Returns:
        labels: array de etiquetas de cluster
        n_clusters: número de clusters encontrados
    """
    try:
        import networkx as nx
        from community import community_louvain
        from sklearn.metrics.pairwise import cosine_similarity

        n_samples = len(embeddings)
        if n_samples < 2:
            return np.zeros(n_samples, dtype=int), 1

        # Calcular matriz de similitud
        sim_matrix = cosine_similarity(embeddings)

        # Construir grafo
        G = nx.Graph()
        G.add_nodes_from(range(n_samples))

        # Añadir aristas donde similitud > umbral
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if sim_matrix[i, j] >= similarity_threshold:
                    G.add_edge(i, j, weight=sim_matrix[i, j])

        # Verificar que el grafo tiene aristas
        if G.number_of_edges() == 0:
            # Si no hay aristas, reducir umbral progresivamente
            for thresh in [0.4, 0.3, 0.2, 0.1]:
                for i in range(n_samples):
                    for j in range(i + 1, n_samples):
                        if sim_matrix[i, j] >= thresh:
                            G.add_edge(i, j, weight=sim_matrix[i, j])
                if G.number_of_edges() > 0:
                    break

        # Si aún no hay aristas, cada nodo es su propio cluster
        if G.number_of_edges() == 0:
            return np.arange(n_samples), n_samples

        # Aplicar Louvain
        partition = community_louvain.best_partition(G, resolution=resolution, random_state=42)

        # Convertir a array de labels
        labels = np.array([partition[i] for i in range(n_samples)])
        n_clusters = len(set(labels))

        print(f"   Louvain: {n_clusters} clusters (threshold={similarity_threshold}, resolution={resolution})")
        return labels, n_clusters

    except ImportError as e:
        print(f"   ⚠️ Louvain no disponible: {e}")
        print(f"   Instalar con: pip install python-louvain networkx")
        # Fallback a un cluster único
        return np.zeros(len(embeddings), dtype=int), 1


def calculate_cluster_authority(own_d, gap_d, own_kws_count, gap_kws_count):
    """
    Calcula Authority Score (0-100) para un cluster basado en:
    - Coverage: 30% - % de espacio semantico cubierto
    - Intent Diversity: 25% - cuantos tipos de intent cubiertos (4 = 100%)
    - Position Strength: 25% - posicion promedio (<=3 = 100%, >=20 = 0%)
    - Volume Control: 20% - volumen propio vs competidores
    """
    total = own_kws_count + gap_kws_count
    coverage_score = (own_kws_count / total * 100) if total > 0 else 0

    intents_covered = set(o.get('intent', 'mixed') for o in own_d if o.get('intent') != 'mixed')
    intent_diversity_score = (len(intents_covered) / 4) * 100

    # Debug: mostrar campos disponibles en primer elemento
    if own_d and len(own_d) > 0:
        sample = own_d[0]
        print(f"   [DEBUG Authority] Campos disponibles: {list(sample.keys())}")
        print(f"   [DEBUG Authority] position={sample.get('position')}, volume={sample.get('volume')}")

    positions = [float(o.get('position')) for o in own_d if o.get('position') is not None and o.get('position') != '']
    avg_pos = np.mean(positions) if positions else 50
    print(f"   [DEBUG Authority] Posiciones encontradas: {len(positions)}, avg={avg_pos}")
    position_score = max(0, min(100, 100 - (avg_pos - 3) * (100 / 17)))

    own_volume = sum(float(o.get('volume', 0) or 0) for o in own_d)
    gap_volume = sum(float(g.get('volume', 0) or 0) for g in gap_d)
    total_volume = own_volume + gap_volume
    volume_score = (own_volume / total_volume * 100) if total_volume > 0 else 0

    authority_score = (
        coverage_score * 0.30 +
        intent_diversity_score * 0.25 +
        position_score * 0.25 +
        volume_score * 0.20
    )

    return {
        'authority_score': round(authority_score, 1),
        'coverage_score': round(coverage_score, 1),
        'intent_diversity_score': round(intent_diversity_score, 1),
        'intents_covered': list(intents_covered),
        'position_score': round(position_score, 1),
        'avg_position': round(avg_pos, 1),
        'volume_score': round(volume_score, 1),
        'own_volume': int(own_volume),
        'gap_volume': int(gap_volume)
    }


def detect_intra_cluster_cannibalization(own_keywords_data, own_embeddings, kws_own, threshold=0.95):
    """Detecta canibalizacion entre URLs propias dentro de un cluster."""
    from sklearn.metrics.pairwise import cosine_similarity

    url_groups = {}
    for i, kw_data in enumerate(own_keywords_data):
        url = kw_data.get('url', '')
        if not url or url == '' or pd.isna(url):
            continue
        url = str(url).strip().lower()
        if url not in url_groups:
            url_groups[url] = {'keywords': [], 'data': [], 'intents': []}

        kw = kw_data['keyword']
        if kw in kws_own:
            idx = kws_own.index(kw)
            url_groups[url]['keywords'].append(kw)
            url_groups[url]['data'].append({'idx': idx, **kw_data})
            url_groups[url]['intents'].append(kw_data.get('intent', 'mixed'))

    if len(url_groups) < 2:
        return {'pairs': [], 'summary': {'total_pairs': 0, 'critical_count': 0, 'high_count': 0, 'medium_count': 0, 'urls_affected': 0}}

    urls = list(url_groups.keys())
    cannibalization_pairs = []

    for i, url1 in enumerate(urls):
        for j, url2 in enumerate(urls):
            if j <= i:
                continue

            group1, group2 = url_groups[url1], url_groups[url2]
            if not group1['data'] or not group2['data']:
                continue

            idx1 = [d['idx'] for d in group1['data']]
            idx2 = [d['idx'] for d in group2['data']]
            emb1, emb2 = own_embeddings[idx1], own_embeddings[idx2]
            sim_matrix = cosine_similarity(emb1, emb2)

            high_sim_pairs = []
            for ki in range(len(group1['keywords'])):
                for kj in range(len(group2['keywords'])):
                    sim = sim_matrix[ki][kj]
                    if sim >= threshold:
                        intent1, intent2 = group1['intents'][ki], group2['intents'][kj]
                        high_sim_pairs.append({
                            'keyword1': group1['keywords'][ki], 'keyword2': group2['keywords'][kj],
                            'similarity': round(float(sim), 3), 'same_intent': intent1 == intent2,
                            'intent1': intent1, 'intent2': intent2
                        })

            if high_sim_pairs:
                avg_sim = np.mean([p['similarity'] for p in high_sim_pairs])
                same_intent_ratio = sum(1 for p in high_sim_pairs if p['same_intent']) / len(high_sim_pairs)

                if avg_sim >= 0.98 or (avg_sim >= 0.95 and same_intent_ratio > 0.8):
                    severity = 'critical'
                    recommendation = 'URGENTE: Consolidar contenido o establecer canonical'
                elif avg_sim >= 0.95 or same_intent_ratio > 0.5:
                    severity = 'high'
                    recommendation = 'Diferenciar angulos: una URL para cada intent diferente'
                else:
                    severity = 'medium'
                    recommendation = 'Monitorear: considerar internal linking entre URLs'

                cannibalization_pairs.append({
                    'url1': url1, 'url2': url2,
                    'conflicting_keywords': high_sim_pairs[:10],
                    'total_conflicts': len(high_sim_pairs),
                    'avg_similarity': round(avg_sim, 3),
                    'same_intent_ratio': round(same_intent_ratio, 2),
                    'severity': severity,
                    'recommendation': recommendation
                })

    severity_order = {'critical': 0, 'high': 1, 'medium': 2}
    cannibalization_pairs.sort(key=lambda x: (severity_order.get(x['severity'], 3), -x['total_conflicts']))

    return {
        'pairs': cannibalization_pairs,
        'summary': {
            'total_pairs': len(cannibalization_pairs),
            'critical_count': sum(1 for p in cannibalization_pairs if p['severity'] == 'critical'),
            'high_count': sum(1 for p in cannibalization_pairs if p['severity'] == 'high'),
            'medium_count': sum(1 for p in cannibalization_pairs if p['severity'] == 'medium'),
            'urls_affected': len(set(url for p in cannibalization_pairs for url in [p['url1'], p['url2']]))
        }
    }


def group_gaps_by_semantic_concept(gap_keywords_data, gap_embeddings, own_keywords_data, own_embeddings, kws_own):
    """Agrupa gaps por concepto semantico y detecta patrones de contenido faltantes."""
    from sklearn.cluster import AgglomerativeClustering
    from collections import Counter

    if len(gap_keywords_data) < 2:
        return []

    gap_concepts = []
    for i, gap in enumerate(gap_keywords_data):
        kw = gap.get('keyword', '')
        gap_concepts.append({
            'index': i, 'keyword': kw,
            'core_concept': extract_core_concept(kw),
            'content_pattern': classify_content_pattern(kw),
            'volume': gap.get('volume', 0) or 0
        })

    n_concepts = max(2, min(len(gap_keywords_data) // 3, 15))

    try:
        clustering = AgglomerativeClustering(n_clusters=n_concepts, metric='cosine', linkage='average')
        concept_labels = clustering.fit_predict(gap_embeddings)
    except:
        concept_labels = [i % n_concepts for i in range(len(gap_keywords_data))]

    concept_groups = {}
    for gc, label in zip(gap_concepts, concept_labels):
        if label not in concept_groups:
            concept_groups[label] = {'keywords': [], 'patterns_found': set(), 'core_concepts': [], 'total_volume': 0}
        concept_groups[label]['keywords'].append(gc)
        concept_groups[label]['patterns_found'].add(gc['content_pattern'])
        concept_groups[label]['core_concepts'].append(gc['core_concept'])
        concept_groups[label]['total_volume'] += gc['volume']

    own_patterns = {}
    for own_kw in own_keywords_data:
        kw = own_kw.get('keyword', '')
        pattern, core = classify_content_pattern(kw), extract_core_concept(kw)
        if core not in own_patterns:
            own_patterns[core] = set()
        own_patterns[core].add(pattern)

    result = []
    pattern_suggestions = {
        'definition': 'Crea una guia definitiva sobre "{}"',
        'how_it_works': 'Explica como funciona "{}"',
        'tutorial': 'Crea un tutorial paso a paso de "{}"',
        'comparison': 'Compara "{}" con alternativas',
        'list': 'Crea un listado/ranking de "{}"',
        'review': 'Publica reviews o casos de estudio de "{}"',
        'price': 'Crea una guia de precios de "{}"',
        'benefits': 'Explica los beneficios de "{}"'
    }

    for label, group in concept_groups.items():
        concept_counts = Counter(group['core_concepts'])
        concept_name = concept_counts.most_common(1)[0][0] if concept_counts else 'Concepto'

        your_patterns = set()
        for own_core, patterns in own_patterns.items():
            if own_core in group['core_concepts'] or any(own_core in gc for gc in group['core_concepts']):
                your_patterns.update(patterns)

        missing_patterns = group['patterns_found'] - your_patterns
        suggestions = [pattern_suggestions[p].format(concept_name) for p in missing_patterns if p in pattern_suggestions][:3]

        result.append({
            'concept_name': concept_name.title(),
            'gap_count': len(group['keywords']),
            'gap_keywords': [g['keyword'] for g in group['keywords'][:20]],
            'patterns_competitors_have': list(group['patterns_found']),
            'patterns_you_have': list(your_patterns),
            'patterns_missing': list(missing_patterns),
            'total_volume': int(group['total_volume']),
            'suggestions': suggestions
        })

    result.sort(key=lambda x: x['gap_count'] * (x['total_volume'] + 1), reverse=True)
    return result[:15]


# Cache global para embeddings de intención
_intent_embeddings_cache = {}

def get_intent_embeddings(model):
    """Genera y cachea embeddings para los ejemplos de cada intención"""
    global _intent_embeddings_cache
    if _intent_embeddings_cache:
        return _intent_embeddings_cache

    print("🎯 Generando embeddings de referencia para intención...")
    for intent, examples in INTENT_EXAMPLES.items():
        embeddings = model.encode(examples)  # Ya usa convert_to_numpy=True via wrapper
        # Promedio de todos los ejemplos como vector representativo
        _intent_embeddings_cache[intent] = embeddings.mean(axis=0)
    print("✅ Embeddings de intención generados")
    return _intent_embeddings_cache


def classify_intent(keyword, keyword_embedding, intent_embeddings):
    """Clasifica la intención de un keyword por similaridad con embeddings de referencia"""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # Calcular similaridad con cada intención
    similarities = {}
    for intent, ref_embedding in intent_embeddings.items():
        sim = cosine_similarity([keyword_embedding], [ref_embedding])[0][0]
        similarities[intent] = float(sim)

    # La intención con mayor similaridad
    best_intent = max(similarities, key=similarities.get)
    confidence = similarities[best_intent]

    # Si la confianza es muy baja, marcar como 'mixed'
    if confidence < 0.3:
        best_intent = 'mixed'

    return {
        'intent': best_intent,
        'confidence': round(confidence, 3),
        'scores': {k: round(v, 3) for k, v in similarities.items()}
    }


def classify_intents_batch(keywords, embeddings, model):
    """Clasifica intenciones para un lote de keywords"""
    intent_embs = get_intent_embeddings(model)
    results = []
    for i, kw in enumerate(keywords):
        result = classify_intent(kw, embeddings[i], intent_embs)
        results.append(result)
    return results


def save_to_cache(file_hash, texts, embeddings, keyword_column):
    """Guarda embeddings en cache SQLite (comprimidos)"""
    try:
        # Comprimir datos para ahorrar espacio
        texts_compressed = zlib.compress(pickle.dumps(texts))
        embeddings_compressed = zlib.compress(pickle.dumps(embeddings))

        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()

        now = time.time()
        cursor.execute('''
            INSERT OR REPLACE INTO embedding_cache
            (file_hash, texts_compressed, embeddings_compressed, keyword_column, num_texts, created_at, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1)
        ''', (file_hash, texts_compressed, embeddings_compressed, keyword_column, len(texts), now, now))

        conn.commit()
        conn.close()

        # Limpiar cache antiguo si hay demasiadas entradas
        clean_old_cache_db()

        print(f"💾 Cache SAVE: {len(texts)} embeddings guardados (hash: {file_hash[:12]}...)")
        return True
    except Exception as e:
        print(f"❌ Error guardando en cache: {e}")
        return False


def get_from_cache(file_hash):
    """Obtiene embeddings del cache SQLite"""
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT texts_compressed, embeddings_compressed, created_at, access_count
            FROM embedding_cache WHERE file_hash = ?
        ''', (file_hash,))

        row = cursor.fetchone()
        if row:
            texts_compressed, embeddings_compressed, created_at, access_count = row

            # Verificar TTL
            if time.time() - created_at > CACHE_TTL:
                cursor.execute('DELETE FROM embedding_cache WHERE file_hash = ?', (file_hash,))
                conn.commit()
                conn.close()
                return None, None

            # Actualizar last_accessed y access_count
            cursor.execute('''
                UPDATE embedding_cache
                SET last_accessed = ?, access_count = ?
                WHERE file_hash = ?
            ''', (time.time(), access_count + 1, file_hash))
            conn.commit()
            conn.close()

            # Descomprimir datos
            texts = pickle.loads(zlib.decompress(texts_compressed))
            embeddings = pickle.loads(zlib.decompress(embeddings_compressed))

            print(f"📦 Cache HIT: {len(texts)} embeddings recuperados (hash: {file_hash[:12]}..., accesos: {access_count + 1})")
            return embeddings, texts

        conn.close()
        return None, None
    except Exception as e:
        print(f"❌ Error leyendo cache: {e}")
        return None, None


def clean_old_cache_db():
    """Limpia entradas antiguas del cache"""
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()

        # Eliminar entradas expiradas
        cursor.execute('DELETE FROM embedding_cache WHERE ? - created_at > ?', (time.time(), CACHE_TTL))

        # Si hay demasiadas entradas, eliminar las menos usadas
        cursor.execute('SELECT COUNT(*) FROM embedding_cache')
        count = cursor.fetchone()[0]

        if count > CACHE_MAX_ENTRIES:
            # Eliminar las menos accedidas
            cursor.execute('''
                DELETE FROM embedding_cache WHERE file_hash IN (
                    SELECT file_hash FROM embedding_cache
                    ORDER BY access_count ASC, last_accessed ASC
                    LIMIT ?
                )
            ''', (count - CACHE_MAX_ENTRIES,))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Error limpiando cache: {e}")


def get_cache_stats():
    """Obtiene estadísticas del cache"""
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*), SUM(num_texts), SUM(LENGTH(embeddings_compressed)) FROM embedding_cache')
        count, total_texts, total_size = cursor.fetchone()

        cursor.execute('SELECT file_hash, num_texts, access_count, created_at FROM embedding_cache ORDER BY last_accessed DESC LIMIT 10')
        recent = cursor.fetchall()

        conn.close()

        return {
            'entries': count or 0,
            'total_keywords': total_texts or 0,
            'size_mb': round((total_size or 0) / 1024 / 1024, 2),
            'max_entries': CACHE_MAX_ENTRIES,
            'ttl_days': CACHE_TTL // 86400,
            'recent': [
                {'hash': r[0][:12], 'keywords': r[1], 'accesses': r[2], 'age_hours': round((time.time() - r[3]) / 3600, 1)}
                for r in recent
            ]
        }
    except Exception as e:
        return {'error': str(e)}


def precalculate_embeddings_task(file_hash, texts, keyword_column):
    """Tarea en background para pre-calcular embeddings y guardar en SQLite"""
    global precalc_tasks
    try:
        precalc_tasks[file_hash] = {'status': 'running', 'progress': 10}

        # Calcular embeddings
        print(f"⏳ Pre-calculando embeddings para {len(texts)} textos (hash: {file_hash[:12]}...)")
        embeddings = model.encode(texts, show_progress_bar=False)

        precalc_tasks[file_hash] = {'status': 'running', 'progress': 80}

        # Guardar en SQLite
        save_to_cache(file_hash, texts, embeddings, keyword_column)

        precalc_tasks[file_hash] = {'status': 'done', 'progress': 100}
        print(f"✅ Pre-cálculo completado y guardado en SQLite (hash: {file_hash[:12]}...)")

    except Exception as e:
        precalc_tasks[file_hash] = {'status': 'error', 'progress': 0, 'error': str(e)}
        print(f"❌ Error en pre-cálculo: {e}")


def get_cached_embeddings(file_hash):
    """Wrapper para obtener embeddings (primero cache SQLite)"""
    return get_from_cache(file_hash)


def init_keyword_cache_db():
    """Inicializa tabla de caché por keyword individual"""
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS keyword_embeddings (
            keyword TEXT PRIMARY KEY,
            embedding BLOB,
            created_at REAL,
            last_accessed REAL,
            access_count INTEGER DEFAULT 1
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_kw_accessed ON keyword_embeddings(last_accessed)')
    conn.commit()
    conn.close()

def get_cached_keywords(keywords):
    """
    Busca keywords en caché.
    Retorna: dict {keyword: embedding} para las encontradas
    """
    if not keywords:
        return {}

    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()

        # Buscar en lotes para evitar límite de parámetros SQLite
        cached = {}
        batch_size = 500
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i+batch_size]
            placeholders = ','.join(['?' for _ in batch])
            cursor.execute(f'''
                SELECT keyword, embedding FROM keyword_embeddings
                WHERE keyword IN ({placeholders})
            ''', batch)

            for row in cursor.fetchall():
                keyword, emb_blob = row
                embedding = pickle.loads(zlib.decompress(emb_blob))
                cached[keyword] = embedding

        # Actualizar access_count para las encontradas
        if cached:
            now = time.time()
            for kw in cached.keys():
                cursor.execute('''
                    UPDATE keyword_embeddings
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE keyword = ?
                ''', (now, kw))
            conn.commit()

        conn.close()
        return cached
    except Exception as e:
        print(f"⚠️ Error leyendo caché de keywords: {e}")
        return {}

def save_keywords_to_cache(keywords, embeddings):
    """
    Guarda keywords y embeddings individuales en caché.
    """
    if len(keywords) != len(embeddings):
        return

    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        now = time.time()

        for kw, emb in zip(keywords, embeddings):
            emb_compressed = zlib.compress(pickle.dumps(emb))
            cursor.execute('''
                INSERT OR REPLACE INTO keyword_embeddings
                (keyword, embedding, created_at, last_accessed, access_count)
                VALUES (?, ?, ?, ?, 1)
            ''', (kw, emb_compressed, now, now))

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Error guardando caché de keywords: {e}")

def get_keyword_cache_stats():
    """Estadísticas del caché de keywords"""
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM keyword_embeddings')
        count = cursor.fetchone()[0]
        conn.close()
        return {'total_keywords': count}
    except:
        return {'total_keywords': 0}

def encode_with_cache(texts, model=None):
    """
    Genera embeddings con caché SQLite POR KEYWORD INDIVIDUAL.
    Solo vectoriza las keywords nuevas que no están en caché.
    """
    if model is None:
        model = model_768

    # Asegurar que la tabla existe
    init_keyword_cache_db()

    # Buscar cuáles ya están en caché
    cached = get_cached_keywords(texts)
    cached_count = len(cached)

    # Identificar keywords nuevas (no cacheadas)
    new_keywords = [t for t in texts if t not in cached]

    if cached_count > 0:
        print(f"⚡ Cache HIT: {cached_count}/{len(texts)} keywords recuperadas de SQLite")

    # Generar embeddings solo para las nuevas
    if new_keywords:
        print(f"💫 Vectorizando {len(new_keywords)} keywords nuevas...")
        new_embeddings = model.encode(new_keywords, show_progress_bar=True)

        # Guardar las nuevas en caché
        save_keywords_to_cache(new_keywords, new_embeddings)
        print(f"💾 {len(new_keywords)} keywords guardadas en caché SQLite")

        # Añadir al diccionario de cacheadas
        for kw, emb in zip(new_keywords, new_embeddings):
            cached[kw] = emb
    else:
        print(f"✅ Todas las {len(texts)} keywords estaban en caché!")

    # Construir array de embeddings en el orden original
    embedding_dim = list(cached.values())[0].shape[0] if cached else 768
    result = np.zeros((len(texts), embedding_dim))
    for i, t in enumerate(texts):
        result[i] = cached[t]

    return result


def wait_for_precalc(file_hash, timeout=120):
    """Espera a que termine el pre-cálculo si está en progreso"""
    start = time.time()
    while file_hash in precalc_tasks:
        task = precalc_tasks[file_hash]
        if task['status'] == 'done':
            return True
        if task['status'] == 'error':
            return False
        if time.time() - start > timeout:
            return False
        time.sleep(0.5)
    # Verificar si está en cache
    embeddings, _ = get_from_cache(file_hash)
    return embeddings is not None


# Inicializar cache SQLite al arrancar
init_cache_db()
print("🚀 Sistema de pre-cálculo con cache SQLite activado")

# Alias para compatibilidad (cache en memoria ya no se usa, pero mantenemos variable)
embedding_cache = {}


# Configuración de LLM para naming de clusters
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'none').lower()
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.3'))
MAX_PAGES_FOR_NAMING = int(os.getenv('MAX_PAGES_FOR_NAMING', '10'))

# Inicializar clientes de LLM
llm_client = None
if LLM_PROVIDER == 'ollama':
    try:
        import ollama
        OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
        # Intentar conexión
        ollama.list()
        llm_client = 'ollama'
        print(f"✅ LLM Ollama configurado: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")
    except Exception as e:
        print(f"⚠️  Ollama no disponible: {e}")
        print(f"   Usando sistema regex para nombres de clusters")
        LLM_PROVIDER = 'none'
elif LLM_PROVIDER == 'openai':
    try:
        from openai import OpenAI
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY no configurada")
        llm_client = OpenAI(api_key=OPENAI_API_KEY)
        print(f"✅ LLM OpenAI configurado: {OPENAI_MODEL}")
    except Exception as e:
        print(f"⚠️  OpenAI no disponible: {e}")
        print(f"   Usando sistema regex para nombres de clusters")
        LLM_PROVIDER = 'none'
else:
    print("ℹ️  LLM desactivado - usando sistema regex para nombres de clusters")

# Configuración de Supabase PostgreSQL
SUPABASE_CONFIG = {
    'host': 'localhost',
    'port': 54323,
    'database': 'postgres',
    'user': 'postgres',
    'password': 'S8kP9mN3vT2xQ7rW5jH4fL1aZ6cV0bN8yM3xR5pK9wT2sD7gF4hJ6lA1qE3uI8oY'
}

# Almacenar estado del último UMAP calculado
umap_state = {
    'reducer': None,
    'embeddings': None,
    'vector_type': None
}

def get_db_connection():
    """Crear conexión a la base de datos Supabase"""
    try:
        conn = psycopg2.connect(**SUPABASE_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Error conectando a la base de datos: {str(e)}")
        raise

def extract_embeddings(df, prefix):
    """Extrae embeddings de un DataFrame según el prefijo"""
    embed_cols = [col for col in df.columns if col.startswith(prefix)]
    if not embed_cols:
        return None
    embeddings = df[embed_cols].values
    return embeddings

def calculate_umap_2d(embeddings, n_neighbors=8, min_dist=0.4):
    """Calcula UMAP en 2D con parámetros optimizados para mayor separación semántica"""
    print(f"🔄 Calculando UMAP con {len(embeddings)} puntos...")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=42
    )

    embedding_2d = reducer.fit_transform(embeddings)
    print("✅ UMAP completado")

    return embedding_2d

def generate_cluster_name_with_llm(pages_sample, keywords_list):
    """
    Usa un LLM (Ollama o OpenAI) para generar un nombre descriptivo del cluster.

    Args:
        pages_sample: Lista de diccionarios con 'title' y 'h1' de páginas del cluster
        keywords_list: Lista de palabras clave del cluster (fallback)

    Returns:
        str: Nombre descriptivo del cluster
    """
    if LLM_PROVIDER == 'none' or not pages_sample:
        # Fallback al sistema regex
        return generate_natural_cluster_name_regex(keywords_list)

    # Preparar contexto para el LLM
    sample_titles = [p.get('title', '') for p in pages_sample[:MAX_PAGES_FOR_NAMING] if p.get('title')]
    sample_h1s = [p.get('h1', '') for p in pages_sample[:MAX_PAGES_FOR_NAMING] if p.get('h1')]

    if not sample_titles and not sample_h1s:
        return generate_natural_cluster_name_regex(keywords_list)

    # Crear prompt
    context_titles = '\n'.join(f"  - {t}" for t in sample_titles[:5])
    context_h1s = '\n'.join(f"  - {h}" for h in sample_h1s[:5])
    keywords_str = ', '.join(keywords_list[:5])

    prompt = f"""Analiza este cluster de páginas web y genera un nombre descriptivo corto (máximo 6 palabras) en español.

Títulos de páginas:
{context_titles}

H1s de páginas:
{context_h1s}

Palabras clave detectadas: {keywords_str}

Reglas:
- Máximo 6 palabras
- Usar lenguaje natural y profesional
- Enfocarse en el tema principal común
- NO uses dos puntos (:)
- NO enumeres ejemplos
- Responde SOLO con el nombre, sin explicaciones

Nombre del cluster:"""

    try:
        if LLM_PROVIDER == 'ollama':
            import ollama
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': LLM_TEMPERATURE}
            )
            cluster_name = response['message']['content'].strip()

        elif LLM_PROVIDER == 'openai':
            response = llm_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=LLM_TEMPERATURE,
                max_tokens=50
            )
            cluster_name = response.choices[0].message.content.strip()

        # Limpiar el nombre generado
        cluster_name = cluster_name.replace('"', '').replace("'", "").strip()

        # Si es muy largo, truncar
        if len(cluster_name.split()) > 8:
            cluster_name = ' '.join(cluster_name.split()[:8])

        return cluster_name if cluster_name else generate_natural_cluster_name_regex(keywords_list)

    except Exception as e:
        print(f"⚠️  Error al generar nombre con LLM: {e}")
        return generate_natural_cluster_name_regex(keywords_list)


def generate_natural_cluster_name_regex(keywords_list, max_words=5):
    """
    Genera un nombre natural para un cluster basado en sus palabras clave principales.
    Sistema de fallback con regex patterns.

    Estrategias:
    1. Templates de lenguaje natural
    2. Análisis de patrones comunes
    3. Generación de frases coherentes
    """
    if not keywords_list or len(keywords_list) == 0:
        return "Contenido General"

    # Limpiar y filtrar palabras
    keywords = [k.strip().lower() for k in keywords_list[:max_words]]

    # Detectar patrones comunes y generar nombres naturales
    patterns = {
        # Verbos de acción
        r'\b(cómo|como|guía|tutorial|aprender)\b': 'Guías de {}',
        r'\b(qué|que|definición|significado)\b': '¿Qué es {}?',
        r'\b(mejores|mejor|top|ranking)\b': 'Los mejores {}',
        r'\b(herramientas|tools|software)\b': 'Herramientas de {}',
        r'\b(consejos|tips|trucos)\b': 'Consejos sobre {}',
        r'\b(ejemplos|casos)\b': 'Ejemplos de {}',
        r'\b(crear|hacer|implementar)\b': 'Cómo crear {}',
        r'\b(optimizar|mejorar)\b': 'Optimización de {}',
        r'\b(estrategias|tácticas)\b': 'Estrategias de {}',
        r'\b(análisis|auditoría)\b': 'Análisis de {}',

        # Temas específicos (SEO/Marketing)
        r'\b(keywords|palabras|clave)\b': 'Investigación de Palabras Clave',
        r'\b(backlinks|enlaces)\b': 'Estrategia de Link Building',
        r'\b(contenido|content)\b': 'Marketing de Contenidos',
        r'\b(redes|sociales|social)\b': 'Redes Sociales',
        r'\b(email|correo)\b': 'Email Marketing',
        r'\b(conversión|conversion)\b': 'Optimización de Conversiones',
        r'\b(analítica|analytics)\b': 'Analítica Web',
        r'\b(ppc|ads|publicidad)\b': 'Publicidad Digital',
        r'\b(local|google|maps)\b': 'SEO Local',
        r'\b(móvil|mobile)\b': 'Optimización Móvil',
    }

    # Buscar patrones en las palabras clave
    import re
    keywords_text = ' '.join(keywords)

    for pattern, template in patterns.items():
        if re.search(pattern, keywords_text):
            # Obtener las palabras restantes para completar el template
            remaining = [k for k in keywords if not re.search(pattern, k)]
            if remaining:
                if '{}' in template:
                    return template.format(remaining[0].title())
                else:
                    return template

    # Si no hay patrón, generar nombre basado en cantidad de palabras
    if len(keywords) == 1:
        return keywords[0].title()
    elif len(keywords) == 2:
        return f"{keywords[0].title()} y {keywords[1].title()}"
    elif len(keywords) == 3:
        return f"{keywords[0].title()}, {keywords[1].title()} y {keywords[2].title()}"
    else:
        # Para 4+ palabras, crear una frase más natural
        main_topic = keywords[0].title()
        subtopics = ', '.join([k.title() for k in keywords[1:3]])
        return f"{main_topic}: {subtopics}"

def generate_natural_meta_cluster_name(cluster_names, page_count):
    """
    Genera un nombre natural para un meta-cluster basado en los nombres de sus clusters.

    Analiza los temas comunes y genera una descripción de alto nivel.
    """
    if not cluster_names or len(cluster_names) == 0:
        return "Contenido General"

    # Extraer todas las palabras de los nombres de clusters
    all_words = []
    for name in cluster_names:
        # Limpiar puntuación y caracteres especiales
        import re
        cleaned = re.sub(r'[^\w\s]', ' ', name.lower())
        words = cleaned.split()
        # Filtrar palabras comunes (stop words básicas)
        stop_words = {'de', 'la', 'el', 'en', 'y', 'a', 'los', 'las', 'del', 'un', 'una',
                     'para', 'con', 'por', 'que', 'qué', 'cómo', 'como', 'sobre', 'más'}
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
        all_words.extend(filtered_words)

    # Contar frecuencias
    from collections import Counter
    word_freq = Counter(all_words)

    # Obtener las 3 palabras más frecuentes
    top_words = [word for word, count in word_freq.most_common(3)]

    if not top_words:
        return f"Zona de Contenido ({page_count} páginas)"

    # Detectar categorías temáticas amplias
    categories = {
        'seo': ['seo', 'keywords', 'posicionamiento', 'ranking', 'google', 'búsqueda',
               'backlinks', 'enlaces', 'indexación', 'rastreo'],
        'contenido': ['contenido', 'content', 'artículos', 'blog', 'posts', 'escritura',
                     'redacción', 'editorial'],
        'técnico': ['técnico', 'código', 'desarrollo', 'web', 'html', 'css', 'javascript',
                   'performance', 'velocidad', 'optimización'],
        'social': ['redes', 'sociales', 'facebook', 'twitter', 'instagram', 'linkedin',
                  'social', 'comunidad'],
        'analítica': ['analytics', 'analítica', 'datos', 'métricas', 'kpi', 'medición',
                     'reportes', 'informes'],
        'marketing': ['marketing', 'publicidad', 'ads', 'ppc', 'campaña', 'promoción',
                     'estrategia', 'conversión'],
        'ecommerce': ['ecommerce', 'tienda', 'productos', 'ventas', 'compra', 'carrito',
                     'checkout', 'pedidos'],
        'local': ['local', 'maps', 'negocios', 'gmb', 'ficha', 'ubicación', 'dirección'],
    }

    # Detectar categoría principal
    detected_category = None
    max_matches = 0

    for category, keywords in categories.items():
        matches = sum(1 for word in top_words if word in keywords)
        if matches > max_matches:
            max_matches = matches
            detected_category = category

    # Generar nombre basado en categoría detectada
    category_names = {
        'seo': 'SEO y Posicionamiento Web',
        'contenido': 'Marketing de Contenidos',
        'técnico': 'Aspectos Técnicos del Sitio',
        'social': 'Redes Sociales y Comunidad',
        'analítica': 'Analítica y Medición',
        'marketing': 'Marketing Digital',
        'ecommerce': 'E-commerce y Ventas Online',
        'local': 'SEO Local y Negocio',
    }

    if detected_category and max_matches >= 1:
        return category_names[detected_category]

    # Si no hay categoría clara, usar las palabras más frecuentes
    if len(top_words) == 1:
        return f"Contenido sobre {top_words[0].title()}"
    elif len(top_words) == 2:
        return f"{top_words[0].title()} y {top_words[1].title()}"
    else:
        return f"{top_words[0].title()}, {top_words[1].title()} y {top_words[2].title()}"

def calculate_topic_centroids(df, embeddings, umap_coords):
    """Calcula centroides de cada topic en el espacio UMAP"""
    centroids = []

    # Agrupar por topic
    topics = df['Topic'].unique()

    for topic in topics:
        mask = df['Topic'] == topic
        topic_coords = umap_coords[mask]

        # Centroide = promedio de posiciones
        centroid_x = np.mean(topic_coords[:, 0])
        centroid_y = np.mean(topic_coords[:, 1])

        # Contar keywords en el topic
        count = np.sum(mask)

        # Calcular dispersión (desviación estándar)
        dispersion = np.std(topic_coords, axis=0).mean()

        centroids.append({
            'topic': topic,
            'x': float(centroid_x),
            'y': float(centroid_y),
            'count': int(count),
            'dispersion': float(dispersion)
        })

    return centroids

def vectorize_general_topics(general_topics, target_dim=None):
    """Vectoriza topics generales usando el modelo español 768D"""
    print(f"🔄 Vectorizando {len(general_topics)} topics generales...")

    # Siempre usar modelo español 768D (mayor precisión)
    print("   Usando modelo español 768D")
    embeddings = model_768.encode(general_topics)

    print(f"✅ Topics vectorizados: {embeddings.shape}")
    return embeddings

@app.route('/process_excel', methods=['POST'])
def process_excel():
    """Endpoint principal: procesa Excel y retorna datos para visualización"""
    try:
        # Obtener archivo
        if 'file' not in request.files:
            return jsonify({'error': 'No se recibió archivo'}), 400

        file = request.files['file']
        vector_type = request.form.get('vector_type', 'keyword')
        n_neighbors = int(request.form.get('n_neighbors', 8))
        min_dist = float(request.form.get('min_dist', 0.4))
        vectorize_file = request.form.get('vectorize', 'false').lower() == 'true'

        print(f"\n📂 Procesando archivo: {file.filename}")
        print(f"   Vector type: {vector_type}")
        print(f"   UMAP params: n_neighbors={n_neighbors}, min_dist={min_dist}")
        print(f"   Vectorizar archivo: {vectorize_file}")

        # Leer Excel
        df = pd.read_excel(file)
        print(f"✅ Excel leído: {len(df)} filas")

        # Si hay que vectorizar, generar embeddings
        if vectorize_file:
            print("🔄 Vectorizando Keywords, Topics y Subtopics...")

            # Vectorizar Keywords
            if 'Keyword' in df.columns:
                keyword_embeddings = model.encode(df['Keyword'].fillna('').tolist())
                for i in range(keyword_embeddings.shape[1]):
                    df[f'keyword_embed_{i}'] = keyword_embeddings[:, i]
                print(f"   ✅ Keywords vectorizadas: {keyword_embeddings.shape[1]} dimensiones")

            # Vectorizar Topics
            if 'Topic' in df.columns:
                topic_embeddings = model.encode(df['Topic'].fillna('').tolist())
                for i in range(topic_embeddings.shape[1]):
                    df[f'topic_embed_{i}'] = topic_embeddings[:, i]
                print(f"   ✅ Topics vectorizados: {topic_embeddings.shape[1]} dimensiones")

            # Vectorizar Subtopics
            if 'Subtopic' in df.columns:
                subtopic_embeddings = model.encode(df['Subtopic'].fillna('').tolist())
                for i in range(subtopic_embeddings.shape[1]):
                    df[f'subtopic_embed_{i}'] = subtopic_embeddings[:, i]
                print(f"   ✅ Subtopics vectorizados: {subtopic_embeddings.shape[1]} dimensiones")

        # Extraer embeddings
        prefix = f'{vector_type}_embed_'
        embeddings = extract_embeddings(df, prefix)

        if embeddings is None:
            return jsonify({'error': f'No se encontraron columnas {prefix}*. ¿Activaste la opción de vectorizar?'}), 400

        print(f"✅ Embeddings extraídos: {embeddings.shape}")

        # Calcular UMAP y guardar el reducer para usarlo después
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine',
            random_state=42
        )
        umap_coords = reducer.fit_transform(embeddings)

        # Guardar estado para topics generales
        umap_state['reducer'] = reducer
        umap_state['embeddings'] = embeddings
        umap_state['vector_type'] = vector_type

        # Calcular centroides de topics
        centroids = calculate_topic_centroids(df, embeddings, umap_coords)

        # Preparar datos de keywords
        keywords_data = []
        for idx, row in df.iterrows():
            keywords_data.append({
                'id': int(idx),
                'keyword': str(row.get('Keyword', '')),
                'traffic': int(row.get('Traffic', 0)),
                'volume': int(row.get('Volume', 0)),
                'position': int(row.get('Position', 0)),
                'kd': float(row.get('KD', 0)),
                'topic': str(row.get('Topic', 'Miscellaneous')),
                'subtopic': str(row.get('Subtopic', '')),
                'url': str(row.get('URL', '')),
                'x': float(umap_coords[idx, 0]),
                'y': float(umap_coords[idx, 1])
            })

        print(f"✅ Datos preparados: {len(keywords_data)} keywords, {len(centroids)} centroides")

        return jsonify({
            'keywords': keywords_data,
            'centroids': centroids,
            'topics': list(df['Topic'].unique())
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/vectorize_topics', methods=['POST'])
def vectorize_topics():
    """Endpoint para vectorizar topics generales y posicionarlos en el mapa existente"""
    try:
        data = request.json
        topics = data.get('topics', [])
        vector_type = data.get('vector_type', 'keyword')

        if not topics:
            return jsonify({'error': 'No se recibieron topics'}), 400

        # Verificar que hay un UMAP previo
        if umap_state['reducer'] is None:
            return jsonify({'error': 'Primero debes cargar datos para generar el espacio UMAP'}), 400

        print(f"\n🎯 Vectorizando {len(topics)} topics generales")
        print(f"   Topics: {topics}")

        # Detectar dimensionalidad de los embeddings existentes
        existing_dim = umap_state['embeddings'].shape[1]
        print(f"   Dimensiones requeridas: {existing_dim}D")

        # Vectorizar topics usando el modelo apropiado
        topic_embeddings = vectorize_general_topics(topics, target_dim=existing_dim)

        # Verificar que las dimensiones coincidan
        new_dim = topic_embeddings.shape[1]

        # Si las dimensiones no coinciden, hay un problema
        if existing_dim != new_dim:
            print(f"⚠️ Incompatibilidad de dimensiones ({existing_dim}D vs {new_dim}D)")
            print("   Recalculando UMAP completo con todos los puntos...")

            # Combinar embeddings existentes y nuevos
            all_embeddings = np.vstack([umap_state['embeddings'], topic_embeddings])

            # Recalcular UMAP con todos los puntos
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, len(all_embeddings) - 1),
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            all_coords = reducer.fit_transform(all_embeddings)

            # Los topics personalizados están al final
            coords_2d = all_coords[-len(topics):]
        else:
            # Usar el UMAP ya entrenado para transformar los nuevos puntos
            print("🔄 Proyectando topics en el espacio UMAP existente...")
            try:
                coords_2d = umap_state['reducer'].transform(topic_embeddings)
            except Exception as e:
                print(f"⚠️ Error en transform: {str(e)}")
                print("   Recalculando UMAP completo...")
                # Si transform falla, combinar y recalcular
                all_embeddings = np.vstack([umap_state['embeddings'], topic_embeddings])
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric='cosine',
                    random_state=42
                )
                all_coords = reducer.fit_transform(all_embeddings)
                coords_2d = all_coords[-len(topics):]

        general_points = []
        for i, topic in enumerate(topics):
            general_points.append({
                'topic': topic,
                'x': float(coords_2d[i, 0]),
                'y': float(coords_2d[i, 1]),
                'is_general': True
            })

        print(f"✅ Topics generales vectorizados y posicionados en el mapa")

        return jsonify({
            'general_topics': general_points
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/process_excel_simple', methods=['POST'])
def process_excel_simple():
    """Endpoint para formato simple: Cluster + Keyword con métricas SEMrush"""
    try:
        # Obtener archivo
        if 'file' not in request.files:
            return jsonify({'error': 'No se recibió archivo'}), 400

        file = request.files['file']
        vector_type = request.form.get('vector_type', 'keyword')
        n_neighbors = int(request.form.get('n_neighbors', 8))
        min_dist = float(request.form.get('min_dist', 0.4))
        vectorize_file = request.form.get('vectorize', 'true').lower() == 'true'

        print(f"\n📂 Procesando archivo SIMPLE: {file.filename}")
        print(f"   Vector type: {vector_type}")
        print(f"   UMAP params: n_neighbors={n_neighbors}, min_dist={min_dist}")
        print(f"   Vectorizar archivo: {vectorize_file}")

        # Leer Excel
        df = pd.read_excel(file)
        print(f"✅ Excel leído: {len(df)} filas")
        print(f"   Columnas detectadas: {list(df.columns)}")

        # Verificar columnas requeridas
        required_cols = ['Cluster', 'Keyword']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return jsonify({'error': f'Faltan columnas requeridas: {missing}'}), 400

        # Si hay que vectorizar, generar embeddings
        if vectorize_file:
            print("🔄 Vectorizando Clusters y Keywords...")

            # Vectorizar Clusters
            if 'Cluster' in df.columns:
                cluster_embeddings = model.encode(df['Cluster'].fillna('').astype(str).tolist())
                for i in range(cluster_embeddings.shape[1]):
                    df[f'cluster_embed_{i}'] = cluster_embeddings[:, i]
                print(f"   ✅ Clusters vectorizados: {cluster_embeddings.shape[1]} dimensiones")

            # Vectorizar Keywords
            if 'Keyword' in df.columns:
                keyword_embeddings = model.encode(df['Keyword'].fillna('').astype(str).tolist())
                for i in range(keyword_embeddings.shape[1]):
                    df[f'keyword_embed_{i}'] = keyword_embeddings[:, i]
                print(f"   ✅ Keywords vectorizadas: {keyword_embeddings.shape[1]} dimensiones")

        # Extraer embeddings según el tipo seleccionado
        prefix = f'{vector_type}_embed_'
        embeddings = extract_embeddings(df, prefix)

        if embeddings is None:
            return jsonify({'error': f'No se encontraron columnas {prefix}*. ¿Activaste la opción de vectorizar?'}), 400

        print(f"✅ Embeddings extraídos: {embeddings.shape}")

        # Calcular UMAP
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine',
            random_state=42
        )
        umap_coords = reducer.fit_transform(embeddings)

        # Guardar estado para topics generales
        umap_state['reducer'] = reducer
        umap_state['embeddings'] = embeddings
        umap_state['vector_type'] = vector_type

        # Calcular centroides de clusters (equivalente a topics)
        centroids = []
        clusters = df['Cluster'].unique()

        for cluster in clusters:
            mask = df['Cluster'] == cluster
            cluster_coords = umap_coords[mask]

            centroid_x = np.mean(cluster_coords[:, 0])
            centroid_y = np.mean(cluster_coords[:, 1])
            count = np.sum(mask)
            dispersion = np.std(cluster_coords, axis=0).mean()

            centroids.append({
                'cluster': cluster,
                'x': float(centroid_x),
                'y': float(centroid_y),
                'count': int(count),
                'dispersion': float(dispersion)
            })

        # Preparar datos de keywords
        keywords_data = []
        for idx, row in df.iterrows():
            keywords_data.append({
                'id': int(idx),
                'keyword': str(row.get('Keyword', '')),
                'cluster': str(row.get('Cluster', 'Miscellaneous')),
                'search_volume': int(row.get('Search Volume', 0)),
                'cpc': float(row.get('CPC', 0)),
                'kd': float(row.get('Keyword Difficulty', 0)),
                'word_count': int(row.get('Word Count', 0)),
                'intersections': int(row.get('Intersections', 0)),
                'search_intents': str(row.get('Search Intents', '')),
                'avg_rank': float(row.get('Avg Rank', 0)),
                'highest_rank': int(row.get('Highest Rank', 0)),
                'ranked_urls': int(row.get('Ranked URLs', 0)),
                'highest_rank_url': str(row.get('Highest Rank URL', '')),
                'x': float(umap_coords[idx, 0]),
                'y': float(umap_coords[idx, 1])
            })

        print(f"✅ Datos preparados: {len(keywords_data)} keywords, {len(centroids)} centroides")

        return jsonify({
            'keywords': keywords_data,
            'centroids': centroids,
            'clusters': list(df['Cluster'].unique())
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/load_page_centroids', methods=['POST'])
def load_page_centroids():
    """Carga centroides de páginas completas desde Supabase"""
    try:
        data = request.json
        n_neighbors = int(data.get('n_neighbors', 8))
        min_dist = float(data.get('min_dist', 0.4))
        url_filter = data.get('url_filter', '').strip()

        print(f"\n📊 Cargando centroides de páginas desde Supabase...")
        print(f"   UMAP params: n_neighbors={n_neighbors}, min_dist={min_dist}")
        if url_filter:
            print(f"   Filtro URL: '{url_filter}'")

        # Conectar a la base de datos
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Cargar centroides con información de páginas
        # Agregar filtro opcional por URL
        if url_filter:
            query = """
                SELECT
                    p.id,
                    p.url,
                    p.title,
                    p.meta_description,
                    pc.num_chunks,
                    pc.centroid_embedding
                FROM page_centroids pc
                JOIN pages p ON pc.page_id = p.id
                WHERE p.status = 'active' AND p.url ILIKE %s
                ORDER BY pc.created_at DESC
            """
            cursor.execute(query, (f'%{url_filter}%',))
        else:
            query = """
                SELECT
                    p.id,
                    p.url,
                    p.title,
                    p.meta_description,
                    pc.num_chunks,
                    pc.centroid_embedding
                FROM page_centroids pc
                JOIN pages p ON pc.page_id = p.id
                WHERE p.status = 'active'
                ORDER BY pc.created_at DESC
            """
            cursor.execute(query)

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return jsonify({'error': 'No se encontraron centroides en la base de datos'}), 404

        print(f"✅ {len(rows)} centroides cargados desde la base de datos")

        # Extraer embeddings de la base de datos
        # NOTA: Si la BD tiene embeddings 384D antiguos, habrá incompatibilidad con el modelo 768D actual
        # Los vectores vienen como strings de PostgreSQL, necesitamos parsearlos
        embeddings = []
        for row in rows:
            vec = row['centroid_embedding']
            if isinstance(vec, str):
                # Parsear string a lista de floats
                vec = vec.strip('[]').split(',')
                vec = [float(x) for x in vec]
            embeddings.append(vec)

        embeddings = np.array(embeddings)
        print(f"✅ Embeddings extraídos: {embeddings.shape}")

        # Verificar compatibilidad de dimensiones - fallback a re-vectorizar si es necesario
        expected_dim = 768
        actual_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else 0
        if actual_dim != expected_dim:
            print(f"⚠️ ADVERTENCIA: Embeddings de la BD tienen {actual_dim}D, se esperan {expected_dim}D")
            print("   Aplicando fallback: re-vectorizando páginas con modelo español 768D...")

            # Re-vectorizar usando título + descripción de cada página
            texts_to_vectorize = []
            for row in rows:
                title = row['title'] or ''
                desc = row['meta_description'] or ''
                text = f"{title} {desc}".strip() or row['url']
                texts_to_vectorize.append(text)

            embeddings = model_768.encode(texts_to_vectorize, show_progress_bar=True)
            print(f"✅ Re-vectorizado con modelo 768D: {embeddings.shape}")

        # Calcular UMAP
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine',
            random_state=42
        )
        umap_coords = reducer.fit_transform(embeddings)

        # Guardar estado
        umap_state['reducer'] = reducer
        umap_state['embeddings'] = embeddings
        umap_state['vector_type'] = 'page_centroid'

        # Preparar datos de páginas
        pages_data = []
        for idx, row in enumerate(rows):
            pages_data.append({
                'id': str(row['id']),
                'url': row['url'],
                'title': row['title'] or 'Sin título',
                'description': row['meta_description'] or '',
                'num_chunks': row['num_chunks'],
                'x': float(umap_coords[idx, 0]),
                'y': float(umap_coords[idx, 1])
            })

        print(f"✅ Datos preparados: {len(pages_data)} páginas")

        return jsonify({
            'pages': pages_data,
            'total': len(pages_data),
            'type': 'page_centroids'
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/load_chunks', methods=['POST'])
def load_chunks():
    """Carga chunks individuales desde Supabase"""
    try:
        data = request.json
        n_neighbors = int(data.get('n_neighbors', 8))
        min_dist = float(data.get('min_dist', 0.4))
        limit = int(data.get('limit', 5000))  # Límite por defecto para no sobrecargar
        url_filter = data.get('url_filter', '').strip()

        print(f"\n📊 Cargando chunks desde Supabase...")
        print(f"   UMAP params: n_neighbors={n_neighbors}, min_dist={min_dist}")
        print(f"   Límite: {limit} chunks")
        if url_filter:
            print(f"   Filtro URL: '{url_filter}'")

        # Conectar a la base de datos
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Cargar chunks con información de páginas
        # Agregar filtro opcional por URL
        if url_filter:
            query = """
                SELECT
                    e.id,
                    e.page_id,
                    e.chunk_index,
                    e.chunk_text,
                    e.embedding,
                    p.url,
                    p.title
                FROM embeddings e
                JOIN pages p ON e.page_id = p.id
                WHERE p.status = 'active' AND p.url ILIKE %s
                ORDER BY e.created_at DESC
                LIMIT %s
            """
            cursor.execute(query, (f'%{url_filter}%', limit))
        else:
            query = """
                SELECT
                    e.id,
                    e.page_id,
                    e.chunk_index,
                    e.chunk_text,
                    e.embedding,
                    p.url,
                    p.title
                FROM embeddings e
                JOIN pages p ON e.page_id = p.id
                WHERE p.status = 'active'
                ORDER BY e.created_at DESC
                LIMIT %s
            """
            cursor.execute(query, (limit,))

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return jsonify({'error': 'No se encontraron chunks en la base de datos'}), 404

        print(f"✅ {len(rows)} chunks cargados desde la base de datos")

        # Extraer embeddings de la base de datos
        # NOTA: Si la BD tiene embeddings 384D antiguos, habrá incompatibilidad con el modelo 768D actual
        # Los vectores vienen como strings de PostgreSQL, necesitamos parsearlos
        embeddings = []
        for row in rows:
            vec = row['embedding']
            if isinstance(vec, str):
                # Parsear string a lista de floats
                vec = vec.strip('[]').split(',')
                vec = [float(x) for x in vec]
            embeddings.append(vec)

        embeddings = np.array(embeddings)
        print(f"✅ Embeddings extraídos: {embeddings.shape}")

        # Verificar compatibilidad de dimensiones - fallback a re-vectorizar si es necesario
        expected_dim = 768
        actual_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else 0
        if actual_dim != expected_dim:
            print(f"⚠️ ADVERTENCIA: Embeddings de la BD tienen {actual_dim}D, se esperan {expected_dim}D")
            print("   Aplicando fallback: re-vectorizando chunks con modelo español 768D...")

            # Re-vectorizar usando el texto de cada chunk
            texts_to_vectorize = [row['chunk_text'] for row in rows]
            embeddings = model_768.encode(texts_to_vectorize, show_progress_bar=True)
            print(f"✅ Re-vectorizado con modelo 768D: {embeddings.shape}")

        # Calcular UMAP
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine',
            random_state=42
        )
        umap_coords = reducer.fit_transform(embeddings)

        # Guardar estado
        umap_state['reducer'] = reducer
        umap_state['embeddings'] = embeddings
        umap_state['vector_type'] = 'chunk'

        # Preparar datos de chunks
        chunks_data = []
        page_groups = {}  # Agrupar por página para estadísticas

        for idx, row in enumerate(rows):
            page_id = str(row['page_id'])

            chunks_data.append({
                'id': str(row['id']),
                'page_id': page_id,
                'chunk_index': row['chunk_index'],
                'text': row['chunk_text'][:200] + '...' if len(row['chunk_text']) > 200 else row['chunk_text'],
                'url': row['url'],
                'page_title': row['title'] or 'Sin título',
                'x': float(umap_coords[idx, 0]),
                'y': float(umap_coords[idx, 1])
            })

            # Agrupar por página
            if page_id not in page_groups:
                page_groups[page_id] = {
                    'url': row['url'],
                    'title': row['title'],
                    'count': 0
                }
            page_groups[page_id]['count'] += 1

        print(f"✅ Datos preparados: {len(chunks_data)} chunks de {len(page_groups)} páginas")

        return jsonify({
            'chunks': chunks_data,
            'page_groups': page_groups,
            'total': len(chunks_data),
            'type': 'chunks'
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/cluster_titles_h1', methods=['POST'])
def cluster_titles_h1():
    """Clustering avanzado con BERTopic para títulos y H1s"""
    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN

        data = request.json
        url_filter = data.get('url_filter', '').strip()
        source = data.get('source', 'title')  # 'title', 'h1', o 'both'
        min_cluster_size = int(data.get('min_cluster_size', 5))  # Mínimo de páginas por cluster

        print(f"\n🔍 Clustering BERTopic de títulos/H1s...")
        print(f"   Fuente: {source}")
        print(f"   Tamaño mínimo de cluster: {min_cluster_size}")
        if url_filter:
            print(f"   Filtro URL: '{url_filter}'")

        # Conectar a la base de datos
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Cargar títulos y H1s
        if url_filter:
            query = """
                SELECT id, url, title, h1, meta_description
                FROM pages
                WHERE status = 'active' AND url ILIKE %s
                ORDER BY scraped_at DESC
            """
            cursor.execute(query, (f'%{url_filter}%',))
        else:
            query = """
                SELECT id, url, title, h1, meta_description
                FROM pages
                WHERE status = 'active'
                ORDER BY scraped_at DESC
            """
            cursor.execute(query)

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return jsonify({'error': 'No se encontraron páginas en la base de datos'}), 404

        print(f"✅ {len(rows)} páginas cargadas")

        # Preparar textos para clustering
        texts = []
        valid_rows = []

        for row in rows:
            text = ""
            if source == 'title':
                text = row['title'] or ""
            elif source == 'h1':
                text = row['h1'] or ""
            elif source == 'both':
                title = row['title'] or ""
                h1 = row['h1'] or ""
                text = f"{title} {h1}".strip()

            if text:  # Solo incluir si hay texto
                texts.append(text)
                valid_rows.append(row)

        if not texts:
            return jsonify({'error': 'No se encontraron textos válidos para clustering'}), 404

        print(f"✅ {len(texts)} textos válidos para clustering")

        # Generar embeddings con el modelo español 768D
        print(f"🔄 Vectorizando textos con modelo español 768D...")
        embeddings = model_768.encode(texts, show_progress_bar=False)
        print(f"✅ Embeddings generados: {embeddings.shape}")

        # Configurar HDBSCAN para clustering robusto
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        # Inicializar BERTopic con configuración para español
        print(f"🔄 Inicializando BERTopic para español de España...")
        topic_model = BERTopic(
            embedding_model=model_768,
            hdbscan_model=hdbscan_model,
            language='spanish',
            calculate_probabilities=True,
            verbose=True
        )

        # Realizar clustering
        print(f"🔄 Ejecutando BERTopic...")
        topics, probs = topic_model.fit_transform(texts, embeddings)
        print(f"✅ Clustering completado")

        # Obtener información de los topics
        topic_info = topic_model.get_topic_info()
        print(f"✅ Topics detectados: {len(topic_info)}")

        # Agrupar resultados por cluster (primero sin nombres)
        clusters = {}
        for idx, topic_id in enumerate(topics):
            topic_id = int(topic_id)

            if topic_id not in clusters:
                clusters[topic_id] = {
                    'cluster_id': topic_id,
                    'cluster_name': '',  # Se generará después
                    'cluster_keywords': '',
                    'pages': [],
                    'count': 0
                }

            row = valid_rows[idx]
            clusters[topic_id]['pages'].append({
                'id': str(row['id']),
                'url': row['url'],
                'title': row['title'],
                'h1': row['h1'],
                'description': row['meta_description'],
                'probability': float(probs[idx][topic_id]) if topic_id >= 0 else 0.0
            })
            clusters[topic_id]['count'] += 1

        # Generar nombres de clusters usando LLM (con acceso a todas las páginas)
        print(f"🤖 Generando nombres de clusters con {'LLM' if LLM_PROVIDER != 'none' else 'sistema regex'}...")
        for topic_id, cluster_data in clusters.items():
            if topic_id == -1:
                cluster_data['cluster_name'] = "Contenido sin Clasificar"
                cluster_data['cluster_keywords'] = ""
            else:
                # Obtener keywords del topic
                topic_words_list = topic_model.get_topic(topic_id)
                if topic_words_list:
                    keywords = [word for word, _ in topic_words_list[:5]]
                    cluster_data['cluster_keywords'] = ", ".join(keywords)

                    # Generar nombre usando LLM con páginas del cluster
                    cluster_data['cluster_name'] = generate_cluster_name_with_llm(
                        cluster_data['pages'],
                        keywords
                    )
                else:
                    cluster_data['cluster_name'] = f"Cluster {topic_id}"
                    cluster_data['cluster_keywords'] = ""

        # Convertir a lista y ordenar (outliers al final)
        clusters_list = sorted(
            clusters.values(),
            key=lambda x: (x['cluster_id'] == -1, -x['count'])
        )

        print(f"✅ Clusters generados: {len(clusters_list)}")
        for c in clusters_list[:10]:  # Mostrar top 10
            print(f"   Cluster {c['cluster_id']}: {c['count']} páginas - {c['cluster_name'][:60]}...")

        # Generar coordenadas UMAP para visualización
        print(f"🔄 Generando coordenadas UMAP para visualización...")
        umap_reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        umap_coords = umap_reducer.fit_transform(embeddings)
        print(f"✅ UMAP completado: {umap_coords.shape}")

        # Agregar coordenadas a cada página
        pages_map = []
        for idx, topic_id in enumerate(topics):
            topic_id = int(topic_id)
            row = valid_rows[idx]

            # Encontrar nombre del cluster
            cluster_info = clusters.get(topic_id, {})
            cluster_name = cluster_info.get('cluster_name', f'Cluster {topic_id}')

            pages_map.append({
                'id': str(row['id']),
                'url': row['url'],
                'title': row['title'],
                'h1': row['h1'],
                'x': float(umap_coords[idx, 0]),
                'y': float(umap_coords[idx, 1]),
                'cluster_id': topic_id,
                'cluster_name': cluster_name,
                'probability': float(probs[idx][topic_id]) if topic_id >= 0 else 0.0
            })

        return jsonify({
            'clusters': clusters_list,
            'pages_map': pages_map,  # Agregar mapa UMAP
            'total_pages': len(valid_rows),
            'total_clusters': len(clusters_list),
            'source': source,
            'method': 'BERTopic + HDBSCAN'
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/hierarchical_clustering', methods=['POST'])
def hierarchical_clustering():
    """Clustering jerárquico: meta-clusters (zonas generales) + clusters específicos"""
    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_similarity

        data = request.json
        url_filter = data.get('url_filter', '').strip()
        source = data.get('source', 'title')  # 'title', 'h1', o 'both'
        min_cluster_size = int(data.get('min_cluster_size', 5))  # Mínimo para clusters específicos
        num_meta_clusters = int(data.get('num_meta_clusters', 8))  # Número de zonas generales

        print(f"\n🌳 Clustering Jerárquico...")
        print(f"   Fuente: {source}")
        print(f"   Clusters específicos (min size): {min_cluster_size}")
        print(f"   Meta-clusters (zonas): {num_meta_clusters}")
        if url_filter:
            print(f"   Filtro URL: '{url_filter}'")

        # Conectar a la base de datos
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Cargar títulos y H1s
        if url_filter:
            query = """
                SELECT id, url, title, h1, meta_description
                FROM pages
                WHERE status = 'active' AND url ILIKE %s
                ORDER BY scraped_at DESC
            """
            cursor.execute(query, (f'%{url_filter}%',))
        else:
            query = """
                SELECT id, url, title, h1, meta_description
                FROM pages
                WHERE status = 'active'
                ORDER BY scraped_at DESC
            """
            cursor.execute(query)

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return jsonify({'error': 'No se encontraron páginas en la base de datos'}), 404

        print(f"✅ {len(rows)} páginas cargadas")

        # Preparar textos para clustering
        texts = []
        valid_rows = []

        for row in rows:
            text = ""
            if source == 'title':
                text = row['title'] or ""
            elif source == 'h1':
                text = row['h1'] or ""
            elif source == 'both':
                title = row['title'] or ""
                h1 = row['h1'] or ""
                text = f"{title} {h1}".strip()

            if text:
                texts.append(text)
                valid_rows.append(row)

        if not texts:
            return jsonify({'error': 'No se encontraron textos válidos para clustering'}), 404

        print(f"✅ {len(texts)} textos válidos para clustering")

        # PASO 1: Generar embeddings
        print(f"🔄 Vectorizando textos con modelo español 768D...")
        embeddings = model_768.encode(texts, show_progress_bar=False)
        print(f"✅ Embeddings generados: {embeddings.shape}")

        # PASO 2: Clustering específico con BERTopic (nivel detallado)
        print(f"🔄 Nivel 1: Clustering específico con BERTopic...")
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        topic_model = BERTopic(
            embedding_model=model_768,
            hdbscan_model=hdbscan_model,
            language='spanish',
            calculate_probabilities=True,
            verbose=False
        )

        topics, probs = topic_model.fit_transform(texts, embeddings)
        print(f"✅ Clusters específicos generados: {len(set(topics))} clusters")

        # PASO 3: Agrupar embeddings por cluster específico
        cluster_embeddings = []
        cluster_info = []

        unique_topics = sorted(set(topics))
        for topic_id in unique_topics:
            if topic_id == -1:  # Outliers
                continue

            # Obtener embeddings del cluster
            mask = np.array([t == topic_id for t in topics])
            cluster_embs = embeddings[mask]

            # Centroide del cluster = promedio de embeddings
            centroid = np.mean(cluster_embs, axis=0)
            cluster_embeddings.append(centroid)

            # Obtener palabras clave del cluster
            topic_words_list = topic_model.get_topic(topic_id)
            if topic_words_list:
                keywords = [word for word, _ in topic_words_list[:5]]
                # Generar nombre natural (usa regex aquí, LLM en endpoint simple)
                cluster_name_natural = generate_natural_cluster_name_regex(keywords)
                cluster_name_keywords = ", ".join(keywords)
            else:
                cluster_name_natural = f"Cluster {topic_id}"
                cluster_name_keywords = ""

            cluster_info.append({
                'cluster_id': topic_id,
                'cluster_name': cluster_name_natural,
                'cluster_keywords': cluster_name_keywords,  # Guardamos también las keywords
                'count': int(np.sum(mask)),
                'centroid': centroid
            })

        cluster_embeddings = np.array(cluster_embeddings)
        print(f"✅ Centroides de clusters calculados: {cluster_embeddings.shape}")

        # PASO 4: Meta-clustering (agrupar clusters en zonas generales)
        print(f"🔄 Nivel 2: Meta-clustering jerárquico...")

        # Usar clustering jerárquico aglomerativo
        meta_clustering = AgglomerativeClustering(
            n_clusters=min(num_meta_clusters, len(cluster_embeddings)),
            metric='cosine',
            linkage='average'
        )

        meta_labels = meta_clustering.fit_predict(cluster_embeddings)
        print(f"✅ Meta-clusters generados: {len(set(meta_labels))} zonas")

        # PASO 5: Organizar jerarquía
        hierarchy = {}

        for meta_id in set(meta_labels):
            # Obtener clusters que pertenecen a este meta-cluster
            cluster_indices = [i for i, label in enumerate(meta_labels) if label == meta_id]
            meta_cluster_info = [cluster_info[i] for i in cluster_indices]

            # Calcular centroide del meta-cluster
            meta_centroid = np.mean([c['centroid'] for c in meta_cluster_info], axis=0)

            # Generar nombre natural del meta-cluster basado en los clusters que contiene
            cluster_names = [c['cluster_name'] for c in meta_cluster_info]
            total_pages_in_meta = sum(c['count'] for c in meta_cluster_info)

            # Usar la función de generación de nombres naturales
            meta_name_natural = generate_natural_meta_cluster_name(cluster_names, total_pages_in_meta)

            # También guardar las keywords para referencia
            all_keywords = []
            for c in meta_cluster_info:
                if 'cluster_keywords' in c and c['cluster_keywords']:
                    words = c['cluster_keywords'].split(', ')
                    all_keywords.extend(words[:3])

            from collections import Counter
            word_counts = Counter(all_keywords)
            meta_keywords = ", ".join([word for word, _ in word_counts.most_common(5)])

            hierarchy[int(meta_id)] = {
                'meta_cluster_id': int(meta_id),
                'meta_cluster_name': meta_name_natural,
                'meta_cluster_keywords': meta_keywords,  # Keywords para referencia
                'total_pages': sum(c['count'] for c in meta_cluster_info),
                'num_clusters': len(meta_cluster_info),
                'clusters': []
            }

            # Agregar clusters específicos
            for cluster_idx, cluster in zip(cluster_indices, meta_cluster_info):
                # Obtener páginas del cluster
                topic_id = cluster['cluster_id']
                cluster_pages = []

                for idx, (t, row) in enumerate(zip(topics, valid_rows)):
                    if t == topic_id:
                        cluster_pages.append({
                            'id': str(row['id']),
                            'url': row['url'],
                            'title': row['title'],
                            'h1': row['h1'],
                            'description': row['meta_description'],
                            'probability': float(probs[idx][topic_id]) if topic_id >= 0 else 0.0
                        })

                hierarchy[int(meta_id)]['clusters'].append({
                    'cluster_id': cluster['cluster_id'],
                    'cluster_name': cluster['cluster_name'],
                    'count': cluster['count'],
                    'pages': cluster_pages
                })

        # Convertir a lista y ordenar por tamaño
        hierarchy_list = sorted(
            hierarchy.values(),
            key=lambda x: -x['total_pages']
        )

        print(f"✅ Jerarquía construida:")
        for meta in hierarchy_list:
            print(f"   Meta-cluster {meta['meta_cluster_id']}: {meta['total_pages']} páginas en {meta['num_clusters']} clusters")
            print(f"      Nombre: {meta['meta_cluster_name'][:60]}...")

        # NUEVO: Calcular coordenadas UMAP para visualización del mapa completo
        print(f"🔄 Calculando coordenadas UMAP para visualización...")

        # Calcular UMAP de todos los embeddings
        umap_reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        umap_coords = umap_reducer.fit_transform(embeddings)

        # Crear lista de páginas con coordenadas y asignación de meta-cluster
        pages_with_coords = []
        for idx, (topic_id, row) in enumerate(zip(topics, valid_rows)):
            # Encontrar el meta-cluster de esta página
            meta_cluster_id = None
            meta_cluster_name = None
            cluster_id = topic_id
            cluster_name = None

            # Buscar en la jerarquía
            for meta in hierarchy_list:
                for cluster in meta['clusters']:
                    if cluster['cluster_id'] == topic_id:
                        meta_cluster_id = meta['meta_cluster_id']
                        meta_cluster_name = meta['meta_cluster_name']
                        cluster_name = cluster['cluster_name']
                        break
                if meta_cluster_id is not None:
                    break

            # Si no se encuentra (outliers), asignar a meta-cluster especial
            if meta_cluster_id is None:
                meta_cluster_id = -1
                meta_cluster_name = "Sin Clasificar"
                cluster_name = "Outliers"

            pages_with_coords.append({
                'id': str(row['id']),
                'url': row['url'],
                'title': row['title'],
                'h1': row['h1'],
                'description': row['meta_description'],
                'x': float(umap_coords[idx, 0]),
                'y': float(umap_coords[idx, 1]),
                'meta_cluster_id': int(meta_cluster_id),
                'meta_cluster_name': meta_cluster_name,
                'cluster_id': int(cluster_id),
                'cluster_name': cluster_name
            })

        print(f"✅ Coordenadas UMAP calculadas para {len(pages_with_coords)} páginas")

        return jsonify({
            'hierarchy': hierarchy_list,
            'pages_map': pages_with_coords,  # NUEVO: Páginas con coordenadas UMAP
            'total_pages': len(valid_rows),
            'total_clusters': len(cluster_info),
            'total_meta_clusters': len(hierarchy_list),
            'source': source,
            'method': 'BERTopic + Agglomerative Hierarchical'
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap():
    """Genera un mapa de calor de similitud entre clusters y topics personalizados"""
    try:
        from sklearn.metrics.pairwise import cosine_similarity

        data = request.json
        clusters_data = data.get('clusters', [])  # Lista de clusters con sus nombres
        custom_topics = data.get('topics', [])    # Lista de topics personalizados
        source = data.get('source', 'title')      # Fuente de datos (title, h1, both)

        if not clusters_data or not custom_topics:
            return jsonify({'error': 'Se requieren clusters y topics'}), 400

        print(f"\n🔥 Generando mapa de calor...")
        print(f"   Clusters: {len(clusters_data)}")
        print(f"   Topics personalizados: {len(custom_topics)}")
        print(f"   Fuente: {source}")

        # Extraer nombres de clusters
        cluster_names = [c['cluster_name'] for c in clusters_data]

        # Vectorizar nombres de clusters usando modelo español 768D
        print(f"🔄 Vectorizando nombres de clusters...")
        cluster_embeddings = model_768.encode(cluster_names, show_progress_bar=False)

        # Vectorizar topics personalizados
        print(f"🔄 Vectorizando topics personalizados...")
        topic_embeddings = model_768.encode(custom_topics, show_progress_bar=False)

        # Calcular similitud coseno entre clusters y topics
        print(f"🔄 Calculando similitudes de clusters...")
        similarities = cosine_similarity(cluster_embeddings, topic_embeddings)

        # Calcular similitudes individuales por URL
        print(f"🔄 Calculando similitudes por URL...")
        url_similarities = []

        for cluster in clusters_data:
            cluster_urls = []
            for page in cluster['pages']:
                # Construir texto según la fuente
                if source == 'title':
                    text = page.get('title', '') or ''
                elif source == 'h1':
                    text = page.get('h1', '') or ''
                elif source == 'both':
                    title = page.get('title', '') or ''
                    h1 = page.get('h1', '') or ''
                    text = f"{title} {h1}".strip()
                else:
                    text = page.get('title', '') or ''

                if text:
                    # Vectorizar texto de la URL
                    url_embedding = model_768.encode([text], show_progress_bar=False)
                    # Calcular similitud con cada topic
                    url_sim = cosine_similarity(url_embedding, topic_embeddings)[0]

                    cluster_urls.append({
                        'url': page['url'],
                        'title': page.get('title', ''),
                        'h1': page.get('h1', ''),
                        'similarities': url_sim.tolist()
                    })

            url_similarities.append(cluster_urls)

        # Convertir a lista para JSON
        similarities_list = similarities.tolist()

        print(f"✅ Mapa de calor generado: {len(cluster_names)}x{len(custom_topics)}")
        print(f"✅ Similitudes por URL calculadas para {sum(len(c) for c in url_similarities)} URLs")

        return jsonify({
            'heatmap': similarities_list,
            'cluster_names': cluster_names,
            'cluster_ids': [c['cluster_id'] for c in clusters_data],
            'cluster_counts': [c['count'] for c in clusters_data],
            'topics': custom_topics,
            'url_similarities': url_similarities,
            'max_similarity': float(similarities.max()),
            'min_similarity': float(similarities.min())
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Endpoint para verificar que el servidor está activo"""
    cache_stats = get_cache_stats()
    return jsonify({
        'status': 'ok',
        'model': 'hiiamsid/sentence_similarity_spanish_es',
        'optimizations': {
            'batch_size': BATCH_SIZE,
            'umap_epochs': UMAP_N_EPOCHS,
            'fp16': USE_FP16,
            'device': device
        },
        'cache': {
            'type': 'sqlite',
            'entries': cache_stats.get('entries', 0),
            'total_keywords': cache_stats.get('total_keywords', 0),
            'size_mb': cache_stats.get('size_mb', 0)
        }
    })


# ============================================
# ENDPOINTS DE PRE-CÁLCULO EN BACKGROUND
# ============================================

@app.route('/precalc/upload', methods=['POST'])
def precalc_upload():
    """
    Sube un archivo Excel e inicia pre-cálculo de embeddings en background.
    Devuelve un hash inmediatamente para consultar el estado después.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se envió archivo'}), 400

        file = request.files['file']
        keyword_column = request.form.get('keyword_column', 'Keyword')

        # Leer archivo
        file_content = file.read()
        file_hash = get_file_hash(file_content)

        # Si ya está en cache SQLite, devolver inmediatamente
        cached_texts, cached_embeddings = get_from_cache(file_hash)
        if cached_texts is not None:
            return jsonify({
                'hash': file_hash,
                'status': 'cached',
                'message': 'Embeddings recuperados de cache SQLite',
                'count': len(cached_texts)
            })

        # Si ya está calculándose, devolver estado
        if file_hash in precalc_tasks and precalc_tasks[file_hash]['status'] == 'running':
            return jsonify({
                'hash': file_hash,
                'status': 'running',
                'progress': precalc_tasks[file_hash]['progress']
            })

        # Leer Excel
        df = pd.read_excel(io.BytesIO(file_content))

        if keyword_column not in df.columns:
            return jsonify({'error': f'Columna "{keyword_column}" no encontrada'}), 400

        texts = df[keyword_column].fillna('').astype(str).tolist()

        # Iniciar pre-cálculo en background
        executor.submit(precalculate_embeddings_task, file_hash, texts, keyword_column)

        return jsonify({
            'hash': file_hash,
            'status': 'started',
            'message': f'Pre-calculando embeddings para {len(texts)} keywords',
            'count': len(texts)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/precalc/status/<file_hash>', methods=['GET'])
def precalc_status(file_hash):
    """Consulta el estado del pre-cálculo de embeddings"""
    # Si está en cache SQLite
    cached_texts, cached_embeddings = get_from_cache(file_hash)
    if cached_texts is not None:
        return jsonify({
            'hash': file_hash,
            'status': 'done',
            'progress': 100,
            'count': len(cached_texts),
            'source': 'sqlite_cache'
        })

    # Si está en proceso
    if file_hash in precalc_tasks:
        task = precalc_tasks[file_hash]
        return jsonify({
            'hash': file_hash,
            'status': task['status'],
            'progress': task['progress'],
            'error': task.get('error')
        })

    return jsonify({
        'hash': file_hash,
        'status': 'not_found',
        'message': 'No hay pre-cálculo para este archivo'
    }), 404


@app.route('/precalc/cache_info', methods=['GET'])
def precalc_cache_info():
    """Información del cache SQLite de embeddings"""
    return jsonify(get_cache_stats())


# ============================================
# Rutas para servir archivos HTML estáticos
# ============================================
@app.route('/')
def serve_index():
    """Servir la página principal"""
    return send_from_directory('.', 'index.html')

@app.route('/index.html')
def serve_index_html():
    """Servir index.html explícitamente"""
    return send_from_directory('.', 'index.html')

@app.route('/topic-map-keywords-multilevel.html')
def serve_multilevel():
    """Servir la herramienta de clustering multinivel"""
    return send_from_directory('.', 'topic-map-keywords-multilevel.html')

@app.route('/content-gap-analysis-v2.html')
def serve_content_gap():
    """Servir la herramienta de content gap analysis"""
    return send_from_directory('.', 'content-gap-analysis-v2.html')

@app.route('/cluster-reinforcement.html')
def serve_cluster_reinforcement():
    """Servir la herramienta de refuerzo de clusters"""
    return send_from_directory('.', 'cluster-reinforcement.html')


@app.route('/cluster-reinforcement-v2.html')
def serve_cluster_reinforcement_v2():
    """Servir la herramienta de refuerzo de clusters V2"""
    return send_from_directory('.', 'cluster-reinforcement-v2.html')


@app.route('/cluster_keywords_multilevel', methods=['POST'])
def cluster_keywords_multilevel():
    """
    Clustering multinivel de keywords desde Excel (Google Search Console o SEMrush).

    Parámetros esperados (form-data):
    - file: Archivo Excel con keywords y métricas
    - keyword_column: Nombre de la columna con keywords (default: 'Keyword')
    - level1_min_cluster_size: Tamaño mínimo para clusters nivel 1 (default: 5)
    - level1_min_samples: Min samples para nivel 1 (default: 2)
    - level2_num_clusters: Número de meta-clusters nivel 2 (default: 8)
    - n_neighbors: UMAP n_neighbors (default: 25)
    - min_dist: UMAP min_dist (default: 0.1)
    - metric_columns: JSON array con columnas de métricas a incluir

    Retorna:
    - keywords_data: Lista de keywords con coordenadas y asignaciones de cluster
    - level1_clusters: Clusters específicos con stats
    - level2_clusters: Meta-clusters (agrupaciones de nivel 1)
    - hierarchy: Estructura jerárquica completa
    """

    def convert_to_native_types(obj):
        """Convertir tipos numpy a tipos Python nativos para JSON"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_types(item) for item in obj]
        else:
            return obj

    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.cluster import KMeans
        import json

        # Leer parámetros
        file = request.files['file']
        keyword_column = request.form.get('keyword_column', 'Keyword').strip()
        level1_min_cluster_size = int(request.form.get('level1_min_cluster_size', 25))  # Aumentado de 5 a 25
        level1_min_samples = int(request.form.get('level1_min_samples', 10))  # Aumentado de 2 a 10
        level2_num_clusters = int(request.form.get('level2_num_clusters', 20))  # Aumentado de 8 a 20
        n_neighbors = int(request.form.get('n_neighbors', 25))
        min_dist = float(request.form.get('min_dist', 0.1))
        metric_columns_json = request.form.get('metric_columns', '[]')
        metric_columns = json.loads(metric_columns_json) if metric_columns_json else []
        use_llm_names = request.form.get('use_llm_names', 'false').lower() == 'true'

        print(f"\n📊 Clustering multinivel de keywords")
        print(f"   Archivo: {file.filename}")
        print(f"   Columna keyword: '{keyword_column}'")
        print(f"   Nivel 1 - min_cluster_size: {level1_min_cluster_size}, min_samples: {level1_min_samples}")
        print(f"   Nivel 2 - num_meta_clusters: {level2_num_clusters}")
        print(f"   UMAP - n_neighbors: {n_neighbors}, min_dist: {min_dist}")
        print(f"   Nombres LLM: {'Sí' if use_llm_names else 'No (regex rápido)'}")
        print(f"   Métricas adicionales: {metric_columns}")

        # Leer Excel
        df = pd.read_excel(file)
        print(f"✅ Excel cargado: {len(df)} filas, {len(df.columns)} columnas")
        print(f"   Columnas disponibles: {list(df.columns)}")

        # Detectar columna de keyword (flexible para GSC y SEMrush)
        kw_col = None
        for col in df.columns:
            if col.lower() in [keyword_column.lower(), 'keyword', 'query', 'keywords', 'palabra clave']:
                kw_col = col
                break

        if kw_col is None:
            return jsonify({'error': f'No se encontró la columna de keywords. Columnas disponibles: {list(df.columns)}'}), 400

        print(f"✅ Columna de keywords detectada: '{kw_col}'")

        # Filtrar keywords válidas
        df = df[df[kw_col].notna()].copy()
        df[kw_col] = df[kw_col].astype(str).str.strip()
        df = df[df[kw_col] != '']

        if len(df) == 0:
            return jsonify({'error': 'No hay keywords válidas en el archivo'}), 400

        print(f"✅ {len(df)} keywords válidas")

        # Vectorizar keywords (con batch size optimizado)
        print(f"🔄 Vectorizando {len(df)} keywords...")
        keywords_list = df[kw_col].tolist()
        embeddings = model_768.encode(
            keywords_list,
            show_progress_bar=False,
            batch_size=128,  # Mayor batch size = más rápido
            convert_to_numpy=True
        )
        print(f"✅ Embeddings generados: {embeddings.shape}")

        # ===== NIVEL 1: Clustering específico con BERTopic =====
        print(f"\n🔄 NIVEL 1: Clustering específico con BERTopic...")
        hdbscan_model = HDBSCAN(
            min_cluster_size=level1_min_cluster_size,
            min_samples=level1_min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        topic_model = BERTopic(
            embedding_model=model_768,
            hdbscan_model=hdbscan_model,
            language='spanish',
            calculate_probabilities=False,  # Deshabilitado para acelerar (no necesario)
            verbose=False
        )

        topics, probs = topic_model.fit_transform(keywords_list, embeddings)
        print(f"✅ Nivel 1 completado: {len(set(topics))} clusters detectados")

        # ===== NIVEL 2: Meta-clustering jerárquico =====
        print(f"\n🔄 NIVEL 2: Meta-clustering...")

        # Calcular centroides de cada cluster nivel 1
        cluster_ids = [t for t in set(topics) if t != -1]
        centroids = []
        centroid_cluster_ids = []

        for cluster_id in cluster_ids:
            indices = [i for i, t in enumerate(topics) if t == cluster_id]
            if len(indices) > 0:
                centroid = np.mean(embeddings[indices], axis=0)
                centroids.append(centroid)
                centroid_cluster_ids.append(cluster_id)

        centroids = np.array(centroids)
        print(f"✅ {len(centroids)} centroides calculados")

        # K-Means para meta-clusters
        if len(centroids) > level2_num_clusters:
            kmeans = KMeans(n_clusters=level2_num_clusters, random_state=42, n_init=10)
            meta_labels = kmeans.fit_predict(centroids)
        else:
            meta_labels = list(range(len(centroids)))

        # Mapear clusters nivel 1 → meta-clusters nivel 2
        cluster_to_meta = dict(zip(centroid_cluster_ids, meta_labels))
        print(f"✅ {len(set(meta_labels))} meta-clusters creados")

        # ===== CALCULAR UMAP PARA VISUALIZACIÓN =====
        print(f"\n🔄 Calculando UMAP 2D...")
        umap_reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine',
            random_state=42
        )
        coords_2d = umap_reducer.fit_transform(embeddings)
        print(f"✅ UMAP completado: {coords_2d.shape}")

        # ===== CONSTRUIR JERARQUÍA =====
        print(f"\n🔄 Construyendo jerarquía...")

        # Agrupar keywords por cluster nivel 1
        level1_clusters = {}
        for idx, (topic_id, keyword) in enumerate(zip(topics, keywords_list)):
            topic_id = int(topic_id)

            if topic_id not in level1_clusters:
                level1_clusters[topic_id] = {
                    'cluster_id': topic_id,
                    'meta_cluster_id': cluster_to_meta.get(topic_id, -1),
                    'keywords': [],
                    'count': 0,
                    'cluster_name': ''
                }

            # Agregar keyword con métricas
            kw_data = {'keyword': keyword}
            for metric_col in metric_columns:
                if metric_col in df.columns:
                    kw_data[metric_col] = df.iloc[idx][metric_col]

            level1_clusters[topic_id]['keywords'].append(kw_data)
            level1_clusters[topic_id]['count'] += 1

        # Generar nombres para clusters nivel 1
        if use_llm_names:
            print(f"\n🤖 Generando nombres de clusters nivel 1 con LLM...")
        else:
            print(f"\n⚡ Generando nombres de clusters nivel 1 (regex rápido)...")

        for cluster_id, cluster_data in level1_clusters.items():
            if cluster_id == -1:
                cluster_data['cluster_name'] = "Sin Clasificar"
                continue

            # Tomar sample de keywords
            sample_kws = [kw['keyword'] for kw in cluster_data['keywords'][:10]]

            if use_llm_names and (LLM_PROVIDER == 'ollama' or LLM_PROVIDER == 'openai'):
                cluster_name = generate_cluster_name_with_llm_keywords(sample_kws)
            else:
                cluster_name = generate_natural_cluster_name_regex(sample_kws)

            cluster_data['cluster_name'] = cluster_name

        # Agrupar por meta-clusters nivel 2
        level2_clusters = {}
        for cluster_id, cluster_data in level1_clusters.items():
            meta_id = cluster_data['meta_cluster_id']

            if meta_id not in level2_clusters:
                level2_clusters[meta_id] = {
                    'meta_cluster_id': meta_id,
                    'meta_cluster_name': '',
                    'level1_clusters': [],
                    'total_keywords': 0
                }

            level2_clusters[meta_id]['level1_clusters'].append(cluster_data)
            level2_clusters[meta_id]['total_keywords'] += cluster_data['count']

        # Generar nombres para meta-clusters nivel 2
        print(f"\n🤖 Generando nombres de meta-clusters nivel 2...")
        for meta_id, meta_data in level2_clusters.items():
            if meta_id == -1:
                meta_data['meta_cluster_name'] = "Sin Clasificar"
                continue

            # Tomar sample de todos los clusters dentro
            all_cluster_names = [c['cluster_name'] for c in meta_data['level1_clusters'] if c['cluster_name']]
            if all_cluster_names:
                meta_data['meta_cluster_name'] = ', '.join(all_cluster_names[:3])
            else:
                meta_data['meta_cluster_name'] = f"Zona {meta_id + 1}"

        # Preparar datos de keywords con coordenadas
        keywords_data = []
        for idx, (keyword, topic_id, coord) in enumerate(zip(keywords_list, topics, coords_2d)):
            kw_data = {
                'keyword': keyword,
                'x': float(coord[0]),
                'y': float(coord[1]),
                'cluster_id': int(topic_id),
                'cluster_name': level1_clusters[int(topic_id)]['cluster_name'],
                'meta_cluster_id': int(level1_clusters[int(topic_id)]['meta_cluster_id']),
                'meta_cluster_name': level2_clusters[level1_clusters[int(topic_id)]['meta_cluster_id']]['meta_cluster_name']
            }

            # Agregar métricas (convertir numpy types a Python nativos)
            for metric_col in metric_columns:
                if metric_col in df.columns:
                    value = df.iloc[idx][metric_col]
                    if pd.notna(value):
                        # Convertir numpy types a Python nativos para JSON
                        if isinstance(value, (np.integer, np.int32, np.int64)):
                            kw_data[metric_col] = int(value)
                        elif isinstance(value, (np.floating, np.float32, np.float64)):
                            kw_data[metric_col] = float(value)
                        else:
                            kw_data[metric_col] = value
                    else:
                        kw_data[metric_col] = None

            keywords_data.append(kw_data)

        print(f"✅ Jerarquía construida:")
        print(f"   - {len(keywords_data)} keywords")
        print(f"   - {len(level1_clusters)} clusters nivel 1")
        print(f"   - {len(level2_clusters)} meta-clusters nivel 2")

        # Convertir todos los valores numpy a tipos Python nativos para JSON
        response_data = {
            'keywords_data': convert_to_native_types(keywords_data),
            'level1_clusters': convert_to_native_types(list(level1_clusters.values())),
            'level2_clusters': convert_to_native_types(list(level2_clusters.values())),
            'total_keywords': int(len(keywords_data)),
            'available_metrics': metric_columns
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error en clustering multinivel: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def generate_cluster_name_with_llm_keywords(keywords_list, urls_list=None):
    """
    Genera nombre de cluster (Level 1) a partir de lista de keywords.
    El nombre debe representar la TEMÁTICA CENTRAL del grupo de URLs.
    """
    if not keywords_list or LLM_PROVIDER == 'none':
        return generate_natural_cluster_name_regex(keywords_list)

    # Usar más keywords para mejor contexto (hasta 10)
    keywords_str = ', '.join(keywords_list[:10])

    # Incluir contexto de URLs si está disponible
    url_context = ""
    if urls_list and len(urls_list) > 0:
        # Extraer paths de las URLs para dar contexto adicional
        url_paths = []
        for url in urls_list[:3]:
            try:
                from urllib.parse import urlparse
                path = urlparse(url).path.strip('/')
                if path:
                    # Limpiar el path
                    path_clean = path.replace('-', ' ').replace('_', ' ').replace('/', ' ')
                    url_paths.append(path_clean)
            except:
                pass
        if url_paths:
            url_context = f"\nContexto de URLs: {', '.join(url_paths)}"

    prompt = f"""Eres un experto en SEO y categorización de contenido. Analiza estas palabras clave que pertenecen a un grupo de páginas web relacionadas.

Palabras clave del grupo: {keywords_str}{url_context}

Tu tarea: Identificar la TEMÁTICA CENTRAL que une a todas estas keywords y generar un nombre descriptivo.

Reglas estrictas:
- Máximo 4 palabras
- Debe ser una categoría temática clara (ej: "Formación Profesional Sanitaria", "Cursos de Marketing Digital")
- NO uses palabras genéricas como "información", "contenido", "artículos"
- NO uses dos puntos (:) ni guiones (-)
- El nombre debe ser específico y representar el TEMA, no las keywords individuales
- Responde SOLO con el nombre, sin explicaciones ni comillas

Temática central:"""

    try:
        if LLM_PROVIDER == 'ollama':
            import ollama
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': LLM_TEMPERATURE}
            )
            cluster_name = response['message']['content'].strip()

        elif LLM_PROVIDER == 'openai':
            response = llm_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=LLM_TEMPERATURE,
                max_tokens=30
            )
            cluster_name = response.choices[0].message.content.strip()

        # Limpiar resultado
        cluster_name = cluster_name.replace('"', '').replace("'", "").replace(':', '').replace('-', ' ').strip()

        if len(cluster_name.split()) > 5:
            cluster_name = ' '.join(cluster_name.split()[:5])

        return cluster_name if cluster_name else generate_natural_cluster_name_regex(keywords_list)

    except Exception as e:
        print(f"⚠️  Error generando nombre con LLM: {e}")
        return generate_natural_cluster_name_regex(keywords_list)


def generate_meta_cluster_name_with_llm(cluster_names, all_keywords):
    """
    Genera nombre de meta-cluster (Level 2) a partir de los clusters que contiene.
    El nombre debe ser un TÓPICO AMPLIO que englobe múltiples temáticas centrales.
    """
    if not cluster_names or LLM_PROVIDER == 'none':
        # Fallback: usar los primeros 2 nombres de clusters
        return ', '.join(cluster_names[:2]) if cluster_names else "Zona Sin Nombre"

    # Preparar lista de temáticas (nombres de clusters)
    themes_str = ', '.join(cluster_names[:6])

    # Muestra de keywords para contexto adicional
    keywords_sample = ', '.join(all_keywords[:15]) if all_keywords else ""

    prompt = f"""Eres un experto en taxonomía y categorización de contenido web. Tienes varios grupos temáticos que necesitas agrupar bajo un TÓPICO MÁS AMPLIO.

Temáticas a agrupar:
{themes_str}

Palabras clave de ejemplo: {keywords_sample}

Tu tarea: Crear un nombre de CATEGORÍA SUPERIOR que englobe todas estas temáticas.

Ejemplos de buena categorización:
- Si las temáticas son "Grado en Enfermería", "Técnico en Farmacia", "Auxiliar de Enfermería" → "Formación Sanitaria"
- Si las temáticas son "SEO On-Page", "Link Building", "Keyword Research" → "Marketing Digital SEO"
- Si las temáticas son "Python Básico", "JavaScript Avanzado", "Desarrollo Web" → "Programación y Desarrollo"

Reglas estrictas:
- Máximo 3 palabras
- Debe ser una CATEGORÍA AMPLIA, no específica
- Debe englobar TODAS las temáticas listadas
- NO uses dos puntos (:) ni guiones (-)
- NO repitas exactamente ninguna de las temáticas
- Responde SOLO con el nombre de la categoría, sin explicaciones

Categoría superior:"""

    try:
        if LLM_PROVIDER == 'ollama':
            import ollama
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': LLM_TEMPERATURE}
            )
            meta_name = response['message']['content'].strip()

        elif LLM_PROVIDER == 'openai':
            response = llm_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=LLM_TEMPERATURE,
                max_tokens=20
            )
            meta_name = response.choices[0].message.content.strip()

        # Limpiar resultado
        meta_name = meta_name.replace('"', '').replace("'", "").replace(':', '').replace('-', ' ').strip()

        if len(meta_name.split()) > 4:
            meta_name = ' '.join(meta_name.split()[:4])

        return meta_name if meta_name else ', '.join(cluster_names[:2])

    except Exception as e:
        print(f"⚠️  Error generando nombre de meta-cluster con LLM: {e}")
        return ', '.join(cluster_names[:2]) if cluster_names else "Zona Sin Nombre"


@app.route('/cluster_by_url', methods=['POST'])
def cluster_by_url():
    """
    Clustering multinivel basado en URLs:
    1. Agrupa keywords por URL
    2. Vectoriza URL + keywords combinados
    3. Clustering de URLs (nivel 1)
    4. Meta-clustering (nivel 2)
    5. UMAP para visualización
    """
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.cluster import KMeans
        import json
        import re

        # Leer parámetros
        file = request.files['file']
        keyword_column = request.form.get('keyword_column', 'Keyword').strip()
        url_column = request.form.get('url_column', 'URL').strip()
        min_cluster_size = int(request.form.get('min_cluster_size', 5))
        min_samples = int(request.form.get('min_samples', 3))
        num_meta_clusters = int(request.form.get('num_meta_clusters', 15))
        n_neighbors = int(request.form.get('n_neighbors', 15))
        min_dist = float(request.form.get('min_dist', 0.1))
        metric_columns_json = request.form.get('metric_columns', '[]')
        metric_columns = json.loads(metric_columns_json) if metric_columns_json else []
        use_llm_names = request.form.get('use_llm_names', 'false').lower() == 'true'

        print(f"\n{'='*60}")
        print(f"🔗 CLUSTERING MULTINIVEL BASADO EN URLs")
        print(f"{'='*60}")
        print(f"   Archivo: {file.filename}")
        print(f"   Columna keyword: '{keyword_column}'")
        print(f"   Columna URL: '{url_column}'")
        print(f"   min_cluster_size: {min_cluster_size}, min_samples: {min_samples}")
        print(f"   num_meta_clusters: {num_meta_clusters}")
        print(f"   Nombres LLM: {'Sí' if use_llm_names else 'No'}")

        # Leer Excel
        df = pd.read_excel(file)
        print(f"\n✅ Excel cargado: {len(df)} filas")

        # Detectar columnas
        kw_col = None
        url_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in [keyword_column.lower(), 'keyword', 'query', 'keywords']:
                kw_col = col
            if col_lower in [url_column.lower(), 'url', 'landing page', 'page', 'address']:
                url_col = col

        if kw_col is None:
            return jsonify({'error': f'No se encontró columna de keywords. Disponibles: {list(df.columns)}'}), 400
        if url_col is None:
            return jsonify({'error': f'No se encontró columna de URL. Disponibles: {list(df.columns)}'}), 400

        print(f"✅ Columnas detectadas: keyword='{kw_col}', url='{url_col}'")

        # Limpiar datos
        df = df[df[kw_col].notna() & df[url_col].notna()].copy()
        df[kw_col] = df[kw_col].astype(str).str.strip()
        df[url_col] = df[url_col].astype(str).str.strip()
        df = df[(df[kw_col] != '') & (df[url_col] != '')]

        print(f"✅ {len(df)} filas válidas con keyword y URL")

        # ===== PASO 1: Agrupar keywords por URL =====
        print(f"\n🔄 PASO 1: Agrupando keywords por URL...")

        url_groups = {}
        for idx, row in df.iterrows():
            url = row[url_col]
            keyword = row[kw_col]

            if url not in url_groups:
                url_groups[url] = {
                    'url': url,
                    'keywords': [],
                    'metrics': {},
                    'row_indices': []
                }

            url_groups[url]['keywords'].append(keyword)
            url_groups[url]['row_indices'].append(idx)

            # Agregar métricas (sumar para métricas numéricas)
            for metric_col in metric_columns:
                if metric_col in df.columns:
                    val = row[metric_col]
                    if pd.notna(val):
                        if metric_col not in url_groups[url]['metrics']:
                            url_groups[url]['metrics'][metric_col] = 0
                        if isinstance(val, (int, float, np.number)):
                            url_groups[url]['metrics'][metric_col] += float(val)

        urls_list = list(url_groups.keys())
        print(f"✅ {len(urls_list)} URLs únicas encontradas")

        # ===== PASO 2: Vectorizar URLs (SOLO keywords agrupadas) =====
        print(f"\n🔄 PASO 2: Vectorizando grupos de keywords por URL...")

        def extract_url_text(url):
            """Extrae texto semántico de una URL, filtrando palabras estructurales"""
            # Remover protocolo y www
            text = re.sub(r'https?://(www\.)?', '', url)
            # Remover dominio principal (mantener path)
            parts = text.split('/', 1)
            if len(parts) > 1:
                text = parts[1]
            else:
                return ''  # Solo dominio, sin path útil

            # Reemplazar separadores por espacios
            text = re.sub(r'[-_/.]', ' ', text)

            # FILTRAR palabras estructurales que no aportan semánticamente
            structural_words = {
                'blog', 'blogs', 'landing', 'landings', 'page', 'pages',
                'category', 'categories', 'categoria', 'categorias',
                'tag', 'tags', 'etiqueta', 'etiquetas',
                'post', 'posts', 'articulo', 'articulos', 'article', 'articles',
                'producto', 'productos', 'product', 'products',
                'servicio', 'servicios', 'service', 'services',
                'index', 'home', 'inicio', 'main',
                'html', 'php', 'aspx', 'htm', 'jsp',
                'es', 'en', 'com', 'net', 'org', 'www',
                'the', 'de', 'la', 'el', 'los', 'las', 'un', 'una',
                'and', 'or', 'y', 'o', 'con', 'para', 'por', 'en', 'a',
                '2020', '2021', '2022', '2023', '2024', '2025'
            }

            words = text.lower().split()
            filtered_words = [w for w in words if w not in structural_words and len(w) > 2]

            return ' '.join(filtered_words)

        # Estrategia: dar MUCHO más peso a las keywords que al texto de URL
        # Las keywords definen la temática, la URL solo complementa
        url_texts = []
        for url in urls_list:
            url_data = url_groups[url]

            # Keywords repetidas para dar más peso (80% keywords, 20% URL)
            keywords_list = url_data['keywords'][:30]  # Más keywords para mejor representación
            keywords_text = ' '.join(keywords_list)

            # Texto de URL filtrado (solo semántico, sin estructura)
            url_text = extract_url_text(url)

            # Combinar: SOLO keywords si URL no aporta, o keywords + URL filtrado
            if url_text and len(url_text) > 5:
                # URL aporta algo semántico: 80% keywords, 20% URL
                combined_text = f"{keywords_text} {keywords_text} {keywords_text} {keywords_text} {url_text}"
            else:
                # URL no aporta: 100% keywords
                combined_text = keywords_text

            url_texts.append(combined_text)

        # Vectorizar los textos combinados
        url_embeddings = model_768.encode(
            url_texts,
            show_progress_bar=False,
            batch_size=64,
            convert_to_numpy=True
        )
        print(f"✅ Embeddings de URLs generados: {url_embeddings.shape}")

        # ===== PASO 3: Clustering de URLs (Nivel 1) =====
        print(f"\n🔄 PASO 3: Clustering de URLs...")

        # Ajustar parámetros para MAXIMIZAR clasificación (reducir "Sin Clasificar")
        # Usar min_cluster_size más pequeño para capturar más URLs
        effective_min_cluster = max(2, min(min_cluster_size, len(urls_list) // 20))
        effective_min_samples = max(1, min(min_samples, effective_min_cluster - 1))

        print(f"   📊 Parámetros ajustados: min_cluster={effective_min_cluster}, min_samples={effective_min_samples}")

        hdbscan_model = HDBSCAN(
            min_cluster_size=effective_min_cluster,
            min_samples=effective_min_samples,
            metric='euclidean',
            cluster_selection_method='leaf',  # 'leaf' crea más clusters pequeños, menos outliers
            prediction_data=True
        )

        topic_model = BERTopic(
            embedding_model=model_768,
            hdbscan_model=hdbscan_model,
            language='spanish',
            calculate_probabilities=False,
            verbose=False
        )

        url_topics, _ = topic_model.fit_transform(url_texts, url_embeddings)

        n_clusters = len(set(url_topics)) - (1 if -1 in url_topics else 0)
        n_outliers = sum(1 for t in url_topics if t == -1)
        print(f"✅ {n_clusters} clusters de URLs detectados, {n_outliers} outliers")

        # ===== REASIGNAR OUTLIERS al cluster más cercano =====
        if n_outliers > 0 and n_clusters > 0:
            print(f"\n🔄 Reasignando {n_outliers} URLs sin clasificar al cluster más cercano...")
            from sklearn.metrics.pairwise import cosine_similarity

            # Calcular centroides de cada cluster
            cluster_centroids = {}
            for cluster_id in set(url_topics):
                if cluster_id == -1:
                    continue
                indices = [i for i, t in enumerate(url_topics) if t == cluster_id]
                cluster_centroids[cluster_id] = np.mean(url_embeddings[indices], axis=0)

            # Reasignar outliers
            url_topics = list(url_topics)
            reassigned = 0
            for i, topic in enumerate(url_topics):
                if topic == -1:
                    # Encontrar cluster más cercano
                    best_cluster = -1
                    best_similarity = -1
                    for cluster_id, centroid in cluster_centroids.items():
                        sim = cosine_similarity([url_embeddings[i]], [centroid])[0][0]
                        if sim > best_similarity:
                            best_similarity = sim
                            best_cluster = cluster_id

                    # Asignar si similitud es razonable (> 0.3)
                    if best_cluster != -1 and best_similarity > 0.3:
                        url_topics[i] = best_cluster
                        reassigned += 1

            print(f"✅ {reassigned} URLs reasignadas (similitud > 0.3)")

        # ===== PASO 4: Meta-clustering (Nivel 2) =====
        print(f"\n🔄 PASO 4: Meta-clustering...")

        # Calcular centroides de cada cluster de URLs
        cluster_ids = [t for t in set(url_topics) if t != -1]
        centroids = []
        centroid_cluster_ids = []

        for cluster_id in cluster_ids:
            indices = [i for i, t in enumerate(url_topics) if t == cluster_id]
            if len(indices) > 0:
                centroid = np.mean(url_embeddings[indices], axis=0)
                centroids.append(centroid)
                centroid_cluster_ids.append(cluster_id)

        if len(centroids) > 0:
            centroids = np.array(centroids)

            # K-Means para meta-clusters
            effective_meta_clusters = min(num_meta_clusters, len(centroids))
            if len(centroids) > effective_meta_clusters:
                kmeans = KMeans(n_clusters=effective_meta_clusters, random_state=42, n_init=10)
                meta_labels = kmeans.fit_predict(centroids)
            else:
                meta_labels = list(range(len(centroids)))

            cluster_to_meta = dict(zip(centroid_cluster_ids, meta_labels))
        else:
            cluster_to_meta = {}

        print(f"✅ {len(set(meta_labels)) if len(centroids) > 0 else 0} meta-clusters creados")

        # ===== PASO 5: Vectorizar keywords individuales para UMAP =====
        print(f"\n🔄 PASO 5: Vectorizando keywords para visualización...")

        all_keywords = df[kw_col].tolist()
        keyword_embeddings = model_768.encode(
            all_keywords,
            show_progress_bar=False,
            batch_size=128,
            convert_to_numpy=True
        )
        print(f"✅ Embeddings de keywords: {keyword_embeddings.shape}")

        # ===== PASO 6: UMAP para visualización =====
        print(f"\n🔄 PASO 6: Calculando UMAP 2D...")

        umap_reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine',
            random_state=42
        )
        coords_2d = umap_reducer.fit_transform(keyword_embeddings)
        print(f"✅ UMAP completado: {coords_2d.shape}")

        # ===== PASO 7: Construir jerarquía =====
        print(f"\n🔄 PASO 7: Construyendo estructura de datos...")

        # Mapear URL -> cluster -> meta-cluster
        url_to_cluster = {}
        for i, url in enumerate(urls_list):
            url_to_cluster[url] = url_topics[i]

        # Construir clusters nivel 1 (basados en URLs)
        level1_clusters = {}
        for i, url in enumerate(urls_list):
            cluster_id = int(url_topics[i])
            meta_id = cluster_to_meta.get(cluster_id, -1) if cluster_id != -1 else -1

            if cluster_id not in level1_clusters:
                level1_clusters[cluster_id] = {
                    'cluster_id': cluster_id,
                    'meta_cluster_id': int(meta_id),
                    'urls': [],
                    'keywords': [],
                    'count': 0,
                    'cluster_name': ''
                }

            url_data = url_groups[url]
            level1_clusters[cluster_id]['urls'].append(url)
            level1_clusters[cluster_id]['keywords'].extend(url_data['keywords'])
            level1_clusters[cluster_id]['count'] += len(url_data['keywords'])

        # Generar nombres para clusters (Level 1 - Temáticas Centrales)
        print(f"\n{'🤖' if use_llm_names else '⚡'} Generando nombres de clusters (temáticas centrales)...")
        for cluster_id, cluster_data in level1_clusters.items():
            if cluster_id == -1:
                cluster_data['cluster_name'] = "Sin Clasificar"
                continue

            sample_kws = cluster_data['keywords'][:15]
            sample_urls = cluster_data['urls'][:5]
            if use_llm_names and (LLM_PROVIDER == 'ollama' or LLM_PROVIDER == 'openai'):
                cluster_data['cluster_name'] = generate_cluster_name_with_llm_keywords(sample_kws, sample_urls)
            else:
                cluster_data['cluster_name'] = generate_natural_cluster_name_regex(sample_kws)

        # Construir meta-clusters nivel 2
        level2_clusters = {}
        for cluster_id, cluster_data in level1_clusters.items():
            meta_id = cluster_data['meta_cluster_id']

            if meta_id not in level2_clusters:
                level2_clusters[meta_id] = {
                    'meta_cluster_id': meta_id,
                    'meta_cluster_name': '',
                    'level1_clusters': [],
                    'total_keywords': 0,
                    'total_urls': 0
                }

            level2_clusters[meta_id]['level1_clusters'].append(cluster_data)
            level2_clusters[meta_id]['total_keywords'] += cluster_data['count']
            level2_clusters[meta_id]['total_urls'] += len(cluster_data['urls'])

        # Nombres para meta-clusters (Level 2 - Tópicos Amplios)
        print(f"\n{'🤖' if use_llm_names else '⚡'} Generando nombres de meta-clusters (tópicos amplios)...")
        for meta_id, meta_data in level2_clusters.items():
            if meta_id == -1:
                meta_data['meta_cluster_name'] = "Sin Clasificar"
                continue

            cluster_names = [c['cluster_name'] for c in meta_data['level1_clusters'] if c['cluster_name'] and c['cluster_name'] != 'Sin Clasificar']
            # Recopilar todas las keywords de los clusters del meta-cluster
            all_meta_keywords = []
            for c in meta_data['level1_clusters']:
                all_meta_keywords.extend(c.get('keywords', [])[:10])

            if use_llm_names and (LLM_PROVIDER == 'ollama' or LLM_PROVIDER == 'openai') and cluster_names:
                meta_data['meta_cluster_name'] = generate_meta_cluster_name_with_llm(cluster_names, all_meta_keywords)
            else:
                meta_data['meta_cluster_name'] = ', '.join(cluster_names[:2]) if cluster_names else f"Zona {meta_id + 1}"

        # Preparar datos de keywords con coordenadas
        keywords_data = []
        for idx, row in df.iterrows():
            keyword = row[kw_col]
            url = row[url_col]
            coord_idx = df.index.get_loc(idx)

            cluster_id = url_to_cluster.get(url, -1)
            cluster_info = level1_clusters.get(cluster_id, {
                'cluster_name': 'Sin Clasificar',
                'meta_cluster_id': -1
            })
            meta_id = cluster_info.get('meta_cluster_id', -1)
            meta_info = level2_clusters.get(meta_id, {'meta_cluster_name': 'Sin Clasificar'})

            kw_data = {
                'keyword': keyword,
                'url': url,
                'x': float(coords_2d[coord_idx][0]),
                'y': float(coords_2d[coord_idx][1]),
                'cluster_id': int(cluster_id),
                'cluster_name': cluster_info.get('cluster_name', 'Sin Clasificar'),
                'meta_cluster_id': int(meta_id),
                'meta_cluster_name': meta_info.get('meta_cluster_name', 'Sin Clasificar')
            }

            # Agregar métricas
            for metric_col in metric_columns:
                if metric_col in df.columns:
                    val = row[metric_col]
                    if pd.notna(val):
                        if isinstance(val, (np.integer, np.int32, np.int64)):
                            kw_data[metric_col] = int(val)
                        elif isinstance(val, (np.floating, np.float32, np.float64)):
                            kw_data[metric_col] = float(val)
                        else:
                            kw_data[metric_col] = val

            keywords_data.append(kw_data)

        # ===== Preparar datos de URLs con coordenadas (centroide de sus keywords) =====
        print(f"\n🔄 Calculando centroides de URLs...")
        urls_data = []
        for url in urls_list:
            url_data = url_groups[url]
            cluster_id = url_to_cluster.get(url, -1)
            cluster_info = level1_clusters.get(cluster_id, {
                'cluster_name': 'Sin Clasificar',
                'meta_cluster_id': -1
            })
            meta_id = cluster_info.get('meta_cluster_id', -1)
            meta_info = level2_clusters.get(meta_id, {'meta_cluster_name': 'Sin Clasificar'})

            # Obtener keywords de esta URL y calcular centroide
            url_keywords = [kw for kw in keywords_data if kw['url'] == url]
            if url_keywords:
                url_x = sum(k['x'] for k in url_keywords) / len(url_keywords)
                url_y = sum(k['y'] for k in url_keywords) / len(url_keywords)

                # Agregar métricas de la URL (suma de sus keywords)
                url_metrics = {}
                for metric_col in metric_columns:
                    metric_values = []
                    for k in url_keywords:
                        val = k.get(metric_col)
                        if val is not None:
                            try:
                                metric_values.append(float(val))
                            except (ValueError, TypeError):
                                pass  # Ignorar valores no numéricos
                    if metric_values:
                        url_metrics[metric_col] = sum(metric_values)

                # Extraer path corto de la URL
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    url_path = parsed.path.strip('/')
                    if len(url_path) > 60:
                        url_path = '...' + url_path[-57:]
                except:
                    url_path = url

                urls_data.append({
                    'url': url,
                    'url_short': url_path or parsed.netloc,
                    'x': float(url_x),
                    'y': float(url_y),
                    'cluster_id': int(cluster_id),
                    'cluster_name': cluster_info.get('cluster_name', 'Sin Clasificar'),
                    'meta_cluster_id': int(meta_id),
                    'meta_cluster_name': meta_info.get('meta_cluster_name', 'Sin Clasificar'),
                    'keywords_count': len(url_keywords),
                    'keywords_sample': [k['keyword'] for k in url_keywords[:5]],
                    **url_metrics
                })

        print(f"✅ {len(urls_data)} URLs con centroides calculados")

        print(f"\n✅ Estructura construida:")
        print(f"   - {len(keywords_data)} keywords")
        print(f"   - {len(urls_data)} URLs con centroides")
        print(f"   - {len(level1_clusters)} clusters de URLs")
        print(f"   - {len(level2_clusters)} meta-clusters")

        response_data = {
            'keywords_data': convert_to_native(keywords_data),
            'urls_data': convert_to_native(urls_data),
            'level1_clusters': convert_to_native(list(level1_clusters.values())),
            'level2_clusters': convert_to_native(list(level2_clusters.values())),
            'total_keywords': len(keywords_data),
            'total_urls': len(urls_data),
            'clustering_mode': 'url_based'
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error en clustering por URL: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_content_gaps', methods=['POST'])
def analyze_content_gaps():
    """
    Análisis de Content Gaps entre sitio propio y competidor(es).

    Proceso:
    1. Carga Excel propio y del competidor
    2. Vectoriza todas las keywords en espacio común
    3. Detecta gaps: keywords que tiene el competidor pero no el sitio propio
    4. Detecta oportunidades: keywords propias sin competencia
    5. Detecta competencia directa: keywords compartidas
    6. Calcula similitud semántica para encontrar gaps "cercanos"
    7. Proyecta todo en UMAP 2D para visualización
    """
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    try:
        from sklearn.metrics.pairwise import cosine_similarity
        import json

        print(f"\n{'='*60}")
        print(f"🔍 ANÁLISIS DE CONTENT GAPS")
        print(f"{'='*60}")

        # Leer archivos
        own_file = request.files.get('own_file')
        competitor_file = request.files.get('competitor_file')

        if not own_file or not competitor_file:
            return jsonify({'error': 'Se requieren ambos archivos: own_file y competitor_file'}), 400

        # Parámetros
        keyword_column = request.form.get('keyword_column', 'Keyword').strip()
        similarity_threshold = float(request.form.get('similarity_threshold', 0.7))
        n_neighbors = int(request.form.get('n_neighbors', 15))
        min_dist = float(request.form.get('min_dist', 0.1))
        metric_columns_json = request.form.get('metric_columns', '[]')
        metric_columns = json.loads(metric_columns_json) if metric_columns_json else []

        print(f"   Archivo propio: {own_file.filename}")
        print(f"   Archivo competidor: {competitor_file.filename}")
        print(f"   Columna keyword: '{keyword_column}'")
        print(f"   Umbral similitud: {similarity_threshold}")

        # Cargar Excel
        df_own = pd.read_excel(own_file)
        df_competitor = pd.read_excel(competitor_file)

        print(f"\n✅ Archivos cargados:")
        print(f"   - Propio: {len(df_own)} filas")
        print(f"   - Competidor: {len(df_competitor)} filas")

        # Detectar columna de keywords
        def find_keyword_column(df, hint):
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in [hint.lower(), 'keyword', 'query', 'keywords', 'palabra clave']:
                    return col
            return df.columns[0]

        kw_col_own = find_keyword_column(df_own, keyword_column)
        kw_col_comp = find_keyword_column(df_competitor, keyword_column)

        print(f"   - Columna KW propio: '{kw_col_own}'")
        print(f"   - Columna KW competidor: '{kw_col_comp}'")

        # Limpiar datos
        df_own = df_own[df_own[kw_col_own].notna()].copy()
        df_own[kw_col_own] = df_own[kw_col_own].astype(str).str.strip().str.lower()
        df_own = df_own[df_own[kw_col_own] != '']

        df_competitor = df_competitor[df_competitor[kw_col_comp].notna()].copy()
        df_competitor[kw_col_comp] = df_competitor[kw_col_comp].astype(str).str.strip().str.lower()
        df_competitor = df_competitor[df_competitor[kw_col_comp] != '']

        # Obtener keywords únicos
        keywords_own = df_own[kw_col_own].unique().tolist()
        keywords_comp = df_competitor[kw_col_comp].unique().tolist()

        set_own = set(keywords_own)
        set_comp = set(keywords_comp)

        print(f"\n📊 Keywords únicos:")
        print(f"   - Propios: {len(keywords_own)}")
        print(f"   - Competidor: {len(keywords_comp)}")

        # ===== PASO 1: Clasificación exacta =====
        print(f"\n🔄 PASO 1: Clasificación por coincidencia exacta...")

        exact_shared = set_own & set_comp
        exact_only_own = set_own - set_comp
        exact_only_comp = set_comp - set_own

        print(f"   - Compartidos (exactos): {len(exact_shared)}")
        print(f"   - Solo propios: {len(exact_only_own)}")
        print(f"   - Solo competidor (GAPS): {len(exact_only_comp)}")

        # ===== PASO 2: Vectorización =====
        print(f"\n🔄 PASO 2: Vectorizando keywords...")

        all_keywords = list(set(keywords_own + keywords_comp))
        embeddings = model_768.encode(
            all_keywords,
            show_progress_bar=False,
            batch_size=128,
            convert_to_numpy=True
        )

        # Crear diccionario keyword -> embedding
        kw_to_embedding = {kw: embeddings[i] for i, kw in enumerate(all_keywords)}

        print(f"✅ {len(all_keywords)} keywords vectorizados")

        # ===== PASO 3: Similitud semántica para gaps cercanos =====
        print(f"\n🔄 PASO 3: Buscando gaps semánticos (similitud > {similarity_threshold})...")

        # Para cada keyword del competidor que no tenemos, buscar el más similar nuestro
        semantic_gaps = []  # Keywords del competidor cercanos a los nuestros (oportunidad)

        if len(exact_only_comp) > 0 and len(keywords_own) > 0:
            embeddings_own = np.array([kw_to_embedding[kw] for kw in keywords_own])

            for comp_kw in exact_only_comp:
                comp_emb = kw_to_embedding[comp_kw].reshape(1, -1)
                similarities = cosine_similarity(comp_emb, embeddings_own)[0]
                max_sim_idx = np.argmax(similarities)
                max_sim = similarities[max_sim_idx]

                if max_sim >= similarity_threshold:
                    semantic_gaps.append({
                        'competitor_keyword': comp_kw,
                        'similar_own_keyword': keywords_own[max_sim_idx],
                        'similarity': float(max_sim)
                    })

        print(f"✅ {len(semantic_gaps)} gaps semánticos encontrados")

        # ===== PASO 4: UMAP para visualización =====
        print(f"\n🔄 PASO 4: Calculando UMAP 2D...")

        umap_reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, len(all_keywords) - 1),
            min_dist=min_dist,
            metric='cosine',
            random_state=42
        )
        coords_2d = umap_reducer.fit_transform(embeddings)

        # Crear diccionario keyword -> coordenadas
        kw_to_coords = {kw: coords_2d[i] for i, kw in enumerate(all_keywords)}

        print(f"✅ UMAP completado")

        # ===== PASO 5: Preparar datos de salida =====
        print(f"\n🔄 PASO 5: Preparando datos de visualización...")

        # Función para obtener métricas de un keyword en un dataframe
        def get_metrics(df, kw_col, keyword, metric_cols):
            rows = df[df[kw_col] == keyword]
            if len(rows) == 0:
                return {}
            row = rows.iloc[0]
            metrics = {}
            # Siempre incluir URL si existe
            url_columns = ['URL', 'url', 'Url', 'Ranked URL', 'Landing Page', 'Page', 'Target URL']
            for url_col in url_columns:
                if url_col in df.columns:
                    val = row[url_col]
                    if pd.notna(val) and isinstance(val, str):
                        metrics['URL'] = val
                        break
            for col in metric_cols:
                if col in df.columns:
                    val = row[col]
                    if pd.notna(val):
                        if isinstance(val, (np.integer, np.int32, np.int64)):
                            metrics[col] = int(val)
                        elif isinstance(val, (np.floating, np.float32, np.float64)):
                            metrics[col] = float(val)
                        else:
                            metrics[col] = val
            return metrics

        # Datos para visualización
        own_data = []
        for kw in keywords_own:
            coords = kw_to_coords[kw]
            metrics = get_metrics(df_own, kw_col_own, kw, metric_columns)
            own_data.append({
                'keyword': kw,
                'x': float(coords[0]),
                'y': float(coords[1]),
                'source': 'own',
                'status': 'shared' if kw in exact_shared else 'exclusive',
                **metrics
            })

        competitor_data = []
        for kw in keywords_comp:
            coords = kw_to_coords[kw]
            metrics = get_metrics(df_competitor, kw_col_comp, kw, metric_columns)
            competitor_data.append({
                'keyword': kw,
                'x': float(coords[0]),
                'y': float(coords[1]),
                'source': 'competitor',
                'status': 'shared' if kw in exact_shared else 'gap',
                **metrics
            })

        # Gaps ordenados por métricas (si hay)
        gaps_list = []
        for kw in exact_only_comp:
            coords = kw_to_coords[kw]
            metrics = get_metrics(df_competitor, kw_col_comp, kw, metric_columns)

            # Buscar keyword propio más similar
            similar_own = None
            similarity = 0
            for sg in semantic_gaps:
                if sg['competitor_keyword'] == kw:
                    similar_own = sg['similar_own_keyword']
                    similarity = sg['similarity']
                    break

            gaps_list.append({
                'keyword': kw,
                'x': float(coords[0]),
                'y': float(coords[1]),
                'similar_own_keyword': similar_own,
                'similarity': similarity,
                **metrics
            })

        # Ordenar gaps por volumen/tráfico si está disponible
        sort_col = None
        for col in ['Volume', 'Volumen', 'Traffic', 'Clicks', 'Impressions']:
            if col in metric_columns:
                sort_col = col
                break

        if sort_col:
            gaps_list.sort(key=lambda x: x.get(sort_col, 0), reverse=True)

        # Oportunidades (keywords propias sin competencia)
        opportunities_list = []
        for kw in exact_only_own:
            coords = kw_to_coords[kw]
            metrics = get_metrics(df_own, kw_col_own, kw, metric_columns)
            opportunities_list.append({
                'keyword': kw,
                'x': float(coords[0]),
                'y': float(coords[1]),
                **metrics
            })

        if sort_col:
            opportunities_list.sort(key=lambda x: x.get(sort_col, 0), reverse=True)

        # Keywords compartidas (competencia directa)
        shared_list = []
        for kw in exact_shared:
            coords = kw_to_coords[kw]
            metrics_own = get_metrics(df_own, kw_col_own, kw, metric_columns)
            metrics_comp = get_metrics(df_competitor, kw_col_comp, kw, metric_columns)
            shared_list.append({
                'keyword': kw,
                'x': float(coords[0]),
                'y': float(coords[1]),
                'own_metrics': metrics_own,
                'competitor_metrics': metrics_comp
            })

        print(f"\n✅ Análisis completado:")
        print(f"   - Keywords propios: {len(own_data)}")
        print(f"   - Keywords competidor: {len(competitor_data)}")
        print(f"   - GAPS (oportunidades de contenido): {len(gaps_list)}")
        print(f"   - Fortalezas (sin competencia): {len(opportunities_list)}")
        print(f"   - Competencia directa: {len(shared_list)}")

        # Guardar estado para proyección de consultas
        global gap_analysis_state
        all_keywords_data = own_data + [d for d in competitor_data if d['status'] != 'shared']
        gap_analysis_state['embeddings'] = embeddings
        gap_analysis_state['coords_2d'] = coords_2d
        gap_analysis_state['keywords'] = all_keywords_data
        print(f"   ✅ Estado guardado para proyección de consultas")

        response_data = {
            'own_data': convert_to_native(own_data),
            'competitor_data': convert_to_native(competitor_data),
            'gaps': convert_to_native(gaps_list),
            'opportunities': convert_to_native(opportunities_list),
            'shared': convert_to_native(shared_list),
            'semantic_gaps': convert_to_native(semantic_gaps),
            'summary': {
                'total_own': len(keywords_own),
                'total_competitor': len(keywords_comp),
                'total_gaps': len(gaps_list),
                'total_opportunities': len(opportunities_list),
                'total_shared': len(shared_list),
                'semantic_gaps_found': len(semantic_gaps)
            }
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error en análisis de gaps: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# Estado global para proyectar consultas en UMAP existente
gap_analysis_state = {
    'embeddings': None,      # Embeddings de todos los keywords
    'coords_2d': None,       # Coordenadas UMAP 2D
    'keywords': None,        # Lista de keywords
    'umap_reducer': None     # Reducer UMAP fitted
}


@app.route('/project_queries', methods=['POST'])
def project_queries():
    """
    Proyecta consultas personalizadas en el espacio UMAP existente.
    Encuentra keywords cercanos a cada consulta para identificar zonas temáticas.
    """
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    try:
        from sklearn.metrics.pairwise import cosine_similarity
        import json

        data = request.get_json()
        queries = data.get('queries', [])
        top_k = data.get('top_k', 10)  # Número de keywords cercanos a devolver

        if not queries:
            return jsonify({'error': 'Se requiere al menos una consulta'}), 400

        if gap_analysis_state['embeddings'] is None:
            return jsonify({'error': 'Primero debes ejecutar un análisis de gaps para crear el espacio UMAP'}), 400

        print(f"\n{'='*60}")
        print(f"🎯 PROYECTANDO CONSULTAS TEMÁTICAS")
        print(f"{'='*60}")
        print(f"   Consultas: {queries}")

        # Vectorizar las consultas
        query_embeddings = model_768.encode(
            queries,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        print(f"✅ Consultas vectorizadas: {query_embeddings.shape}")

        # Proyectar en UMAP existente usando transform
        # Si no hay reducer guardado, aproximar posición por similitud
        results = []

        stored_embeddings = gap_analysis_state['embeddings']
        stored_coords = gap_analysis_state['coords_2d']
        stored_keywords = gap_analysis_state['keywords']

        for i, query in enumerate(queries):
            query_emb = query_embeddings[i].reshape(1, -1)

            # Calcular similitud con todos los keywords
            similarities = cosine_similarity(query_emb, stored_embeddings)[0]

            # Encontrar los top_k más similares
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            nearby_keywords = []
            for idx in top_indices:
                nearby_keywords.append({
                    'keyword': stored_keywords[idx]['keyword'],
                    'similarity': float(similarities[idx]),
                    'x': float(stored_coords[idx][0]),
                    'y': float(stored_coords[idx][1]),
                    'source': stored_keywords[idx].get('source', 'unknown'),
                    'status': stored_keywords[idx].get('status', 'unknown')
                })

            # Aproximar posición de la consulta como promedio ponderado de los cercanos
            weights = similarities[top_indices]
            weights = weights / weights.sum()  # Normalizar

            approx_x = np.average([stored_coords[idx][0] for idx in top_indices], weights=weights)
            approx_y = np.average([stored_coords[idx][1] for idx in top_indices], weights=weights)

            # Calcular cobertura: qué porcentaje de keywords cercanos son propios vs competidor
            own_count = sum(1 for kw in nearby_keywords if kw['source'] == 'own')
            comp_count = sum(1 for kw in nearby_keywords if kw['source'] == 'competitor')

            coverage = {
                'own': own_count,
                'competitor': comp_count,
                'gap_opportunity': comp_count > own_count  # True si hay más del competidor
            }

            results.append({
                'query': query,
                'x': float(approx_x),
                'y': float(approx_y),
                'nearby_keywords': nearby_keywords,
                'avg_similarity': float(np.mean(similarities[top_indices])),
                'max_similarity': float(np.max(similarities)),
                'coverage': coverage
            })

            print(f"   ✅ '{query}' → ({approx_x:.2f}, {approx_y:.2f})")
            print(f"      Cobertura: {own_count} propios, {comp_count} competidor")

        print(f"\n✅ {len(results)} consultas proyectadas")

        return jsonify({
            'projected_queries': convert_to_native(results)
        })

    except Exception as e:
        print(f"❌ Error proyectando consultas: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# CONTENT GAP ANALYSIS V2 - Multi-competidor + Clustering + Scoring + Intención
# ============================================================================

# Estado global para v2
gap_analysis_state_v2 = {
    'embeddings': None,
    'coords_2d': None,
    'keywords': None,
    'all_keywords_list': None,  # Lista de strings de keywords (para filtro semántico)
    'gaps_embeddings': None,
    'gaps_data': None
}

def detect_search_intent(keyword):
    """
    Detecta la intención de búsqueda de un keyword usando patrones.
    Returns: (intent_type, confidence)
    """
    keyword_lower = keyword.lower()

    # Patrones por intención
    transactional_patterns = [
        'comprar', 'precio', 'barato', 'oferta', 'descuento', 'tienda',
        'venta', 'coste', 'contratar', 'presupuesto', 'tarifa', 'cuanto cuesta',
        'mejor precio', 'donde comprar', 'online', 'envio', 'gratis'
    ]

    informational_patterns = [
        'qué es', 'que es', 'cómo', 'como', 'por qué', 'porque', 'para qué',
        'guía', 'guia', 'tutorial', 'aprende', 'significado', 'definición',
        'diferencia entre', 'tipos de', 'ejemplos', 'ventajas', 'desventajas',
        'beneficios', 'características', 'historia', 'cuando', 'cuándo'
    ]

    navigational_patterns = [
        'login', 'acceso', 'entrar', 'iniciar sesión', 'registro',
        'contacto', 'teléfono', 'dirección', 'horario', 'ubicación'
    ]

    commercial_patterns = [
        'mejor', 'mejores', 'top', 'comparativa', 'vs', 'versus',
        'review', 'reseña', 'opiniones', 'análisis', 'comparar',
        'alternativas', 'recomendaciones', 'ranking'
    ]

    # Detectar intención
    for pattern in transactional_patterns:
        if pattern in keyword_lower:
            return ('transactional', 0.8)

    for pattern in commercial_patterns:
        if pattern in keyword_lower:
            return ('commercial', 0.8)

    for pattern in informational_patterns:
        if pattern in keyword_lower:
            return ('informational', 0.8)

    for pattern in navigational_patterns:
        if pattern in keyword_lower:
            return ('navigational', 0.8)

    # Por defecto, informacional con baja confianza
    return ('informational', 0.3)


def calculate_opportunity_score(keyword_data, competitors_count=1, has_similar_own=False, similarity=0):
    """
    Calcula un score de oportunidad 0-100 para un gap.

    Factores:
    - Volumen de búsqueda (más = mejor)
    - Dificultad inversa (menor KD = mejor)
    - Número de competidores que lo tienen (más = más importante)
    - Proximidad semántica a contenido propio (más cercano = más fácil)
    """
    score = 50  # Base

    # Factor volumen (0-30 puntos)
    volume = keyword_data.get('Volume', keyword_data.get('Volumen', 0)) or 0
    if volume > 10000:
        score += 30
    elif volume > 5000:
        score += 25
    elif volume > 1000:
        score += 20
    elif volume > 500:
        score += 15
    elif volume > 100:
        score += 10
    elif volume > 0:
        score += 5

    # Factor dificultad inversa (0-20 puntos)
    kd = keyword_data.get('KD', keyword_data.get('Keyword Difficulty', 100)) or 100
    if kd < 20:
        score += 20
    elif kd < 40:
        score += 15
    elif kd < 60:
        score += 10
    elif kd < 80:
        score += 5

    # Factor competidores (0-15 puntos)
    if competitors_count >= 3:
        score += 15
    elif competitors_count >= 2:
        score += 10
    elif competitors_count >= 1:
        score += 5

    # Factor proximidad semántica (0-15 puntos)
    if has_similar_own and similarity > 0:
        score += int(similarity * 15)

    # Normalizar a 0-100
    return min(100, max(0, score))


@app.route('/analyze_content_gaps_v2', methods=['POST'])
def analyze_content_gaps_v2():
    """
    Análisis de Content Gaps V2 - Multi-competidor con clustering, scoring e intención.

    Mejoras sobre v1:
    - Soporte para múltiples competidores
    - Clustering automático de gaps por tema
    - Score de oportunidad para priorización
    - Detección de intención de búsqueda
    - Tracking de qué competidores tienen cada keyword
    """
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    try:
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.cluster import KMeans
        import hdbscan
        import json

        print(f"\n{'='*70}")
        print(f"🔍 ANÁLISIS DE CONTENT GAPS V2 - MULTI-COMPETIDOR")
        print(f"{'='*70}")

        # === Leer archivos ===
        own_file = request.files.get('own_file')
        if not own_file:
            return jsonify({'error': 'Se requiere archivo propio (own_file)'}), 400

        # Obtener múltiples archivos de competidores
        competitor_files = request.files.getlist('competitor_files')

        # Fallback: también aceptar 'competitor_file' (singular) para compatibilidad
        if not competitor_files:
            single_comp = request.files.get('competitor_file')
            if single_comp:
                competitor_files = [single_comp]

        if not competitor_files:
            return jsonify({'error': 'Se requiere al menos un archivo de competidor'}), 400

        # === Parámetros ===
        keyword_column = request.form.get('keyword_column', 'Keyword').strip()
        similarity_threshold = float(request.form.get('similarity_threshold', 0.7))
        n_neighbors = int(request.form.get('n_neighbors', 15))
        min_dist = float(request.form.get('min_dist', 0.1))
        num_clusters = int(request.form.get('num_clusters', 10))
        min_cluster_size = int(request.form.get('min_cluster_size', 5))
        use_llm_names = request.form.get('use_llm_names', 'false').lower() == 'true'
        metric_columns_json = request.form.get('metric_columns', '[]')
        metric_columns = json.loads(metric_columns_json) if metric_columns_json else []
        extract_url_content = request.form.get('extract_url_content', 'false').lower() == 'true'

        print(f"   Archivo propio: {own_file.filename}")
        print(f"   Competidores: {len(competitor_files)} archivos")
        for i, cf in enumerate(competitor_files):
            print(f"      {i+1}. {cf.filename}")
        print(f"   Columna keyword: '{keyword_column}'")
        print(f"   Umbral similitud: {similarity_threshold}")
        print(f"   Clusters objetivo: {num_clusters}")

        # === Función para encontrar columna de keywords ===
        def find_keyword_column(df, hint):
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in [hint.lower(), 'keyword', 'query', 'keywords', 'palabra clave', 'top queries']:
                    return col
            return df.columns[0]

        # === Cargar archivo propio ===
        df_own = pd.read_excel(own_file)
        kw_col_own = find_keyword_column(df_own, keyword_column)
        df_own = df_own[df_own[kw_col_own].notna()].copy()
        df_own[kw_col_own] = df_own[kw_col_own].astype(str).str.strip().str.lower()
        df_own = df_own[df_own[kw_col_own] != '']
        keywords_own = set(df_own[kw_col_own].unique().tolist())

        print(f"\n✅ Archivo propio cargado: {len(keywords_own)} keywords únicos")

        # === Cargar archivos de competidores ===
        competitors_data = []  # Lista de {name, df, kw_col, keywords}
        all_competitor_keywords = {}  # keyword -> [lista de competidores que lo tienen]

        for i, comp_file in enumerate(competitor_files):
            comp_name = comp_file.filename.replace('.xlsx', '').replace('.xls', '').replace('.csv', '')
            df_comp = pd.read_excel(comp_file)
            kw_col_comp = find_keyword_column(df_comp, keyword_column)

            df_comp = df_comp[df_comp[kw_col_comp].notna()].copy()
            df_comp[kw_col_comp] = df_comp[kw_col_comp].astype(str).str.strip().str.lower()
            df_comp = df_comp[df_comp[kw_col_comp] != '']

            keywords_comp = set(df_comp[kw_col_comp].unique().tolist())

            competitors_data.append({
                'name': comp_name,
                'df': df_comp,
                'kw_col': kw_col_comp,
                'keywords': keywords_comp
            })

            # Registrar qué competidores tienen cada keyword
            for kw in keywords_comp:
                if kw not in all_competitor_keywords:
                    all_competitor_keywords[kw] = []
                all_competitor_keywords[kw].append(comp_name)

            print(f"   ✅ Competidor '{comp_name}': {len(keywords_comp)} keywords")

        # === Calcular conjuntos ===
        all_comp_keywords = set(all_competitor_keywords.keys())

        exact_shared = keywords_own & all_comp_keywords
        exact_only_own = keywords_own - all_comp_keywords
        exact_only_comp = all_comp_keywords - keywords_own  # GAPS

        print(f"\n📊 Análisis de conjuntos:")
        print(f"   - Keywords propios: {len(keywords_own)}")
        print(f"   - Keywords competidores (total único): {len(all_comp_keywords)}")
        print(f"   - Compartidos: {len(exact_shared)}")
        print(f"   - Solo propios (fortalezas): {len(exact_only_own)}")
        print(f"   - Solo competidores (GAPS): {len(exact_only_comp)}")

        # === Vectorizar todos los keywords (con contenido de URLs si está activado) ===
        all_keywords = list(keywords_own | all_comp_keywords)

        # Diccionario para guardar contenido de URLs
        url_content_cache = {}

        if extract_url_content:
            print(f"\n🌐 Extrayendo contenido de URLs para embeddings enriquecidos...")
            import concurrent.futures
            import requests
            from bs4 import BeautifulSoup

            def extract_page_content_fast(url, timeout=8):
                """Extrae título, descripción y H1 de una URL."""
                try:
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'}
                    response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')

                    title = soup.find('title')
                    title = title.get_text(strip=True) if title else ''

                    meta_desc = soup.find('meta', attrs={'name': 'description'})
                    description = meta_desc.get('content', '') if meta_desc else ''

                    h1 = soup.find('h1')
                    h1 = h1.get_text(strip=True) if h1 else ''

                    h2s = [h2.get_text(strip=True) for h2 in soup.find_all('h2')[:3]]

                    return {'title': title, 'description': description, 'h1': h1, 'h2s': h2s, 'success': True}
                except:
                    return {'title': '', 'description': '', 'h1': '', 'h2s': [], 'success': False}

            # Recopilar URLs únicas de todos los dataframes
            url_columns = ['URL', 'url', 'Url', 'Ranked URL', 'Landing Page', 'Page', 'Target URL']
            all_urls = set()

            def get_url_from_df(df):
                for col in url_columns:
                    if col in df.columns:
                        urls = df[col].dropna().astype(str)
                        return set(urls[urls.str.startswith('http')].tolist())
                return set()

            all_urls.update(get_url_from_df(df_own))
            for comp in competitors_data:
                all_urls.update(get_url_from_df(comp['df']))

            all_urls = list(all_urls)[:200]  # Limitar a 200 URLs
            print(f"   URLs únicas encontradas: {len(all_urls)}")

            # Extraer contenido en paralelo
            with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                future_to_url = {executor.submit(extract_page_content_fast, url): url for url in all_urls}
                for i, future in enumerate(concurrent.futures.as_completed(future_to_url)):
                    url = future_to_url[future]
                    try:
                        result = future.result()
                        if result['success']:
                            url_content_cache[url] = result
                    except:
                        pass
                    if (i + 1) % 50 == 0:
                        print(f"   Procesadas {i + 1}/{len(all_urls)} URLs...")

            print(f"✅ Contenido extraído de {len(url_content_cache)} URLs")

            # Crear mapeo keyword -> URL para textos enriquecidos
            kw_to_url = {}
            for col in url_columns:
                if col in df_own.columns:
                    for _, row in df_own.iterrows():
                        kw = row[kw_col_own]
                        url = row[col]
                        if pd.notna(url) and isinstance(url, str) and url.startswith('http'):
                            kw_to_url[kw] = url
                    break

            for comp in competitors_data:
                for col in url_columns:
                    if col in comp['df'].columns:
                        for _, row in comp['df'].iterrows():
                            kw = row[comp['kw_col']]
                            url = row[col]
                            if pd.notna(url) and isinstance(url, str) and url.startswith('http'):
                                if kw not in kw_to_url:  # No sobrescribir si ya existe
                                    kw_to_url[kw] = url
                        break

        # Crear textos para embeddings
        print(f"\n🔄 Vectorizando {len(all_keywords)} keywords...")

        texts_for_embedding = []
        for kw in all_keywords:
            if extract_url_content and kw in kw_to_url:
                url = kw_to_url[kw]
                if url in url_content_cache:
                    content = url_content_cache[url]
                    # Texto enriquecido: keyword + título + descripción + H1
                    enriched = f"{kw} {content['title']} {content['description']} {content['h1']} {' '.join(content['h2s'])}"
                    texts_for_embedding.append(enriched.strip())
                else:
                    texts_for_embedding.append(kw)
            else:
                texts_for_embedding.append(kw)

        embeddings = model_768.encode(
            texts_for_embedding,
            show_progress_bar=True,
            batch_size=128,
            convert_to_numpy=True
        )

        kw_to_embedding = {kw: embeddings[i] for i, kw in enumerate(all_keywords)}
        kw_to_idx = {kw: i for i, kw in enumerate(all_keywords)}

        # Guardar contenido de URLs en estado global para Topic Authority
        if extract_url_content:
            url_content_state['url_content_cache'] = url_content_cache
            url_content_state['kw_to_url'] = kw_to_url

        print(f"✅ Vectorización completada" + (" (con contenido de URLs)" if extract_url_content else ""))

        # === Buscar gaps semánticos ===
        print(f"\n🔄 Buscando gaps semánticos (similitud > {similarity_threshold})...")

        semantic_gaps = []
        if len(exact_only_comp) > 0 and len(keywords_own) > 0:
            embeddings_own = np.array([kw_to_embedding[kw] for kw in keywords_own])
            keywords_own_list = list(keywords_own)

            for comp_kw in exact_only_comp:
                comp_emb = kw_to_embedding[comp_kw].reshape(1, -1)
                similarities = cosine_similarity(comp_emb, embeddings_own)[0]
                max_sim_idx = np.argmax(similarities)
                max_sim = similarities[max_sim_idx]

                if max_sim >= similarity_threshold:
                    semantic_gaps.append({
                        'competitor_keyword': comp_kw,
                        'similar_own_keyword': keywords_own_list[max_sim_idx],
                        'similarity': float(max_sim)
                    })

        semantic_gaps_dict = {sg['competitor_keyword']: sg for sg in semantic_gaps}
        print(f"✅ {len(semantic_gaps)} gaps semánticos encontrados")

        # === Función para obtener métricas ===
        def get_metrics(df, kw_col, keyword, metric_cols):
            rows = df[df[kw_col] == keyword]
            if len(rows) == 0:
                return {}
            row = rows.iloc[0]
            metrics = {}
            # Siempre incluir URL si existe
            url_columns = ['URL', 'url', 'Url', 'Ranked URL', 'Landing Page', 'Page', 'Target URL']
            for url_col in url_columns:
                if url_col in df.columns:
                    val = row[url_col]
                    if pd.notna(val) and isinstance(val, str):
                        metrics['URL'] = val
                        break
            # Incluir otras métricas
            for col in metric_cols:
                if col in df.columns:
                    val = row[col]
                    if pd.notna(val):
                        if isinstance(val, (np.integer, np.int32, np.int64)):
                            metrics[col] = int(val)
                        elif isinstance(val, (np.floating, np.float32, np.float64)):
                            metrics[col] = float(val)
                        else:
                            metrics[col] = val
            return metrics

        # === Preparar lista de GAPS con toda la info ===
        print(f"\n🔄 Preparando datos de gaps...")

        gaps_list = []
        gaps_embeddings_list = []

        for kw in exact_only_comp:
            # Métricas (del primer competidor que lo tenga)
            metrics = {}
            for comp in competitors_data:
                if kw in comp['keywords']:
                    metrics = get_metrics(comp['df'], comp['kw_col'], kw, metric_columns)
                    break

            # Competidores que tienen este keyword
            competitors_with_kw = all_competitor_keywords.get(kw, [])

            # Similitud con keyword propio
            sg = semantic_gaps_dict.get(kw)
            has_similar = sg is not None
            similar_own = sg['similar_own_keyword'] if sg else None
            similarity = sg['similarity'] if sg else 0

            # Intención de búsqueda
            intent_type, intent_confidence = detect_search_intent(kw)

            # Score de oportunidad
            score = calculate_opportunity_score(
                metrics,
                competitors_count=len(competitors_with_kw),
                has_similar_own=has_similar,
                similarity=similarity
            )

            gap_data = {
                'keyword': kw,
                'competitors': competitors_with_kw,
                'competitors_count': len(competitors_with_kw),
                'similar_own_keyword': similar_own,
                'similarity': similarity,
                'intent_type': intent_type,
                'intent_confidence': intent_confidence,
                'opportunity_score': score,
                **metrics
            }

            gaps_list.append(gap_data)
            gaps_embeddings_list.append(kw_to_embedding[kw])

        # Ordenar por score
        gaps_list.sort(key=lambda x: x['opportunity_score'], reverse=True)

        print(f"✅ {len(gaps_list)} gaps preparados con scoring e intención")

        # === CLUSTERING de gaps ===
        print(f"\n🔄 Clustering de gaps...")

        clusters_data = []

        if len(gaps_list) >= min_cluster_size:
            gaps_embeddings_array = np.array(gaps_embeddings_list)

            # Usar HDBSCAN o KMeans según tamaño
            if len(gaps_list) > 50:
                # HDBSCAN para datasets más grandes
                try:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=max(min_cluster_size, 3),
                        min_samples=2,
                        metric='euclidean',
                        cluster_selection_method='eom'
                    )
                    cluster_labels = clusterer.fit_predict(gaps_embeddings_array)
                    print(f"   Usando HDBSCAN: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters")
                except Exception as e:
                    print(f"   HDBSCAN falló, usando KMeans: {e}")
                    n_clust = min(num_clusters, len(gaps_list) // 3)
                    kmeans = KMeans(n_clusters=max(2, n_clust), random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(gaps_embeddings_array)
            else:
                # KMeans para datasets pequeños
                n_clust = min(num_clusters, max(2, len(gaps_list) // 3))
                kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(gaps_embeddings_array)
                print(f"   Usando KMeans: {n_clust} clusters")

            # Asignar cluster a cada gap
            for i, gap in enumerate(gaps_list):
                gap['cluster_id'] = int(cluster_labels[i])

            # Agrupar gaps por cluster
            clusters = {}
            for gap in gaps_list:
                cid = gap['cluster_id']
                if cid not in clusters:
                    clusters[cid] = []
                clusters[cid].append(gap)

            # Generar info de cada cluster
            for cid, cluster_gaps in clusters.items():
                if cid == -1:  # Ruido de HDBSCAN
                    cluster_name = "Sin clasificar"
                else:
                    # Nombre basado en keywords más comunes
                    top_keywords = sorted(cluster_gaps, key=lambda x: x['opportunity_score'], reverse=True)[:5]

                    if use_llm_names and ollama_available:
                        # Usar LLM para nombre natural
                        kw_sample = [g['keyword'] for g in top_keywords]
                        try:
                            cluster_name = generate_cluster_name_with_llm(kw_sample)
                        except:
                            # Fallback a extracción simple
                            words = ' '.join([g['keyword'] for g in cluster_gaps[:10]]).split()
                            word_freq = {}
                            for w in words:
                                if len(w) > 3:
                                    word_freq[w] = word_freq.get(w, 0) + 1
                            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
                            cluster_name = ' '.join([w[0] for w in top_words]).title()
                    else:
                        # Nombre basado en palabras frecuentes
                        words = ' '.join([g['keyword'] for g in cluster_gaps[:15]]).split()
                        word_freq = {}
                        stopwords = {'de', 'la', 'el', 'en', 'y', 'a', 'para', 'con', 'por', 'que', 'un', 'una', 'los', 'las', 'del'}
                        for w in words:
                            if len(w) > 2 and w not in stopwords:
                                word_freq[w] = word_freq.get(w, 0) + 1
                        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
                        cluster_name = ' '.join([w[0].capitalize() for w in top_words])

                # Calcular métricas agregadas del cluster
                total_volume = sum(g.get('Volume', g.get('Volumen', 0)) or 0 for g in cluster_gaps)
                avg_score = sum(g['opportunity_score'] for g in cluster_gaps) / len(cluster_gaps)
                avg_kd = sum(g.get('KD', g.get('Keyword Difficulty', 50)) or 50 for g in cluster_gaps) / len(cluster_gaps)

                # Intención dominante
                intent_counts = {}
                for g in cluster_gaps:
                    it = g['intent_type']
                    intent_counts[it] = intent_counts.get(it, 0) + 1
                dominant_intent = max(intent_counts.items(), key=lambda x: x[1])[0]

                # Competidores que más aparecen en este cluster
                comp_counts = {}
                for g in cluster_gaps:
                    for c in g['competitors']:
                        comp_counts[c] = comp_counts.get(c, 0) + 1
                top_competitors = sorted(comp_counts.items(), key=lambda x: x[1], reverse=True)[:3]

                clusters_data.append({
                    'cluster_id': cid,
                    'cluster_name': cluster_name if cluster_name else f'Cluster {cid}',
                    'keywords_count': len(cluster_gaps),
                    'total_volume': int(total_volume),
                    'avg_opportunity_score': round(avg_score, 1),
                    'avg_kd': round(avg_kd, 1),
                    'dominant_intent': dominant_intent,
                    'top_competitors': [{'name': c[0], 'count': c[1]} for c in top_competitors],
                    'top_keywords': [g['keyword'] for g in sorted(cluster_gaps, key=lambda x: x['opportunity_score'], reverse=True)[:5]],
                    'keywords': [g['keyword'] for g in cluster_gaps]
                })

            # Ordenar clusters por score promedio
            clusters_data.sort(key=lambda x: x['avg_opportunity_score'], reverse=True)

            print(f"✅ {len(clusters_data)} clusters generados")
        else:
            # Sin suficientes gaps para clustering
            for gap in gaps_list:
                gap['cluster_id'] = 0
            print(f"   ⚠️ Muy pocos gaps para clustering ({len(gaps_list)})")

        # === UMAP para visualización ===
        print(f"\n🔄 Calculando UMAP 2D...")

        umap_reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, len(all_keywords) - 1),
            min_dist=min_dist,
            metric='cosine',
            random_state=42
        )
        coords_2d = umap_reducer.fit_transform(embeddings)
        kw_to_coords = {kw: coords_2d[i] for i, kw in enumerate(all_keywords)}

        print(f"✅ UMAP completado")

        # === Añadir coordenadas a gaps ===
        for gap in gaps_list:
            coords = kw_to_coords[gap['keyword']]
            gap['x'] = float(coords[0])
            gap['y'] = float(coords[1])

        # === Preparar datos de visualización propios ===
        own_data = []
        for kw in keywords_own:
            coords = kw_to_coords[kw]
            metrics = get_metrics(df_own, kw_col_own, kw, metric_columns)
            intent_type, intent_conf = detect_search_intent(kw)
            own_data.append({
                'keyword': kw,
                'x': float(coords[0]),
                'y': float(coords[1]),
                'source': 'own',
                'status': 'shared' if kw in exact_shared else 'exclusive',
                'intent_type': intent_type,
                **metrics
            })

        # === Preparar datos de competidores ===
        competitor_data = []
        seen_comp_kw = set()
        for comp in competitors_data:
            for kw in comp['keywords']:
                if kw not in seen_comp_kw:
                    coords = kw_to_coords[kw]
                    metrics = get_metrics(comp['df'], comp['kw_col'], kw, metric_columns)
                    intent_type, _ = detect_search_intent(kw)
                    competitor_data.append({
                        'keyword': kw,
                        'x': float(coords[0]),
                        'y': float(coords[1]),
                        'source': 'competitor',
                        'competitor_name': comp['name'],
                        'competitors': all_competitor_keywords.get(kw, []),
                        'status': 'shared' if kw in exact_shared else 'gap',
                        'intent_type': intent_type,
                        **metrics
                    })
                    seen_comp_kw.add(kw)

        # === Oportunidades (keywords propios sin competencia) ===
        opportunities_list = []
        for kw in exact_only_own:
            coords = kw_to_coords[kw]
            metrics = get_metrics(df_own, kw_col_own, kw, metric_columns)
            intent_type, _ = detect_search_intent(kw)
            opportunities_list.append({
                'keyword': kw,
                'x': float(coords[0]),
                'y': float(coords[1]),
                'intent_type': intent_type,
                **metrics
            })

        # === Keywords compartidas ===
        shared_list = []
        for kw in exact_shared:
            coords = kw_to_coords[kw]
            metrics_own = get_metrics(df_own, kw_col_own, kw, metric_columns)

            # Métricas de cada competidor
            comp_metrics = {}
            for comp in competitors_data:
                if kw in comp['keywords']:
                    comp_metrics[comp['name']] = get_metrics(comp['df'], comp['kw_col'], kw, metric_columns)

            intent_type, _ = detect_search_intent(kw)
            shared_list.append({
                'keyword': kw,
                'x': float(coords[0]),
                'y': float(coords[1]),
                'own_metrics': metrics_own,
                'competitor_metrics': comp_metrics,
                'competitors': list(comp_metrics.keys()),
                'intent_type': intent_type
            })

        # === Guardar estado para proyección ===
        global gap_analysis_state_v2
        gap_analysis_state_v2['embeddings'] = embeddings
        gap_analysis_state_v2['coords_2d'] = coords_2d
        gap_analysis_state_v2['keywords'] = own_data + [d for d in competitor_data if d['status'] != 'shared']
        gap_analysis_state_v2['all_keywords_list'] = all_keywords  # Lista de strings en mismo orden que embeddings
        gap_analysis_state_v2['gaps_data'] = gaps_list

        # === Resumen por competidor ===
        competitors_summary = []
        for comp in competitors_data:
            comp_gaps = [kw for kw in exact_only_comp if comp['name'] in all_competitor_keywords.get(kw, [])]
            comp_shared = [kw for kw in exact_shared if comp['name'] in all_competitor_keywords.get(kw, [])]
            competitors_summary.append({
                'name': comp['name'],
                'total_keywords': len(comp['keywords']),
                'gaps_from_this': len(comp_gaps),
                'shared_with_this': len(comp_shared)
            })

        print(f"\n{'='*70}")
        print(f"✅ ANÁLISIS V2 COMPLETADO")
        print(f"{'='*70}")
        print(f"   Keywords propios: {len(own_data)}")
        print(f"   Keywords competidores: {len(competitor_data)}")
        print(f"   GAPS totales: {len(gaps_list)}")
        print(f"   Clusters de gaps: {len(clusters_data)}")
        print(f"   Fortalezas: {len(opportunities_list)}")
        print(f"   Competencia directa: {len(shared_list)}")

        response_data = {
            'version': '2.0',
            'own_data': convert_to_native(own_data),
            'competitor_data': convert_to_native(competitor_data),
            'gaps': convert_to_native(gaps_list),
            'gaps_clusters': convert_to_native(clusters_data),
            'opportunities': convert_to_native(opportunities_list),
            'shared': convert_to_native(shared_list),
            'semantic_gaps': convert_to_native(semantic_gaps),
            'competitors_summary': convert_to_native(competitors_summary),
            'summary': {
                'total_own': len(keywords_own),
                'total_competitors': len(competitors_data),
                'total_competitor_keywords': len(all_comp_keywords),
                'total_gaps': len(gaps_list),
                'total_clusters': len(clusters_data),
                'total_opportunities': len(opportunities_list),
                'total_shared': len(shared_list),
                'semantic_gaps_found': len(semantic_gaps),
                'avg_opportunity_score': round(sum(g['opportunity_score'] for g in gaps_list) / len(gaps_list), 1) if gaps_list else 0
            }
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error en análisis de gaps v2: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# REFUERZO DE CLUSTERS - Análisis de oportunidades por cluster
# ============================================================================

@app.route('/analyze_cluster_reinforcement', methods=['POST'])
def analyze_cluster_reinforcement():
    """
    Análisis de Refuerzo de Clusters.

    Flujo:
    1. Clusteriza los keywords PROPIOS por semántica
    2. Identifica qué keywords de competidores caen en cada cluster (GAPS)
    3. Calcula score de oportunidad para cada gap
    4. Sugiere expansiones basadas en patrones semánticos

    Returns:
        - clusters: Lista de clusters con keywords propios y gaps
        - all_keywords: Todos los puntos para visualización UMAP
        - summary: Estadísticas generales
    """
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    try:
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.cluster import KMeans
        import hdbscan

        print(f"\n{'='*70}")
        print(f"🎯 ANÁLISIS DE REFUERZO DE CLUSTERS")
        print(f"{'='*70}")

        # === Leer archivos ===
        own_file = request.files.get('own_file')
        if not own_file:
            return jsonify({'error': 'Se requiere archivo propio (own_file)'}), 400

        competitor_files = request.files.getlist('competitor_files')
        if not competitor_files:
            return jsonify({'error': 'Se requiere al menos un archivo de competidor'}), 400

        # === Parámetros ===
        keyword_column = request.form.get('keyword_column', 'Keyword').strip()
        similarity_threshold = float(request.form.get('similarity_threshold', 0.7))
        num_clusters = int(request.form.get('num_clusters', 15))
        min_cluster_size = int(request.form.get('min_cluster_size', 3))

        print(f"   Archivo propio: {own_file.filename}")
        print(f"   Competidores: {len(competitor_files)} archivos")
        print(f"   Clusters objetivo: {num_clusters}")
        print(f"   Umbral similitud: {similarity_threshold}")

        # === Función para encontrar columna de keywords ===
        def find_keyword_column(df, hint):
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in [hint.lower(), 'keyword', 'query', 'keywords', 'palabra clave', 'top queries']:
                    return col
            return df.columns[0]

        # === Función para extraer métricas ===
        def get_metrics(df, kw_col, keyword):
            row = df[df[kw_col] == keyword]
            if row.empty:
                return {}
            row = row.iloc[0]
            metrics = {}
            metric_mappings = {
                'volume': ['Volume', 'Search Volume', 'Volumen', 'Avg. monthly searches'],
                'position': ['Position', 'Posición', 'Pos', 'Ranking'],
                'kd': ['KD', 'Keyword Difficulty', 'Difficulty', 'KD%'],
                'traffic': ['Traffic', 'Tráfico', 'Organic Traffic'],
                'cpc': ['CPC', 'Cost per click'],
                'url': ['URL', 'url', 'Url', 'Ranked URL', 'Landing Page', 'Page', 'Target URL', 'Current URL']
            }
            for metric_name, possible_cols in metric_mappings.items():
                for col in possible_cols:
                    if col in df.columns:
                        val = row.get(col)
                        if pd.notna(val):
                            if metric_name == 'url':
                                metrics[metric_name] = str(val)
                            else:
                                try:
                                    metrics[metric_name] = float(val)
                                except:
                                    pass
                        break
            return metrics

        # === Cargar archivo propio ===
        df_own = pd.read_excel(own_file)
        kw_col_own = find_keyword_column(df_own, keyword_column)
        df_own = df_own[df_own[kw_col_own].notna()].copy()
        df_own[kw_col_own] = df_own[kw_col_own].astype(str).str.strip().str.lower()
        df_own = df_own[df_own[kw_col_own] != '']
        keywords_own = list(df_own[kw_col_own].unique())

        print(f"\n✅ Archivo propio: {len(keywords_own)} keywords únicos")

        # === Cargar archivos de competidores ===
        competitors_data = []
        all_competitor_keywords = {}

        for comp_file in competitor_files:
            comp_name = comp_file.filename.replace('.xlsx', '').replace('.xls', '').replace('.csv', '')
            df_comp = pd.read_excel(comp_file)
            kw_col_comp = find_keyword_column(df_comp, keyword_column)

            df_comp = df_comp[df_comp[kw_col_comp].notna()].copy()
            df_comp[kw_col_comp] = df_comp[kw_col_comp].astype(str).str.strip().str.lower()
            df_comp = df_comp[df_comp[kw_col_comp] != '']

            keywords_comp = list(df_comp[kw_col_comp].unique())

            competitors_data.append({
                'name': comp_name,
                'df': df_comp,
                'kw_col': kw_col_comp,
                'keywords': set(keywords_comp)
            })

            for kw in keywords_comp:
                if kw not in all_competitor_keywords:
                    all_competitor_keywords[kw] = []
                all_competitor_keywords[kw].append(comp_name)

            print(f"   ✅ Competidor '{comp_name}': {len(keywords_comp)} keywords")

        # === Identificar GAPS (keywords de competidores que no tenemos) ===
        keywords_own_set = set(keywords_own)
        gap_keywords = [kw for kw in all_competitor_keywords.keys() if kw not in keywords_own_set]

        print(f"\n📊 Resumen:")
        print(f"   - Keywords propios: {len(keywords_own)}")
        print(f"   - Keywords competidores únicos: {len(all_competitor_keywords)}")
        print(f"   - GAPS identificados: {len(gap_keywords)}")

        # === Vectorizar TODOS los keywords (propios + gaps) CON CACHÉ ===
        all_keywords = keywords_own + gap_keywords
        print(f"\n🔄 Vectorizando {len(all_keywords)} keywords (con caché SQLite)...")
        all_embeddings = encode_with_cache(all_keywords, model_768)
        print(f"✅ Embeddings generados: {all_embeddings.shape}")

        # Separar embeddings propios
        own_embeddings = all_embeddings[:len(keywords_own)]
        gap_embeddings = all_embeddings[len(keywords_own):]

        # === Clusterizar keywords PROPIOS ===
        print(f"\n🔄 Clusterizando {len(keywords_own)} keywords propios...")

        if len(keywords_own) >= num_clusters:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            own_cluster_labels = kmeans.fit_predict(own_embeddings)
            cluster_centers = kmeans.cluster_centers_
        else:
            # Si hay pocos keywords, usar HDBSCAN
            clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, min_cluster_size), metric='euclidean')
            own_cluster_labels = clusterer.fit_predict(own_embeddings)
            # Calcular centros manualmente
            unique_labels = set(own_cluster_labels)
            cluster_centers = []
            for label in sorted(unique_labels):
                if label >= 0:
                    mask = own_cluster_labels == label
                    center = own_embeddings[mask].mean(axis=0)
                    cluster_centers.append(center)
            cluster_centers = np.array(cluster_centers) if cluster_centers else np.array([])

        print(f"✅ {len(set(own_cluster_labels)) - (1 if -1 in own_cluster_labels else 0)} clusters generados")

        # === Asignar gaps a clusters por similitud semántica ===
        print(f"\n🔄 Asignando {len(gap_keywords)} gaps a clusters...")

        gap_cluster_assignments = []
        gap_similarities = []

        if len(cluster_centers) > 0 and len(gap_embeddings) > 0:
            # Calcular similitud de cada gap con cada centro de cluster
            similarities = cosine_similarity(gap_embeddings, cluster_centers)

            for i, gap_kw in enumerate(gap_keywords):
                max_sim_idx = np.argmax(similarities[i])
                max_sim = similarities[i][max_sim_idx]

                if max_sim >= similarity_threshold:
                    gap_cluster_assignments.append(max_sim_idx)
                    gap_similarities.append(max_sim)
                else:
                    gap_cluster_assignments.append(-1)  # No asignado
                    gap_similarities.append(max_sim)

        # === Generar nombres de clusters basados en keywords más representativos ===
        def generate_cluster_name(cluster_keywords, max_words=3):
            if not cluster_keywords:
                return "Sin nombre"

            # Contar palabras
            word_counts = {}
            stopwords = {'de', 'la', 'el', 'en', 'y', 'a', 'los', 'las', 'del', 'un', 'una', 'para', 'con', 'por', 'que', 'se', 'es', 'al', 'como'}

            for kw in cluster_keywords:
                words = kw.lower().split()
                for word in words:
                    if len(word) > 2 and word not in stopwords:
                        word_counts[word] = word_counts.get(word, 0) + 1

            # Top palabras
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_words]
            return ' + '.join([w[0].capitalize() for w in top_words]) if top_words else "Cluster"

        # === Construir datos de clusters ===
        clusters_data = []
        unique_clusters = sorted(set(own_cluster_labels))

        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue

            # Keywords propios en este cluster
            own_in_cluster = [keywords_own[i] for i, label in enumerate(own_cluster_labels) if label == cluster_id]

            # Gaps asignados a este cluster
            gaps_in_cluster = []
            for i, gap_kw in enumerate(gap_keywords):
                if i < len(gap_cluster_assignments) and gap_cluster_assignments[i] == cluster_id:
                    # Obtener métricas del gap
                    metrics = {}
                    for comp in competitors_data:
                        if gap_kw in comp['keywords']:
                            metrics = get_metrics(comp['df'], comp['kw_col'], gap_kw)
                            break

                    # Calcular score de oportunidad
                    volume = metrics.get('volume', 0)
                    kd = metrics.get('kd', 50)
                    similarity = gap_similarities[i] if i < len(gap_similarities) else 0

                    # Score: alto volumen, baja dificultad, alta similitud = mejor oportunidad
                    volume_score = min(100, (volume / 1000) * 20) if volume else 0
                    kd_score = max(0, 100 - kd)
                    similarity_score = similarity * 100

                    opportunity_score = (volume_score * 0.4 + kd_score * 0.3 + similarity_score * 0.3)

                    gaps_in_cluster.append({
                        'keyword': gap_kw,
                        'similarity': float(similarity),
                        'competitors': all_competitor_keywords.get(gap_kw, []),
                        'opportunity_score': float(opportunity_score),
                        **metrics
                    })

            # Calcular métricas del cluster
            total_volume = sum(g.get('volume', 0) for g in gaps_in_cluster)
            avg_opportunity = np.mean([g['opportunity_score'] for g in gaps_in_cluster]) if gaps_in_cluster else 0

            # Ordenar gaps por oportunidad
            gaps_in_cluster.sort(key=lambda x: x['opportunity_score'], reverse=True)

            # Keywords propios con métricas
            own_keywords_data = []
            for kw in own_in_cluster:
                metrics = get_metrics(df_own, kw_col_own, kw)
                own_keywords_data.append({
                    'keyword': kw,
                    **metrics
                })

            cluster_name = generate_cluster_name(own_in_cluster)

            clusters_data.append({
                'id': str(cluster_id),
                'name': cluster_name,
                'own_keywords_count': len(own_in_cluster),
                'gap_keywords_count': len(gaps_in_cluster),
                'total_volume': int(total_volume),
                'avg_opportunity_score': float(avg_opportunity),
                'own_keywords': own_keywords_data,
                'gap_keywords': gaps_in_cluster,
                'expansion_suggestions': []  # Se puede expandir más adelante
            })

        # Ordenar clusters por cantidad de gaps (oportunidades)
        clusters_data.sort(key=lambda x: x['gap_keywords_count'], reverse=True)

        # === UMAP para visualización ===
        print(f"\n🔄 Calculando UMAP 2D...")

        n_neighbors_umap = min(15, len(all_keywords) - 1)
        umap_reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors_umap,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        coords_2d = umap_reducer.fit_transform(all_embeddings)

        # Preparar datos para visualización
        all_keywords_data = []

        # Keywords propios
        for i, kw in enumerate(keywords_own):
            metrics = get_metrics(df_own, kw_col_own, kw)
            all_keywords_data.append({
                'keyword': kw,
                'x': float(coords_2d[i][0]),
                'y': float(coords_2d[i][1]),
                'source': 'own',
                'cluster_id': str(own_cluster_labels[i]) if own_cluster_labels[i] >= 0 else 'noise',
                **metrics
            })

        # Gaps
        for i, kw in enumerate(gap_keywords):
            idx = len(keywords_own) + i
            cluster_assignment = gap_cluster_assignments[i] if i < len(gap_cluster_assignments) else -1

            metrics = {}
            for comp in competitors_data:
                if kw in comp['keywords']:
                    metrics = get_metrics(comp['df'], comp['kw_col'], kw)
                    break

            all_keywords_data.append({
                'keyword': kw,
                'x': float(coords_2d[idx][0]),
                'y': float(coords_2d[idx][1]),
                'source': 'competitor',
                'cluster_id': str(cluster_assignment) if cluster_assignment >= 0 else 'unassigned',
                'competitors': all_competitor_keywords.get(kw, []),
                **metrics
            })

        print(f"\n{'='*70}")
        print(f"✅ ANÁLISIS DE REFUERZO COMPLETADO")
        print(f"{'='*70}")
        print(f"   Clusters generados: {len(clusters_data)}")
        print(f"   Keywords propios: {len(keywords_own)}")
        print(f"   Gaps totales: {len(gap_keywords)}")
        print(f"   Gaps asignados a clusters: {sum(1 for a in gap_cluster_assignments if a >= 0)}")

        response_data = {
            'clusters': convert_to_native(clusters_data),
            'all_keywords': convert_to_native(all_keywords_data),
            'summary': {
                'total_own_keywords': len(keywords_own),
                'total_gaps': len(gap_keywords),
                'total_clusters': len(clusters_data),
                'gaps_assigned': sum(1 for a in gap_cluster_assignments if a >= 0),
                'competitors_count': len(competitors_data)
            }
        }

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error en análisis de refuerzo: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500



# ============================================================================


# ============================================================================
# ENDPOINTS DE CACHE DE ANÁLISIS
# ============================================================================

@app.route('/list_analyses', methods=['GET'])
def list_analyses():
    """Lista los análisis guardados en cache"""
    try:
        limit = request.args.get('limit', 20, type=int)
        analyses = list_cached_analyses(limit)
        return jsonify({'analyses': analyses, 'count': len(analyses)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/load_analysis/<analysis_id>', methods=['GET'])
def load_analysis(analysis_id):
    """Carga un análisis guardado desde cache"""
    try:
        data = load_analysis_from_cache(analysis_id)
        if not data:
            return jsonify({'error': 'Análisis no encontrado'}), 404
        return jsonify({
            'clusters': data['clusters'],
            'unassigned_gaps': data['unassigned_gaps'],
            'summary': data['summary'],
            'total_unassigned': len(data['unassigned_gaps']),
            'meta': data['meta'],
            'from_cache': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/delete_analysis/<analysis_id>', methods=['DELETE'])
def delete_analysis(analysis_id):
    """Elimina un análisis del cache"""
    try:
        deleted = delete_analysis_from_cache(analysis_id)
        if deleted:
            return jsonify({'success': True, 'message': 'Análisis eliminado'})
        return jsonify({'error': 'Análisis no encontrado'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# REFUERZO DE CLUSTERS V2 - SIN UMAP (MAS RAPIDO)
# ============================================================================

@app.route('/analyze_cluster_reinforcement_v2', methods=['POST'])
def analyze_cluster_reinforcement_v2():
    """Analisis de Refuerzo V2 - Sin UMAP."""
    def conv(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)): return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, dict): return {k: conv(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [conv(i) for i in obj]
        return obj
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.cluster import KMeans
        import hdbscan
        print(f"\n{'='*60}\n REFUERZO V2 (SIN UMAP)\n{'='*60}")
        own_file = request.files.get('own_file')
        if not own_file: return jsonify({'error': 'Se requiere archivo propio'}), 400
        competitor_files = request.files.getlist('competitor_files')
        if not competitor_files: return jsonify({'error': 'Se requiere al menos un competidor'}), 400
        kw_hint = request.form.get('keyword_column', 'Keyword').strip()
        sim_thresh = float(request.form.get('similarity_threshold', 0.7))
        n_clusters = int(request.form.get('num_clusters', 15))
        clustering_method = request.form.get('clustering_method', 'kmeans').lower()  # kmeans, hdbscan, louvain
        louvain_resolution = float(request.form.get('louvain_resolution', 1.0))
        print(f"   Método de clustering: {clustering_method.upper()}")
        def find_col(df, hint):
            for c in df.columns:
                if c.lower().strip() in [hint.lower(), 'keyword', 'query', 'keywords']: return c
            return df.columns[0]
        def get_m(df, col, kw):
            row = df[df[col] == kw]
            if row.empty: return {}
            row = row.iloc[0]
            m = {}
            for n, cs in {'volume': ['Volume','Search Volume'], 'position': ['Position','Pos'],
                         'kd': ['KD','Keyword Difficulty'], 'traffic': ['Traffic'],
                         'url': ['URL','url','Ranked URL','Landing Page']}.items():
                for c in cs:
                    if c in df.columns and pd.notna(row.get(c)):
                        m[n] = str(row[c]) if n == 'url' else (float(row[c]) if not isinstance(row[c], str) else row[c])
                        break
            return m
        df_own = pd.read_excel(own_file)
        col_own = find_col(df_own, kw_hint)
        df_own = df_own[df_own[col_own].notna()].copy()
        df_own[col_own] = df_own[col_own].astype(str).str.strip().str.lower()
        kws_own = list(df_own[df_own[col_own] != ''][col_own].unique())
        print(f"Propios: {len(kws_own)}")
        comps, all_comp = [], {}
        for cf in competitor_files:
            nm = cf.filename.replace('.xlsx','').replace('.xls','')
            dfc = pd.read_excel(cf)
            col = find_col(dfc, kw_hint)
            dfc = dfc[dfc[col].notna()].copy()
            dfc[col] = dfc[col].astype(str).str.strip().str.lower()
            kws = list(dfc[dfc[col] != ''][col].unique())
            comps.append({'name': nm, 'df': dfc, 'col': col, 'kws': set(kws)})
            for k in kws:
                if k not in all_comp: all_comp[k] = []
                all_comp[k].append(nm)
            print(f"   {nm}: {len(kws)}")
        gaps = [k for k in all_comp if k not in set(kws_own)]
        print(f"GAPS: {len(gaps)}")
        all_kws = kws_own + gaps
        print(f"Vectorizando {len(all_kws)}...")
        embs = encode_with_cache(all_kws, model_768)
        own_emb, gap_emb = embs[:len(kws_own)], embs[len(kws_own):]

        # Clasificar intención de búsqueda
        print("Clasificando intención...")
        intent_embs = get_intent_embeddings(model_768)
        all_intents = [classify_intent(kw, embs[i], intent_embs) for i, kw in enumerate(all_kws)]
        own_intents = all_intents[:len(kws_own)]
        gap_intents = all_intents[len(kws_own):]

        # Clustering según método seleccionado
        print(f"Aplicando clustering ({clustering_method})...")
        if clustering_method == 'louvain':
            labels, _ = cluster_with_louvain(own_emb, similarity_threshold=sim_thresh, resolution=louvain_resolution)
            # Calcular centroides para cada cluster
            unique_labels = sorted(set(labels))
            centers = np.array([own_emb[labels == l].mean(0) for l in unique_labels])
        elif clustering_method == 'hdbscan':
            hdb = hdbscan.HDBSCAN(min_cluster_size=max(2, len(kws_own) // 20), metric='euclidean')
            labels = hdb.fit_predict(own_emb)
            centers = np.array([own_emb[labels == l].mean(0) for l in sorted(set(labels)) if l >= 0]) if any(l >= 0 for l in labels) else np.array([])
            print(f"   HDBSCAN: {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
        else:  # kmeans (default)
            actual_n_clusters = min(n_clusters, len(kws_own))
            km = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(own_emb)
            centers = km.cluster_centers_
            print(f"   KMeans: {actual_n_clusters} clusters")
        g_assign, g_sims, g_rel = [], [], []
        if len(centers) > 0 and len(gap_emb) > 0:
            sc = cosine_similarity(gap_emb, centers)
            so = cosine_similarity(gap_emb, own_emb)
            for i in range(len(gaps)):
                mx_i, mx_s = np.argmax(sc[i]), sc[i].max()
                g_assign.append(mx_i if mx_s >= sim_thresh else -1)
                g_sims.append(mx_s)
                top = np.argsort(so[i])[-3:][::-1]
                g_rel.append([{'keyword': kws_own[j], 'similarity': float(so[i][j])} for j in top if so[i][j] >= 0.5])
        def gen_name(kws):
            if not kws: return "Cluster"
            stops = {'de','la','el','en','y','a','los','las','del','un','una','para','con','por','que'}
            cnt = {}
            for k in kws:
                for w in k.split():
                    if len(w) > 2 and w not in stops: cnt[w] = cnt.get(w,0)+1
            top = sorted(cnt.items(), key=lambda x: x[1], reverse=True)[:3]
            return ' + '.join([w[0].capitalize() for w in top]) or "Cluster"
        clusters, unassigned = [], []
        for cid in sorted(set(labels)):
            if cid == -1: continue
            own_in = [kws_own[i] for i,l in enumerate(labels) if l == cid]
            gaps_in = []
            for i, gk in enumerate(gaps):
                if i < len(g_assign) and g_assign[i] == cid:
                    m = next((get_m(c['df'], c['col'], gk) for c in comps if gk in c['kws']), {})
                    v, kd, nc = m.get('volume',0), m.get('kd',50), len(all_comp.get(gk,[]))
                    sim = g_sims[i] if i < len(g_sims) else 0
                    score = min(100,v/500*25)*0.35 + max(0,100-kd)*0.25 + min(100,nc*25)*0.2 + sim*100*0.2
                    intent_info = gap_intents[i] if i < len(gap_intents) else {'intent': 'mixed'}
                    gaps_in.append({'keyword': gk, 'similarity': float(sim), 'competitors': all_comp.get(gk,[]),
                                   'num_competitors': nc, 'opportunity_score': float(score),
                                   'intent': intent_info['intent'],
                                   'related_own_keywords': g_rel[i] if i < len(g_rel) else [], **m})
            gaps_in.sort(key=lambda x: x['opportunity_score'], reverse=True)
            # Keywords propios con intent y detección de débiles
            own_d = []
            weak_d = []
            for k in own_in:
                idx = kws_own.index(k)
                intent_info = own_intents[idx] if idx < len(own_intents) else {'intent': 'mixed'}
                m = get_m(df_own, col_own, k)
                pos = float(m.get('position', 0)) if m.get('position') else 0
                vol = float(m.get('volume', 0)) if m.get('volume') else 0
                traf = float(m.get('traffic', 0)) if m.get('traffic') else 0
                kd = float(m.get('kd', 50)) if m.get('kd') else 50
                comp_has = all_comp.get(k, [])  # Competidores que también tienen esta keyword
                # Calcular score de debilidad (0-100, mayor = más débil)
                weakness_score = 0
                weakness_reasons = []
                # Posición fuera de primera página
                if pos > 10:
                    weakness_score += min(40, (pos - 10) * 2)
                    weakness_reasons.append(f'Posición {int(pos)}')
                elif pos > 5:
                    weakness_score += (pos - 5) * 3
                    weakness_reasons.append(f'Posición {int(pos)}')
                # Competencia directa
                if comp_has:
                    weakness_score += min(30, len(comp_has) * 10)
                    weakness_reasons.append(f'{len(comp_has)} competidor(es)')
                # Tráfico bajo vs volumen (potencial no aprovechado)
                if vol > 0 and traf < vol * 0.1:
                    weakness_score += min(20, 20 * (1 - traf/vol))
                    weakness_reasons.append('Bajo CTR')
                # KD alta sin buen posicionamiento
                if kd > 50 and pos > 10:
                    weakness_score += min(10, (kd - 50) * 0.2)
                kw_data = {'keyword': k, 'intent': intent_info['intent'],
                          'weakness_score': round(weakness_score, 1),
                          'weakness_reasons': weakness_reasons,
                          'competitors_also_ranking': comp_has, **m}
                own_d.append(kw_data)
                # Es débil si score > 25
                if weakness_score > 25:
                    weak_d.append(kw_data)
            # Ordenar débiles por score
            weak_d.sort(key=lambda x: x['weakness_score'], reverse=True)
            vg = sum(g.get('volume',0) for g in gaps_in)
            vo = sum(get_m(df_own, col_own, k).get('volume',0) for k in own_in)
            ao = np.mean([g['opportunity_score'] for g in gaps_in]) if gaps_in else 0
            t = len(own_in) + len(gaps_in)
            # Estadísticas de intent del cluster
            all_cluster_intents = [o.get('intent','mixed') for o in own_d] + [g.get('intent','mixed') for g in gaps_in]
            intent_counts = {}
            for intent in all_cluster_intents:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            dominant_intent = max(intent_counts, key=intent_counts.get) if intent_counts else 'mixed'
            # Calcular Topic Authority
            authority = calculate_cluster_authority(own_d, gaps_in, len(own_in), len(gaps_in))

            # Detectar Canibalizacion (usar threshold del form o default 0.95)
            canib_threshold = float(request.form.get('cannibalization_threshold', 0.95))
            cannibalization = detect_intra_cluster_cannibalization(own_d, own_emb, kws_own, canib_threshold)

            # Agrupar gaps por concepto semantico
            if len(gaps_in) >= 2:
                # Obtener embeddings de los gaps de este cluster
                gap_indices = [gaps.index(g['keyword']) for g in gaps_in if g['keyword'] in gaps]
                if gap_indices:
                    cluster_gap_embs = gap_emb[gap_indices]
                    semantic_gaps = group_gaps_by_semantic_concept(gaps_in, cluster_gap_embs, own_d, own_emb, kws_own)
                else:
                    semantic_gaps = []
            else:
                semantic_gaps = []

            clusters.append({'id': str(cid), 'name': gen_name(own_in), 'own_keywords_count': len(own_in),
                           'gap_keywords_count': len(gaps_in), 'weak_keywords_count': len(weak_d),
                           'total_volume_gaps': int(vg), 'total_volume_own': int(vo),
                           'avg_opportunity_score': float(ao), 'coverage': float(len(own_in)/t*100 if t else 100),
                           'dominant_intent': dominant_intent, 'intent_distribution': intent_counts,
                           'authority': authority,
                           'cannibalization': cannibalization,
                           'semantic_gaps': semantic_gaps,
                           'own_keywords': own_d, 'gap_keywords': gaps_in[:50], 'weak_keywords': weak_d,
                           'total_gaps_in_cluster': len(gaps_in)})
        for i, gk in enumerate(gaps):
            if i < len(g_assign) and g_assign[i] == -1:
                m = next((get_m(c['df'], c['col'], gk) for c in comps if gk in c['kws']), {})
                v, kd, nc = m.get('volume',0), m.get('kd',50), len(all_comp.get(gk,[]))
                score = min(100,v/500*25)*0.4 + max(0,100-kd)*0.3 + min(100,nc*25)*0.3
                intent_info = gap_intents[i] if i < len(gap_intents) else {'intent': 'mixed'}
                unassigned.append({'keyword': gk, 'competitors': all_comp.get(gk,[]), 'num_competitors': nc,
                                  'opportunity_score': float(score), 'max_similarity': float(g_sims[i]) if i < len(g_sims) else 0,
                                  'intent': intent_info['intent'],
                                  'related_own_keywords': g_rel[i] if i < len(g_rel) else [], **m})
        unassigned.sort(key=lambda x: x['opportunity_score'], reverse=True)
        clusters.sort(key=lambda x: x['gap_keywords_count'] * x['avg_opportunity_score'], reverse=True)
        print(f"OK: {len(clusters)} clusters, {len(unassigned)} sin asignar")

        total_weak = sum(c['weak_keywords_count'] for c in clusters)
        total_cannibalization = sum(c['cannibalization']['summary']['total_pairs'] for c in clusters)
        avg_authority = np.mean([c['authority']['authority_score'] for c in clusters]) if clusters else 0
        high_authority = sum(1 for c in clusters if c['authority']['authority_score'] >= 70)
        low_authority = sum(1 for c in clusters if c['authority']['authority_score'] < 40)

        summary = {'total_own_keywords': len(kws_own), 'total_gaps': len(gaps),
                   'total_weak_keywords': total_weak,
                   'total_cannibalization_pairs': total_cannibalization,
                   'avg_authority_score': round(avg_authority, 1),
                   'high_authority_clusters': high_authority,
                   'low_authority_clusters': low_authority,
                   'total_clusters': len(clusters), 'gaps_assigned': sum(c['gap_keywords_count'] for c in clusters),
                   'gaps_unassigned': len(unassigned), 'competitors': [c['name'] for c in comps],
                   'competitors_count': len(comps)}

        # Guardar en cache automaticamente
        import uuid
        analysis_id = str(uuid.uuid4())[:12]
        own_hash = hashlib.md5(own_file.filename.encode()).hexdigest()[:8]
        comp_hashes = [hashlib.md5(c['name'].encode()).hexdigest()[:8] for c in comps]
        comp_names = [c['name'] for c in comps]
        analysis_name = request.form.get('analysis_name', f"{own_file.filename.split('.')[0]} vs {len(comps)} comp")
        params = {'similarity_threshold': sim_thresh, 'num_clusters': n_clusters}
        save_analysis_to_cache(analysis_id, analysis_name, own_file.filename, own_hash,
                               comp_names, comp_hashes, params, clusters, unassigned, summary)

        return jsonify(conv({'clusters': clusters, 'unassigned_gaps': unassigned[:100],
            'total_unassigned': len(unassigned), 'summary': summary,
            'analysis_id': analysis_id, 'analysis_name': analysis_name}))
    except Exception as e:
        print(f"Error V2: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# EXTRACCIÓN DE CONTENIDO DE URLs CON ADVERTOOLS
# ============================================================================

# Estado global para almacenar datos extraídos de URLs
url_content_state = {
    'own_urls_data': None,      # {url: {title, description, h1, h2s, keywords: [...]}}
    'competitor_urls_data': {},  # {competitor_name: {url: {...}}}
    'combined_embeddings': None, # Embeddings de contenido combinado
    'url_to_embedding_idx': {},  # Mapeo URL -> índice en embeddings
    'all_urls_info': []          # Lista completa de info de URLs
}

@app.route('/extract_url_content', methods=['POST'])
def extract_url_content():
    """
    Extrae contenido (título, descripción, encabezados) de las URLs usando advertools.
    Agrupa keywords por URL y crea embeddings combinados.
    """
    import concurrent.futures
    import requests
    from bs4 import BeautifulSoup

    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    def extract_page_content(url, timeout=10):
        """Extrae título, descripción y encabezados de una URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Título
            title = ''
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Meta descripción
            description = ''
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                description = meta_desc.get('content', '')

            # H1
            h1 = ''
            h1_tag = soup.find('h1')
            if h1_tag:
                h1 = h1_tag.get_text(strip=True)

            # H2s (primeros 5)
            h2s = []
            for h2_tag in soup.find_all('h2')[:5]:
                h2_text = h2_tag.get_text(strip=True)
                if h2_text:
                    h2s.append(h2_text)

            # H3s (primeros 5)
            h3s = []
            for h3_tag in soup.find_all('h3')[:5]:
                h3_text = h3_tag.get_text(strip=True)
                if h3_text:
                    h3s.append(h3_text)

            return {
                'url': url,
                'title': title,
                'description': description,
                'h1': h1,
                'h2s': h2s,
                'h3s': h3s,
                'success': True,
                'error': None
            }

        except Exception as e:
            return {
                'url': url,
                'title': '',
                'description': '',
                'h1': '',
                'h2s': [],
                'h3s': [],
                'success': False,
                'error': str(e)
            }

    try:
        print(f"\n{'='*70}")
        print(f"🌐 EXTRACCIÓN DE CONTENIDO DE URLs")
        print(f"{'='*70}")

        # Verificar que hay datos del análisis de gaps
        if gap_analysis_state_v2['keywords'] is None:
            return jsonify({'error': 'Primero debes ejecutar el análisis de Content Gaps V2'}), 400

        data = request.get_json()
        max_urls = data.get('max_urls', 100)  # Límite de URLs a procesar
        timeout_per_url = data.get('timeout', 10)

        # Obtener datos guardados
        stored_keywords = gap_analysis_state_v2['keywords']
        gaps_data = gap_analysis_state_v2.get('gaps_data', [])

        # Función para extraer URL de un diccionario con múltiples nombres posibles
        def get_url_from_data(data_dict):
            url_columns = ['URL', 'url', 'Url', 'Ranked URL', 'Landing Page', 'Page',
                          'Target URL', 'Highest Rank URL', 'Best URL', 'Current URL',
                          'landing_page', 'page_url', 'target_url']
            for col in url_columns:
                val = data_dict.get(col, '')
                if val and isinstance(val, str) and (val.startswith('http') or val.startswith('/')):
                    return val.strip()
            return ''

        # Agrupar keywords por URL
        print(f"\n📊 Agrupando keywords por URL...")
        print(f"   Keywords almacenados: {len(stored_keywords)}")

        # Debug: mostrar columnas disponibles en el primer keyword
        if stored_keywords and len(stored_keywords) > 0:
            print(f"   Columnas disponibles: {list(stored_keywords[0].keys())[:15]}...")

        url_keywords_map = {}  # {url: {'keywords': [], 'source': 'own'/'competitor', 'metrics': {}}}

        for kw_data in stored_keywords:
            url = get_url_from_data(kw_data)
            if not url:
                continue

            url = url.strip()
            source = kw_data.get('source', 'unknown')

            if url not in url_keywords_map:
                url_keywords_map[url] = {
                    'keywords': [],
                    'source': source,
                    'total_volume': 0,
                    'avg_position': 0,
                    'competitor_name': kw_data.get('competitor_name', '')
                }

            url_keywords_map[url]['keywords'].append(kw_data.get('keyword', ''))
            url_keywords_map[url]['total_volume'] += kw_data.get('Volume', 0) or 0

        # También procesar gaps_data
        for gap in gaps_data:
            url = get_url_from_data(gap)
            if not url:
                continue

            url = url.strip()

            if url not in url_keywords_map:
                url_keywords_map[url] = {
                    'keywords': [],
                    'source': 'competitor',
                    'total_volume': 0,
                    'avg_position': 0,
                    'competitor_name': ', '.join(gap.get('competitors', []))
                }

            if gap.get('keyword') not in url_keywords_map[url]['keywords']:
                url_keywords_map[url]['keywords'].append(gap.get('keyword', ''))
                url_keywords_map[url]['total_volume'] += gap.get('Volume', 0) or 0

        unique_urls = list(url_keywords_map.keys())[:max_urls]

        print(f"   URLs únicas encontradas: {len(url_keywords_map)}")
        print(f"   URLs a procesar: {len(unique_urls)}")

        if len(unique_urls) == 0:
            return jsonify({'error': 'No se encontraron URLs válidas en los datos. Asegúrate de que los Excel tengan una columna URL.'}), 400

        # Extraer contenido de URLs en paralelo
        print(f"\n🔄 Extrayendo contenido de {len(unique_urls)} URLs...")

        extracted_content = {}
        failed_urls = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {executor.submit(extract_page_content, url, timeout_per_url): url for url in unique_urls}

            for i, future in enumerate(concurrent.futures.as_completed(future_to_url)):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result['success']:
                        extracted_content[url] = result
                    else:
                        failed_urls.append({'url': url, 'error': result['error']})
                except Exception as e:
                    failed_urls.append({'url': url, 'error': str(e)})

                if (i + 1) % 10 == 0:
                    print(f"   Procesadas {i + 1}/{len(unique_urls)} URLs...")

        print(f"✅ Extracción completada: {len(extracted_content)} exitosas, {len(failed_urls)} fallidas")

        # Crear textos combinados para embeddings
        print(f"\n🔄 Creando embeddings combinados...")

        all_urls_info = []
        texts_for_embedding = []

        for url, content in extracted_content.items():
            url_data = url_keywords_map.get(url, {})
            keywords = url_data.get('keywords', [])

            # Combinar todo el contenido para embedding
            combined_text = f"{content['title']} {content['description']} {content['h1']} {' '.join(content['h2s'])} {' '.join(content['h3s'])} {' '.join(keywords[:20])}"
            combined_text = combined_text.strip()

            if not combined_text:
                combined_text = ' '.join(keywords[:20]) if keywords else url

            url_info = {
                'url': url,
                'title': content['title'],
                'description': content['description'],
                'h1': content['h1'],
                'h2s': content['h2s'],
                'h3s': content['h3s'],
                'keywords': keywords,
                'keywords_count': len(keywords),
                'total_volume': url_data.get('total_volume', 0),
                'source': url_data.get('source', 'unknown'),
                'competitor_name': url_data.get('competitor_name', ''),
                'combined_text': combined_text
            }

            all_urls_info.append(url_info)
            texts_for_embedding.append(combined_text)

        # Generar embeddings
        if texts_for_embedding:
            combined_embeddings = model_768.encode(
                texts_for_embedding,
                show_progress_bar=True,
                batch_size=32,
                convert_to_numpy=True
            )

            # Crear mapeo URL -> índice
            url_to_idx = {info['url']: i for i, info in enumerate(all_urls_info)}

            # Guardar en estado global
            global url_content_state
            url_content_state['combined_embeddings'] = combined_embeddings
            url_content_state['url_to_embedding_idx'] = url_to_idx
            url_content_state['all_urls_info'] = all_urls_info

            print(f"✅ Embeddings generados: {combined_embeddings.shape}")

        # Separar por source
        own_urls = [u for u in all_urls_info if u['source'] == 'own']
        competitor_urls = [u for u in all_urls_info if u['source'] != 'own']

        print(f"\n{'='*70}")
        print(f"✅ EXTRACCIÓN COMPLETADA")
        print(f"{'='*70}")
        print(f"   URLs propias: {len(own_urls)}")
        print(f"   URLs competidores: {len(competitor_urls)}")
        print(f"   URLs fallidas: {len(failed_urls)}")

        return jsonify({
            'success': True,
            'summary': {
                'total_urls_found': len(url_keywords_map),
                'urls_processed': len(unique_urls),
                'urls_extracted': len(extracted_content),
                'urls_failed': len(failed_urls),
                'own_urls': len(own_urls),
                'competitor_urls': len(competitor_urls),
                'embeddings_shape': list(combined_embeddings.shape) if texts_for_embedding else [0, 0]
            },
            'own_urls': convert_to_native(own_urls[:50]),  # Limitar respuesta
            'competitor_urls': convert_to_native(competitor_urls[:50]),
            'failed_urls': convert_to_native(failed_urls[:20])
        })

    except Exception as e:
        print(f"❌ Error extrayendo contenido: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# TOPIC AUTHORITY ANALYSIS - Informe de contenido por topics a mejorar
# ============================================================================

@app.route('/analyze_topic_authority', methods=['POST'])
def analyze_topic_authority():
    """
    Analiza la autoridad temática para topics específicos usando datos de contenido de URLs.
    Compara topics con embeddings enriquecidos (título + descripción + encabezados + keywords).
    Genera clusters de contenido recomendado basándose en el contenido REAL de las URLs.
    """
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    try:
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.cluster import KMeans

        print(f"\n{'='*70}")
        print(f"🎯 ANÁLISIS DE TOPIC AUTHORITY (CON CONTENIDO DE URLs)")
        print(f"{'='*70}")

        data = request.get_json()
        topics = data.get('topics', [])
        top_k = data.get('top_k', 30)
        min_similarity = data.get('min_similarity', 0.25)

        if not topics:
            return jsonify({'error': 'Se requiere al menos un topic para analizar'}), 400

        # Verificar que hay datos de URLs extraídos
        if url_content_state['combined_embeddings'] is None or len(url_content_state['all_urls_info']) == 0:
            return jsonify({'error': 'Primero debes extraer el contenido de las URLs (botón "Extraer Contenido")'}), 400

        print(f"   Topics a analizar: {topics}")
        print(f"   Top K URLs: {top_k}")
        print(f"   Similitud mínima: {min_similarity}")

        # Vectorizar los topics
        topic_embeddings = model_768.encode(
            topics,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        print(f"✅ Topics vectorizados: {topic_embeddings.shape}")

        # Obtener datos de URLs
        url_embeddings = url_content_state['combined_embeddings']
        all_urls_info = url_content_state['all_urls_info']

        topic_reports = []

        for i, topic in enumerate(topics):
            topic_emb = topic_embeddings[i].reshape(1, -1)

            # Calcular similitud con todos los embeddings de URLs
            similarities = cosine_similarity(topic_emb, url_embeddings)[0]

            # Encontrar URLs con similitud >= min_similarity
            matching_indices = np.where(similarities >= min_similarity)[0]

            # Ordenar por similitud
            sorted_indices = matching_indices[np.argsort(similarities[matching_indices])[::-1]]

            # Limitar a top_k
            top_indices = sorted_indices[:min(top_k, len(sorted_indices))]

            related_urls = []
            own_urls = []
            competitor_urls = []
            total_volume = 0
            all_keywords = []

            for idx in top_indices:
                url_info = all_urls_info[idx]
                similarity = float(similarities[idx])

                url_entry = {
                    'url': url_info['url'],
                    'title': url_info['title'],
                    'description': url_info['description'],
                    'h1': url_info['h1'],
                    'h2s': url_info['h2s'],
                    'h3s': url_info['h3s'],
                    'keywords': url_info['keywords'],
                    'keywords_count': url_info['keywords_count'],
                    'total_volume': url_info['total_volume'],
                    'source': url_info['source'],
                    'competitor_name': url_info.get('competitor_name', ''),
                    'similarity': similarity
                }

                related_urls.append(url_entry)
                all_keywords.extend(url_info['keywords'])
                total_volume += url_info['total_volume']

                if url_info['source'] == 'own':
                    own_urls.append(url_entry)
                else:
                    competitor_urls.append(url_entry)

            # Calcular cobertura
            coverage_percentage = (len(own_urls) / len(related_urls) * 100) if related_urls else 0

            # Generar clusters de contenido basados en URLs de competidores
            content_clusters = []

            if len(competitor_urls) >= 2:
                # Obtener embeddings de las URLs de competidores
                comp_indices = [all_urls_info.index(u) for u in all_urls_info
                               if u['url'] in [cu['url'] for cu in competitor_urls]]

                if len(comp_indices) >= 2:
                    comp_embeddings = url_embeddings[comp_indices]

                    # Clustering
                    n_clusters = min(max(2, len(competitor_urls) // 3), 5)
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(comp_embeddings)

                    # Agrupar por cluster
                    clusters = {}
                    for j, label in enumerate(cluster_labels):
                        if label not in clusters:
                            clusters[label] = []
                        clusters[label].append(competitor_urls[j])

                    # Generar info de cada cluster
                    for cluster_id, cluster_urls in clusters.items():
                        if len(cluster_urls) < 1:
                            continue

                        # Extraer todos los keywords y H2s del cluster
                        cluster_keywords = []
                        cluster_h2s = []
                        cluster_volume = 0

                        for cu in cluster_urls:
                            cluster_keywords.extend(cu['keywords'][:10])
                            cluster_h2s.extend(cu['h2s'])
                            cluster_volume += cu['total_volume']

                        # Nombre del cluster basado en contenido común
                        all_texts = cluster_keywords + cluster_h2s
                        words = ' '.join(all_texts).split()
                        word_freq = {}
                        stopwords = {'de', 'la', 'el', 'en', 'y', 'a', 'para', 'con', 'por', 'que', 'un', 'una', 'los', 'las', 'del', 'como', 'qué', 'es', 'más', 'tu', 'su', 'te', 'se'}
                        for w in words:
                            w_lower = w.lower()
                            if len(w_lower) > 3 and w_lower not in stopwords:
                                word_freq[w_lower] = word_freq.get(w_lower, 0) + 1
                        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:3]
                        cluster_name = ' '.join([w[0].capitalize() for w in top_words])

                        # Prioridad basada en volumen
                        priority = 'high' if cluster_volume > 10000 else 'medium' if cluster_volume > 3000 else 'low'

                        # Sugerencia de contenido basada en títulos y H2s de competidores
                        suggested_titles = list(set([cu['title'] for cu in cluster_urls if cu['title']]))[:3]
                        suggested_h2s = list(set(cluster_h2s))[:8]

                        content_clusters.append({
                            'name': cluster_name if cluster_name else f'Cluster {cluster_id + 1}',
                            'urls_count': len(cluster_urls),
                            'keywords': list(set(cluster_keywords))[:15],
                            'total_volume': int(cluster_volume),
                            'priority': priority,
                            'competitor_urls': [{'url': cu['url'], 'title': cu['title'], 'similarity': cu['similarity']} for cu in cluster_urls],
                            'suggested_titles': suggested_titles,
                            'suggested_h2s': suggested_h2s,
                            'content_brief': {
                                'topic': topic,
                                'main_keywords': list(set(cluster_keywords))[:10],
                                'suggested_structure': suggested_h2s[:5],
                                'reference_content': [cu['title'] for cu in cluster_urls[:3]]
                            }
                        })

                    # Ordenar por prioridad
                    priority_order = {'high': 0, 'medium': 1, 'low': 2}
                    content_clusters.sort(key=lambda x: (priority_order.get(x['priority'], 3), -x['total_volume']))

            # Generar recomendación
            if coverage_percentage >= 50:
                recommendation = f"Buena cobertura ({len(own_urls)} URLs propias). Optimiza el contenido existente con {len(list(set(all_keywords)))} keywords adicionales."
            elif coverage_percentage >= 25:
                recommendation = f"Cobertura media. Crea {len(content_clusters)} contenidos nuevos basándote en los competidores."
            else:
                recommendation = f"Baja cobertura. Los competidores tienen {len(competitor_urls)} URLs sobre este topic. Prioridad ALTA."

            # Añadir URLs propias relevantes
            own_urls_for_topic = [u for u in own_urls]

            topic_reports.append({
                'topic': topic,
                'total_urls': len(related_urls),
                'own_urls_count': len(own_urls),
                'competitor_urls_count': len(competitor_urls),
                'coverage_percentage': round(coverage_percentage, 1),
                'total_volume': int(total_volume),
                'total_keywords': len(list(set(all_keywords))),
                'recommendation': recommendation,
                'own_urls': own_urls_for_topic[:10],
                'competitor_urls': competitor_urls[:15],
                'content_clusters': content_clusters,
                'all_keywords': list(set(all_keywords))[:50]
            })

            print(f"\n   ✅ Topic '{topic}':")
            print(f"      URLs relacionadas: {len(related_urls)}")
            print(f"      Propias: {len(own_urls)}, Competidores: {len(competitor_urls)}")
            print(f"      Cobertura: {coverage_percentage:.1f}%")
            print(f"      Clusters de contenido: {len(content_clusters)}")

        print(f"\n{'='*70}")
        print(f"✅ ANÁLISIS DE TOPIC AUTHORITY COMPLETADO")
        print(f"{'='*70}")

        return jsonify({
            'success': True,
            'topic_reports': convert_to_native(topic_reports),
            'summary': {
                'total_topics': len(topics),
                'avg_coverage': round(sum(r['coverage_percentage'] for r in topic_reports) / len(topic_reports), 1) if topic_reports else 0,
                'total_content_clusters': sum(len(r['content_clusters']) for r in topic_reports),
                'total_competitor_urls_analyzed': sum(r['competitor_urls_count'] for r in topic_reports)
            }
        })

    except Exception as e:
        print(f"❌ Error en análisis de topic authority: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# SEMANTIC FILTER - Filtrar por tema usando similaridad de embeddings
# ============================================================================

@app.route('/semantic_filter', methods=['POST'])
def semantic_filter():
    """
    Filtra los keywords ya analizados por similitud semántica con una query.
    Usa los embeddings ya calculados de gap_analysis_state_v2.
    """
    global gap_analysis_state_v2

    # Función para convertir tipos numpy a nativos de Python
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj

    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        threshold = float(data.get('threshold', 0.5))
        candidates = data.get('candidates', None)  # Lista opcional de textos candidatos
        top_k = int(data.get('top_k', 50))

        if not query:
            return jsonify({'error': 'Se requiere una query'}), 400

        # Modo candidatos: buscar similitud con lista proporcionada
        if candidates and isinstance(candidates, list) and len(candidates) > 0:
            print(f"\n{'='*70}")
            print(f"🔎 FILTRO SEMÁNTICO (modo candidatos)")
            print(f"{'='*70}")
            print(f"   Query: '{query}'")
            print(f"   Candidatos: {len(candidates)}")
            print(f"   Umbral: {threshold*100:.0f}%")

            from sklearn.metrics.pairwise import cosine_similarity

            # Vectorizar query y candidatos
            query_emb = model_768.encode([query])[0]
            candidates_emb = encode_with_cache(candidates, model_768)

            # Calcular similitudes
            similarities = cosine_similarity([query_emb], candidates_emb)[0]

            # Filtrar y ordenar
            results = []
            for i, (text, sim) in enumerate(zip(candidates, similarities)):
                if sim >= threshold:
                    results.append({'text': text, 'similarity': float(sim), 'index': i})

            results.sort(key=lambda x: x['similarity'], reverse=True)
            results = results[:top_k]

            print(f"   Resultados: {len(results)} (umbral >= {threshold*100:.0f}%)")

            return jsonify({
                'query': query,
                'results': results,
                'total_candidates': len(candidates),
                'total_matches': len(results)
            })

        # Modo original: usar datos de análisis previo
        # Verificar que hay datos de análisis previo
        if gap_analysis_state_v2['embeddings'] is None or gap_analysis_state_v2['all_keywords_list'] is None:
            return jsonify({'error': 'Primero ejecuta el análisis de gaps V2 o proporciona candidatos'}), 400

        print(f"\n{'='*70}")
        print(f"🔎 FILTRO SEMÁNTICO")
        print(f"{'='*70}")
        print(f"   Query: '{query}'")
        print(f"   Umbral de similitud: {threshold*100:.0f}%")

        # Vectorizar la query del usuario
        query_embedding = model_768.encode([query])[0]

        # Obtener embeddings y keywords almacenados
        stored_embeddings = gap_analysis_state_v2['embeddings']
        all_keywords_list = gap_analysis_state_v2['all_keywords_list']  # Lista de strings
        stored_keywords_data = gap_analysis_state_v2['keywords']  # Lista de diccionarios con datos
        gaps_data = gap_analysis_state_v2.get('gaps_data', [])

        print(f"   Keywords almacenados: {len(all_keywords_list)}")
        print(f"   Embeddings: {stored_embeddings.shape}")
        print(f"   Datos de keywords: {len(stored_keywords_data)}")

        # Crear diccionario de keyword -> datos para búsqueda rápida
        kw_to_data = {}
        for kw_data in stored_keywords_data:
            kw = kw_data.get('keyword', '')
            if kw:
                kw_to_data[kw] = kw_data

        # También incluir gaps_data
        for gap in gaps_data:
            kw = gap.get('keyword', '')
            if kw and kw not in kw_to_data:
                kw_to_data[kw] = gap

        print(f"   Mapa keyword->datos: {len(kw_to_data)} entradas")

        # Calcular similitudes con todos los keywords
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], stored_embeddings)[0]

        # Filtrar por umbral
        results = []

        for i, kw in enumerate(all_keywords_list):
            sim = float(similarities[i])
            if sim >= threshold:
                # Buscar datos del keyword
                kw_data = kw_to_data.get(kw, {})

                result = {
                    'keyword': kw,
                    'similarity': sim,
                    'type': 'own' if kw_data.get('source') == 'own' else 'gap',
                    'Volume': kw_data.get('Volume', kw_data.get('Volumen', kw_data.get('Search Volume', 0))),
                    'Position': kw_data.get('Position', kw_data.get('Posición', None)),
                    'KD': kw_data.get('KD', kw_data.get('Keyword Difficulty', None)),
                    'intent_type': kw_data.get('intent_type', ''),
                    'URL': kw_data.get('URL', '')
                }
                results.append(result)

        # Ordenar por similitud descendente
        results.sort(key=lambda x: x['similarity'], reverse=True)

        # Calcular estadísticas
        gaps_count = sum(1 for r in results if r['type'] == 'gap')
        own_count = sum(1 for r in results if r['type'] == 'own')
        total_volume = sum(r.get('Volume', 0) or 0 for r in results)

        print(f"\n   ✅ Resultados encontrados: {len(results)}")
        print(f"      - Gaps: {gaps_count}")
        print(f"      - Propios: {own_count}")
        print(f"      - Volumen total: {total_volume:,}")

        if results:
            print(f"\n   🔝 Top 5 más similares:")
            for r in results[:5]:
                print(f"      {r['similarity']*100:.0f}% | {r['keyword'][:50]} [{r['type']}]")

        print(f"\n{'='*70}")

        return jsonify({
            'success': True,
            'query': query,
            'threshold': threshold,
            'total_matches': len(results),
            'gaps_count': gaps_count,
            'own_count': own_count,
            'total_volume': total_volume,
            'results': convert_to_native(results[:100])  # Limitar a 100 resultados
        })

    except Exception as e:
        print(f"❌ Error en filtro semántico: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# RENAME CLUSTERS WITH LLM - Renombrar clusters usando Ollama
# ============================================================================

@app.route('/rename_clusters_llm', methods=['POST'])
def rename_clusters_llm():
    """
    Renombra clusters usando Ollama LLM (llama3.1:8b).
    Recibe los clusters actuales y genera nombres más descriptivos.
    """
    try:
        import ollama

        data = request.json
        clusters = data.get('clusters', [])
        ollama_model = data.get('model', 'llama3.1:8b')

        if not clusters:
            return jsonify({'error': 'No se proporcionaron clusters'}), 400

        print(f"\n🤖 Renombrando {len(clusters)} clusters con Ollama ({ollama_model})...")

        # Verificar que Ollama está disponible
        try:
            ollama.list()
        except Exception as e:
            return jsonify({
                'error': f'Ollama no está disponible: {str(e)}',
                'hint': 'Asegúrate de que Ollama esté corriendo (ollama serve)'
            }), 503

        renamed_clusters = []

        for cluster in clusters:
            cluster_id = cluster.get('cluster_id')
            pages = cluster.get('pages', [])
            old_name = cluster.get('cluster_name', '')
            keywords = cluster.get('cluster_keywords', '')

            # Saltar outliers
            if cluster_id == -1:
                renamed_clusters.append({
                    'cluster_id': cluster_id,
                    'old_name': old_name,
                    'new_name': 'Contenido sin Clasificar',
                    'confidence': 1.0
                })
                continue

            # Preparar contexto para el LLM
            sample_titles = [p.get('title', '') for p in pages[:8] if p.get('title')]
            sample_h1s = [p.get('h1', '') for p in pages[:8] if p.get('h1')]

            if not sample_titles and not sample_h1s:
                renamed_clusters.append({
                    'cluster_id': cluster_id,
                    'old_name': old_name,
                    'new_name': old_name,
                    'confidence': 0.0
                })
                continue

            # Crear prompt optimizado
            context_titles = '\n'.join(f"  - {t}" for t in sample_titles[:6])
            # Filtrar H1s que no sean igual al primer título
            first_title = sample_titles[0] if sample_titles else ''
            context_h1s = '\n'.join(f"  - {h}" for h in sample_h1s[:6] if h != first_title)

            prompt = f"""Eres un experto en SEO y categorización de contenido. Analiza este grupo de páginas web y genera un nombre de categoría descriptivo.

TÍTULOS DE PÁGINAS:
{context_titles}

{"ENCABEZADOS H1:" + chr(10) + context_h1s if context_h1s else ""}

{"PALABRAS CLAVE: " + keywords if keywords else ""}

REGLAS ESTRICTAS:
1. Responde SOLO con el nombre de la categoría (máximo 5 palabras)
2. Usa español natural y profesional
3. Identifica el tema principal común a todas las páginas
4. NO uses dos puntos (:) ni guiones (-)
5. NO incluyas explicaciones ni texto adicional
6. Evita nombres genéricos como "Información general"

NOMBRE DE LA CATEGORÍA:"""

            try:
                response = ollama.chat(
                    model=ollama_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={
                        'temperature': 0.3,
                        'num_predict': 30
                    }
                )
                new_name = response['message']['content'].strip()

                # Limpiar el nombre
                new_name = new_name.replace('"', '').replace("'", "").replace(':', ' ').replace('-', ' ').strip()
                new_name = ' '.join(new_name.split())  # Normalizar espacios

                # Si es muy largo, truncar
                words = new_name.split()
                if len(words) > 6:
                    new_name = ' '.join(words[:6])

                # Capitalizar primera letra
                if new_name:
                    new_name = new_name[0].upper() + new_name[1:] if len(new_name) > 1 else new_name.upper()

                renamed_clusters.append({
                    'cluster_id': cluster_id,
                    'old_name': old_name,
                    'new_name': new_name,
                    'confidence': 0.9
                })

                print(f"   Cluster {cluster_id}: '{old_name}' → '{new_name}'")

            except Exception as e:
                print(f"   ⚠️ Error en cluster {cluster_id}: {e}")
                renamed_clusters.append({
                    'cluster_id': cluster_id,
                    'old_name': old_name,
                    'new_name': old_name,
                    'confidence': 0.0,
                    'error': str(e)
                })

        print(f"✅ {len(renamed_clusters)} clusters procesados")

        return jsonify({
            'renamed_clusters': renamed_clusters,
            'total': len(renamed_clusters),
            'model': ollama_model
        })

    except ImportError:
        return jsonify({
            'error': 'El módulo ollama no está instalado',
            'hint': 'pip install ollama'
        }), 500
    except Exception as e:
        print(f"❌ Error renombrando clusters: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# CHECK OLLAMA STATUS - Verificar estado de Ollama
# ============================================================================

@app.route('/ollama_status', methods=['GET'])
def ollama_status():
    """Verifica si Ollama está disponible y qué modelos tiene."""
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            data = response.json()
            model_names = [m.get('name', m.get('model', '')) for m in data.get('models', [])]

            return jsonify({
                'available': True,
                'models': model_names,
                'recommended': 'llama3.1:8b' if 'llama3.1:8b' in model_names else model_names[0] if model_names else None
            })
        else:
            return jsonify({
                'available': False,
                'error': f'Ollama respondió con código {response.status_code}',
                'hint': 'Asegúrate de que Ollama esté corriendo: ollama serve'
            })
    except Exception as e:
        return jsonify({
            'available': False,
            'error': str(e),
            'hint': 'Asegúrate de que Ollama esté corriendo: ollama serve'
        })


# ============================================================================
# SEMANTIC SEARCH PAGES - Búsqueda semántica en páginas de Supabase
# ============================================================================

@app.route('/semantic_search_pages', methods=['POST'])
def semantic_search_pages():
    """
    Búsqueda semántica en páginas de Supabase.
    Vectoriza la query y busca las páginas más similares usando UMAP state.
    """
    global umap_state

    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = int(data.get('top_k', 10))
        url_filter = data.get('url_filter', '').strip()

        if not query:
            return jsonify({'error': 'Se requiere una query'}), 400

        print(f"\n🔍 Búsqueda semántica: '{query}'")

        # Conectar a Supabase para obtener páginas
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Query para obtener páginas con centroides
        if url_filter:
            cursor.execute("""
                SELECT p.id, p.url, p.title, p.meta_description, pc.centroid_embedding
                FROM page_centroids pc
                JOIN pages p ON pc.page_id = p.id
                WHERE p.status = 'active' AND p.url ILIKE %s
            """, (f'%{url_filter}%',))
        else:
            cursor.execute("""
                SELECT p.id, p.url, p.title, p.meta_description, pc.centroid_embedding
                FROM page_centroids pc
                JOIN pages p ON pc.page_id = p.id
                WHERE p.status = 'active'
            """)

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return jsonify({'results': [], 'message': 'No hay páginas en la base de datos'})

        # Extraer embeddings
        embeddings = []
        for row in rows:
            vec = row['centroid_embedding']
            if isinstance(vec, str):
                vec = vec.strip('[]').split(',')
                vec = [float(x) for x in vec]
            embeddings.append(vec)

        embeddings = np.array(embeddings)

        # Verificar dimensiones y re-vectorizar si es necesario
        if embeddings.shape[1] != 768:
            print(f"   Re-vectorizando páginas (dimensión actual: {embeddings.shape[1]})")
            texts = [f"{r['title']} {r['meta_description'] or ''}" for r in rows]
            embeddings = model_768.encode(texts, show_progress_bar=False)

        # Vectorizar query
        query_embedding = model_768.encode([query], show_progress_bar=False)[0]

        # Calcular similitud coseno
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], embeddings)[0]

        # Ordenar por similitud
        sorted_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in sorted_indices:
            row = rows[idx]
            results.append({
                'url': row['url'],
                'title': row['title'] or 'Sin título',
                'description': row['meta_description'] or '',
                'similarity': float(similarities[idx])
            })

        print(f"   Encontrados {len(results)} resultados")
        return jsonify({'results': results, 'query': query})

    except Exception as e:
        print(f"❌ Error en búsqueda semántica: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# DETECT CANNIBALIZATION - Detectar canibalización de contenido
# ============================================================================

@app.route('/detect_cannibalization', methods=['POST'])
def detect_cannibalization():
    """
    Detecta páginas con posible canibalización de contenido.
    Busca páginas muy similares semánticamente que podrían competir entre sí.
    """
    try:
        data = request.get_json()
        threshold = float(data.get('threshold', 0.85))
        url_filter = data.get('url_filter', '').strip()
        max_pairs = int(data.get('max_pairs', 50))

        print(f"\n🔎 Detectando canibalización (umbral: {threshold*100:.0f}%)")

        # Conectar a Supabase
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        if url_filter:
            cursor.execute("""
                SELECT p.id, p.url, p.title, p.meta_description, pc.centroid_embedding
                FROM page_centroids pc
                JOIN pages p ON pc.page_id = p.id
                WHERE p.status = 'active' AND p.url ILIKE %s
            """, (f'%{url_filter}%',))
        else:
            cursor.execute("""
                SELECT p.id, p.url, p.title, p.meta_description, pc.centroid_embedding
                FROM page_centroids pc
                JOIN pages p ON pc.page_id = p.id
                WHERE p.status = 'active'
            """)

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if len(rows) < 2:
            return jsonify({'pairs': [], 'message': 'Se necesitan al menos 2 páginas'})

        # Extraer embeddings
        embeddings = []
        for row in rows:
            vec = row['centroid_embedding']
            if isinstance(vec, str):
                vec = vec.strip('[]').split(',')
                vec = [float(x) for x in vec]
            embeddings.append(vec)

        embeddings = np.array(embeddings)

        # Re-vectorizar si es necesario
        if embeddings.shape[1] != 768:
            texts = [f"{r['title']} {r['meta_description'] or ''}" for r in rows]
            embeddings = model_768.encode(texts, show_progress_bar=True)

        # Calcular matriz de similitud
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)

        # Encontrar pares sobre el umbral
        pairs = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                sim = sim_matrix[i][j]
                if sim >= threshold:
                    # Determinar severidad
                    if sim >= 0.95:
                        severity = 'critical'
                    elif sim >= 0.90:
                        severity = 'high'
                    else:
                        severity = 'medium'

                    pairs.append({
                        'page1': {
                            'url': rows[i]['url'],
                            'title': rows[i]['title'] or 'Sin título'
                        },
                        'page2': {
                            'url': rows[j]['url'],
                            'title': rows[j]['title'] or 'Sin título'
                        },
                        'similarity': float(sim),
                        'severity': severity
                    })

        # Ordenar por similitud descendente
        pairs.sort(key=lambda x: x['similarity'], reverse=True)
        pairs = pairs[:max_pairs]

        print(f"   Encontrados {len(pairs)} pares con posible canibalización")

        return jsonify({
            'pairs': pairs,
            'total_pages': len(rows),
            'threshold': threshold
        })

    except Exception as e:
        print(f"❌ Error detectando canibalización: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# TOPIC COVERAGE ANALYSIS - Análisis de cobertura temática
# ============================================================================

@app.route('/analyze_topic_coverage', methods=['POST'])
def analyze_topic_coverage():
    """
    Analiza la cobertura de topics específicos en el sitio.
    Recibe una lista de topics y evalúa cuánto contenido existe para cada uno.
    """
    try:
        data = request.get_json()
        topics = data.get('topics', [])
        url_filter = data.get('url_filter', '').strip()

        if not topics:
            return jsonify({'error': 'Se requiere una lista de topics'}), 400

        print(f"\n📊 Analizando cobertura de {len(topics)} topics")

        # Conectar a Supabase
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        if url_filter:
            cursor.execute("""
                SELECT p.id, p.url, p.title, p.meta_description, pc.centroid_embedding
                FROM page_centroids pc
                JOIN pages p ON pc.page_id = p.id
                WHERE p.status = 'active' AND p.url ILIKE %s
            """, (f'%{url_filter}%',))
        else:
            cursor.execute("""
                SELECT p.id, p.url, p.title, p.meta_description, pc.centroid_embedding
                FROM page_centroids pc
                JOIN pages p ON pc.page_id = p.id
                WHERE p.status = 'active'
            """)

        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            return jsonify({'coverage': [], 'message': 'No hay páginas'})

        # Extraer embeddings
        embeddings = []
        for row in rows:
            vec = row['centroid_embedding']
            if isinstance(vec, str):
                vec = vec.strip('[]').split(',')
                vec = [float(x) for x in vec]
            embeddings.append(vec)

        embeddings = np.array(embeddings)

        # Re-vectorizar si es necesario
        if embeddings.shape[1] != 768:
            texts = [f"{r['title']} {r['meta_description'] or ''}" for r in rows]
            embeddings = model_768.encode(texts, show_progress_bar=True)

        # Vectorizar topics
        topic_embeddings = model_768.encode(topics, show_progress_bar=False)

        # Calcular similitud
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(topic_embeddings, embeddings)

        # Analizar cobertura por topic
        coverage = []
        for idx, topic in enumerate(topics):
            sims = sim_matrix[idx]

            # Páginas relevantes (> 0.5 similitud)
            relevant_mask = sims >= 0.5
            relevant_count = np.sum(relevant_mask)

            # Páginas muy relevantes (> 0.7)
            highly_relevant = np.sum(sims >= 0.7)

            # Mejor match
            best_idx = np.argmax(sims)
            best_page = rows[best_idx]

            # Score de cobertura (0-100)
            if relevant_count == 0:
                score = 0
            elif highly_relevant >= 5:
                score = min(100, 50 + highly_relevant * 5)
            else:
                score = min(50, relevant_count * 10)

            coverage.append({
                'topic': topic,
                'score': int(score),
                'relevant_pages': int(relevant_count),
                'highly_relevant': int(highly_relevant),
                'best_match': {
                    'url': best_page['url'],
                    'title': best_page['title'],
                    'similarity': float(sims[best_idx])
                },
                'status': 'good' if score >= 60 else 'medium' if score >= 30 else 'poor'
            })

        # Ordenar por score
        coverage.sort(key=lambda x: x['score'], reverse=True)

        return jsonify({
            'coverage': coverage,
            'total_pages': len(rows),
            'avg_score': sum(c['score'] for c in coverage) / len(coverage) if coverage else 0
        })

    except Exception as e:
        print(f"❌ Error en análisis de cobertura: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# TAXONOMY COVERAGE AUDIT - Auditoría de cobertura de taxonomía SEO
# ============================================================================

@app.route('/audit_taxonomy_coverage', methods=['POST'])
def audit_taxonomy_coverage():
    """
    Auditoría profesional de cobertura semántica.
    Compara una taxonomía de keywords/topics con las páginas indexadas.

    Recibe un Excel con la taxonomía y analiza:
    - Cobertura por Pillar
    - Cobertura por cluster
    - Gaps de contenido
    - Páginas que mejor cubren cada topic
    - Score de visibilidad semántica
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity

        # Verificar archivo
        if 'file' not in request.files:
            return jsonify({'error': 'Se requiere un archivo Excel'}), 400

        file = request.files['file']
        sheet_name = request.form.get('sheet_name', '1.Taxonomia')
        url_filter = request.form.get('url_filter', '').strip()

        print(f"\n{'='*70}")
        print("📊 AUDITORÍA DE COBERTURA DE TAXONOMÍA SEO")
        print(f"{'='*70}")
        print(f"   Archivo: {file.filename}")
        print(f"   Hoja: {sheet_name}")
        if url_filter:
            print(f"   Filtro URL: {url_filter}")

        # Leer Excel
        df = pd.read_excel(file, sheet_name=sheet_name, header=None)

        # Parsear taxonomía
        print("\n🔍 Parseando taxonomía...")
        pillars = []
        current_pillar = None
        current_pillar_clusters = []

        for idx, row in df.iterrows():
            col1 = str(row[1]) if pd.notna(row[1]) else ''
            col2 = str(row[2]) if pd.notna(row[2]) else ''
            col3 = str(row[3]) if pd.notna(row[3]) else ''
            col4 = str(row[4]) if pd.notna(row[4]) else ''

            if 'PILLAR' in col1.upper():
                # Guardar pillar anterior
                if current_pillar and current_pillar_clusters:
                    pillars.append({
                        'name': current_pillar,
                        'clusters': current_pillar_clusters.copy()
                    })
                current_pillar = col1
                current_pillar_clusters = []
            elif col2 and 'Vector' not in col2 and 'nan' not in col2.lower() and current_pillar and idx > 2:
                # Es un cluster real
                current_pillar_clusters.append({
                    'name': col2,
                    'intention': col3,
                    'keywords': col4
                })

        # Guardar último pillar
        if current_pillar and current_pillar_clusters:
            pillars.append({
                'name': current_pillar,
                'clusters': current_pillar_clusters
            })

        total_clusters = sum(len(p['clusters']) for p in pillars)
        print(f"✅ {len(pillars)} Pillars, {total_clusters} clusters encontrados")

        # Cargar páginas de Supabase
        print("\n📥 Cargando páginas desde Supabase...")
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        if url_filter:
            cursor.execute("""
                SELECT p.id, p.url, p.title, p.meta_description, p.h1, pc.centroid_embedding
                FROM page_centroids pc
                JOIN pages p ON pc.page_id = p.id
                WHERE p.status = 'active' AND p.url ILIKE %s
            """, (f'%{url_filter}%',))
        else:
            cursor.execute("""
                SELECT p.id, p.url, p.title, p.meta_description, p.h1, pc.centroid_embedding
                FROM page_centroids pc
                JOIN pages p ON pc.page_id = p.id
                WHERE p.status = 'active'
            """)

        pages = cursor.fetchall()
        cursor.close()
        conn.close()

        if not pages:
            return jsonify({'error': 'No se encontraron páginas en la base de datos'}), 404

        print(f"✅ {len(pages)} páginas cargadas")

        # Extraer embeddings de páginas
        page_embeddings = []
        for page in pages:
            vec = page['centroid_embedding']
            if isinstance(vec, str):
                vec = vec.strip('[]').split(',')
                vec = [float(x) for x in vec]
            page_embeddings.append(vec)

        page_embeddings = np.array(page_embeddings)

        # Re-vectorizar si dimensiones no coinciden
        if page_embeddings.shape[1] != 768:
            print("   Re-vectorizando páginas con modelo 768D...")
            texts = [f"{p['title']} {p['meta_description'] or ''} {p['h1'] or ''}" for p in pages]
            page_embeddings = model_768.encode(texts, show_progress_bar=True)

        # =====================================================================
        # ANÁLISIS SEO SEMÁNTICO PROFESIONAL
        # =====================================================================

        print("\n🔄 Analizando cobertura semántica profesional...")
        pillar_results = []
        all_gaps = []
        all_cannibalization = []  # Detección de canibalización
        all_recommendations = []  # Recomendaciones de contenido

        # Función helper para clasificar intención de búsqueda
        def classify_search_intent(text, intention_hint=''):
            """Clasifica la intención de búsqueda basándose en el texto"""
            text_lower = (text + ' ' + intention_hint).lower()

            # Indicadores transaccionales/comerciales
            transactional_signals = ['comprar', 'precio', 'oferta', 'descuento', 'tienda',
                                    'barato', 'mejor', 'comparar', 'envío', 'pedido', 'online',
                                    'promoción', 'rebaja', 'outlet']

            # Indicadores navegacionales
            navigational_signals = ['horario', 'teléfono', 'dirección', 'contacto', 'ubicación',
                                   'cerca', 'sucursal', 'tiendas en', 'cómo llegar', 'mapa']

            # Indicadores informacionales
            informational_signals = ['qué es', 'cómo', 'por qué', 'guía', 'tutorial',
                                    'receta', 'consejos', 'beneficios', 'tipos de', 'diferencia',
                                    'ingredientes', 'propiedades', 'historia']

            # Indicadores comerciales-investigacionales
            commercial_signals = ['mejor', 'top', 'vs', 'comparativa', 'alternativas',
                                 'opiniones', 'review', 'análisis', 'vale la pena']

            transactional_count = sum(1 for s in transactional_signals if s in text_lower)
            navigational_count = sum(1 for s in navigational_signals if s in text_lower)
            informational_count = sum(1 for s in informational_signals if s in text_lower)
            commercial_count = sum(1 for s in commercial_signals if s in text_lower)

            counts = {
                'transactional': transactional_count,
                'navigational': navigational_count,
                'informational': informational_count,
                'commercial_investigation': commercial_count
            }

            max_intent = max(counts, key=counts.get)

            if counts[max_intent] == 0:
                # Si no hay señales claras, inferir por tipo de keywords
                if any(word in text_lower for word in ['producto', 'marca', 'compra']):
                    return 'transactional'
                return 'informational'

            return max_intent

        # Función para generar recomendación de contenido
        def generate_content_recommendation(cluster_name, intention, coverage_score, best_similarity, top_pages):
            """Genera una recomendación profesional de contenido"""
            intent_type = classify_search_intent(cluster_name, intention)

            if coverage_score < 10:  # GAP total
                if intent_type == 'transactional':
                    return {
                        'priority': 'CRÍTICA',
                        'action': 'Crear página de producto/categoría',
                        'type': 'Landing page comercial',
                        'suggestion': f'Crear página optimizada para "{cluster_name}" con CTA claro, precios y opciones de compra.',
                        'content_elements': ['H1 con keyword principal', 'Lista de productos', 'Precios visibles', 'CTA de compra', 'Opiniones/valoraciones']
                    }
                elif intent_type == 'informational':
                    return {
                        'priority': 'ALTA',
                        'action': 'Crear contenido informativo',
                        'type': 'Artículo/Guía',
                        'suggestion': f'Crear artículo completo sobre "{cluster_name}" respondiendo las principales dudas del usuario.',
                        'content_elements': ['H1 optimizado', 'Índice de contenidos', 'FAQ estructuradas', 'Imágenes explicativas', 'Internal linking a productos']
                    }
                elif intent_type == 'navigational':
                    return {
                        'priority': 'ALTA',
                        'action': 'Crear página de localización/contacto',
                        'type': 'Página de servicio',
                        'suggestion': f'Crear página específica para "{cluster_name}" con información de contacto y ubicación.',
                        'content_elements': ['Mapa integrado', 'Horarios', 'Teléfono y dirección', 'Formulario de contacto']
                    }
                else:
                    return {
                        'priority': 'ALTA',
                        'action': 'Crear contenido comparativo',
                        'type': 'Guía de compra',
                        'suggestion': f'Crear comparativa/guía de compra para "{cluster_name}".',
                        'content_elements': ['Tabla comparativa', 'Pros y contras', 'Recomendación final', 'Precios']
                    }

            elif coverage_score < 30:  # Cobertura débil
                if top_pages and top_pages[0]['similarity'] < 0.6:
                    return {
                        'priority': 'ALTA',
                        'action': 'Optimizar contenido existente',
                        'type': 'Mejora de página existente',
                        'suggestion': f'La página "{top_pages[0]["title"]}" cubre parcialmente este topic. Enriquecer con más contenido específico sobre "{cluster_name}".',
                        'content_elements': ['Añadir sección específica', 'Expandir H2/H3', 'Añadir FAQs', 'Mejorar internal linking']
                    }
                else:
                    return {
                        'priority': 'MEDIA',
                        'action': 'Crear contenido complementario',
                        'type': 'Página de soporte',
                        'suggestion': f'Crear página específica que profundice en "{cluster_name}" y enlace a páginas existentes.',
                        'content_elements': ['Contenido long-form', 'Internal links bidireccionales', 'Schema markup']
                    }

            elif coverage_score < 50:  # Cobertura parcial
                return {
                    'priority': 'MEDIA',
                    'action': 'Mejorar optimización on-page',
                    'type': 'Optimización',
                    'suggestion': f'Mejorar la optimización de las páginas existentes para "{cluster_name}". Revisar titles, metas y estructura de contenido.',
                    'content_elements': ['Revisar title tag', 'Mejorar meta description', 'Añadir schema', 'Expandir contenido']
                }

            else:  # Buena cobertura
                return {
                    'priority': 'BAJA',
                    'action': 'Mantener y reforzar',
                    'type': 'Mantenimiento',
                    'suggestion': f'La cobertura de "{cluster_name}" es buena. Reforzar con actualizaciones periódicas y backlinks.',
                    'content_elements': ['Actualizar contenido', 'Añadir contenido fresco', 'Linkbuilding']
                }

        for pillar in pillars:
            print(f"\n   Pillar: {pillar['name'][:50]}...")
            cluster_results = []
            pillar_pages_used = {}  # Para detectar canibalización

            for cluster in pillar['clusters']:
                # Vectorizar el cluster (nombre + keywords)
                cluster_text = f"{cluster['name']} {cluster['keywords']}"
                cluster_embedding = model_768.encode([cluster_text], show_progress_bar=False)[0]

                # Calcular similitud con todas las páginas
                similarities = cosine_similarity([cluster_embedding], page_embeddings)[0]

                # Encontrar mejores matches
                sorted_indices = np.argsort(similarities)[::-1]

                # Métricas avanzadas
                best_similarity = float(similarities[sorted_indices[0]])
                highly_relevant = int(np.sum(similarities >= 0.7))
                relevant = int(np.sum(similarities >= 0.5))
                moderately_relevant = int(np.sum(similarities >= 0.4))

                # Topic Authority Score (más sofisticado)
                # Considera: best match, densidad de páginas relevantes, y distribución
                topic_authority = 0
                if best_similarity >= 0.75:
                    topic_authority += 40
                elif best_similarity >= 0.6:
                    topic_authority += 25
                elif best_similarity >= 0.5:
                    topic_authority += 10

                # Bonus por "cobertura en profundidad" (múltiples páginas)
                topic_authority += min(30, highly_relevant * 10)
                topic_authority += min(20, relevant * 3)

                # Bonus por consistencia (si las top páginas tienen similitudes cercanas)
                top_sims = [similarities[sorted_indices[i]] for i in range(min(5, len(sorted_indices)))]
                sim_variance = np.var(top_sims) if len(top_sims) > 1 else 0
                if sim_variance < 0.01 and np.mean(top_sims) > 0.6:
                    topic_authority += 10  # Cobertura consistente

                topic_authority = min(100, topic_authority)

                # Score de cobertura mejorado
                if best_similarity >= 0.8:
                    base_score = 70
                elif best_similarity >= 0.7:
                    base_score = 50
                elif best_similarity >= 0.6:
                    base_score = 30
                elif best_similarity >= 0.5:
                    base_score = 15
                else:
                    base_score = max(0, int(best_similarity * 30))

                bonus = min(30, highly_relevant * 5 + relevant * 2)
                coverage_score = min(100, base_score + bonus)

                # Determinar estado
                if coverage_score >= 70:
                    status = 'excellent'
                elif coverage_score >= 50:
                    status = 'good'
                elif coverage_score >= 30:
                    status = 'partial'
                elif coverage_score >= 10:
                    status = 'weak'
                else:
                    status = 'gap'

                # Top 5 páginas más relevantes (más detallado)
                top_pages = []
                for i in range(min(5, len(sorted_indices))):
                    idx = sorted_indices[i]
                    page_url = pages[idx]['url']
                    sim = float(similarities[idx])

                    # Solo incluir si similitud > 0.3
                    if sim < 0.3:
                        continue

                    top_pages.append({
                        'url': page_url,
                        'title': pages[idx]['title'] or 'Sin título',
                        'h1': pages[idx].get('h1', ''),
                        'similarity': sim,
                        'relevance_level': 'high' if sim >= 0.7 else 'medium' if sim >= 0.5 else 'low'
                    })

                    # Detectar posible canibalización
                    if sim >= 0.5:
                        if page_url not in pillar_pages_used:
                            pillar_pages_used[page_url] = []
                        pillar_pages_used[page_url].append({
                            'cluster': cluster['name'],
                            'similarity': sim
                        })

                # Clasificar intención de búsqueda
                search_intent = classify_search_intent(cluster['name'], cluster.get('intention', ''))

                # Generar recomendación
                recommendation = generate_content_recommendation(
                    cluster['name'],
                    cluster.get('intention', ''),
                    coverage_score,
                    best_similarity,
                    top_pages
                )

                cluster_result = {
                    'name': cluster['name'],
                    'intention': cluster.get('intention', ''),
                    'search_intent': search_intent,  # Nuevo: tipo de intención
                    'keywords_sample': cluster['keywords'][:300] + '...' if len(cluster['keywords']) > 300 else cluster['keywords'],
                    'coverage_score': coverage_score,
                    'topic_authority': topic_authority,  # Nuevo: Topic Authority Score
                    'status': status,
                    'best_similarity': best_similarity,
                    'highly_relevant_pages': highly_relevant,
                    'relevant_pages': relevant,
                    'moderately_relevant_pages': moderately_relevant,  # Nuevo
                    'top_pages': top_pages,
                    'recommendation': recommendation  # Nuevo: recomendación específica
                }

                cluster_results.append(cluster_result)

                # Si es un gap o weak, añadirlo a la lista con recomendación
                if status in ['gap', 'weak']:
                    all_gaps.append({
                        'pillar': pillar['name'],
                        'cluster': cluster['name'],
                        'intention': cluster.get('intention', ''),
                        'search_intent': search_intent,
                        'keywords': cluster['keywords'][:500],
                        'coverage_score': coverage_score,
                        'best_similarity': best_similarity,
                        'recommendation': recommendation
                    })

                # Añadir todas las recomendaciones de alta prioridad
                if recommendation['priority'] in ['CRÍTICA', 'ALTA']:
                    all_recommendations.append({
                        'pillar': pillar['name'],
                        'cluster': cluster['name'],
                        **recommendation
                    })

            # Detectar canibalización en este pillar
            for page_url, clusters_targeting in pillar_pages_used.items():
                if len(clusters_targeting) > 1:
                    all_cannibalization.append({
                        'pillar': pillar['name'],
                        'url': page_url,
                        'competing_clusters': clusters_targeting,
                        'severity': 'high' if len(clusters_targeting) >= 3 else 'medium'
                    })

            # Calcular métricas del pillar
            pillar_scores = [c['coverage_score'] for c in cluster_results]
            pillar_authority_scores = [c['topic_authority'] for c in cluster_results]
            pillar_avg = sum(pillar_scores) / len(pillar_scores) if pillar_scores else 0
            pillar_authority_avg = sum(pillar_authority_scores) / len(pillar_authority_scores) if pillar_authority_scores else 0

            excellent = len([c for c in cluster_results if c['status'] == 'excellent'])
            good = len([c for c in cluster_results if c['status'] == 'good'])
            partial = len([c for c in cluster_results if c['status'] == 'partial'])
            weak = len([c for c in cluster_results if c['status'] == 'weak'])
            gaps = len([c for c in cluster_results if c['status'] == 'gap'])

            # Contar por tipo de intención
            intent_counts = {}
            for c in cluster_results:
                intent = c['search_intent']
                intent_counts[intent] = intent_counts.get(intent, 0) + 1

            pillar_results.append({
                'name': pillar['name'],
                'avg_coverage': round(pillar_avg, 1),
                'avg_topic_authority': round(pillar_authority_avg, 1),
                'total_clusters': len(cluster_results),
                'excellent': excellent,
                'good': good,
                'partial': partial,
                'weak': weak,
                'gaps': gaps,
                'intent_distribution': intent_counts,
                'clusters': sorted(cluster_results, key=lambda x: x['coverage_score'])
            })

        # Ordenar gaps por prioridad (menor score = mayor prioridad)
        all_gaps.sort(key=lambda x: x['coverage_score'])

        # Ordenar recomendaciones por prioridad
        priority_order = {'CRÍTICA': 0, 'ALTA': 1, 'MEDIA': 2, 'BAJA': 3}
        all_recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))

        # Calcular métricas globales
        all_scores = []
        all_authority = []
        for p in pillar_results:
            for c in p['clusters']:
                all_scores.append(c['coverage_score'])
                all_authority.append(c['topic_authority'])

        global_avg = sum(all_scores) / len(all_scores) if all_scores else 0
        global_authority = sum(all_authority) / len(all_authority) if all_authority else 0

        print(f"\n✅ Auditoría SEO Semántica completada")
        print(f"   Score global de cobertura: {global_avg:.1f}%")
        print(f"   Topic Authority promedio: {global_authority:.1f}")
        print(f"   Gaps críticos detectados: {len(all_gaps)}")
        print(f"   Posibles canibalizaciones: {len(all_cannibalization)}")
        print(f"   Recomendaciones prioritarias: {len(all_recommendations)}")

        return jsonify({
            'summary': {
                'total_pillars': len(pillar_results),
                'total_clusters': total_clusters,
                'total_pages': len(pages),
                'global_coverage_score': round(global_avg, 1),
                'global_topic_authority': round(global_authority, 1),
                'total_gaps': len(all_gaps),
                'total_cannibalization_issues': len(all_cannibalization),
                'high_priority_recommendations': len([r for r in all_recommendations if r['priority'] in ['CRÍTICA', 'ALTA']])
            },
            'pillars': pillar_results,
            'priority_gaps': all_gaps[:20],
            'all_gaps': all_gaps,
            'cannibalization_issues': all_cannibalization[:15],
            'priority_recommendations': all_recommendations[:25]
        })

    except Exception as e:
        print(f"❌ Error en auditoría: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# EXPORT AUDIT REPORT - Exportar reporte de auditoría
# ============================================================================

@app.route('/export_audit_report', methods=['POST'])
def export_audit_report():
    """
    Genera un reporte Excel de la auditoría de cobertura.
    """
    try:
        data = request.json

        if not data or 'pillars' not in data:
            return jsonify({'error': 'Se requieren datos de auditoría'}), 400

        # Crear DataFrame para el reporte
        rows = []
        for pillar in data['pillars']:
            for cluster in pillar['clusters']:
                rows.append({
                    'Pillar': pillar['name'],
                    'Cluster': cluster['name'],
                    'Intención': cluster['intention'],
                    'Score Cobertura': cluster['coverage_score'],
                    'Estado': cluster['status'],
                    'Mejor Similitud': round(cluster['best_similarity'] * 100, 1),
                    'Páginas Muy Relevantes': cluster['highly_relevant_pages'],
                    'Páginas Relevantes': cluster['relevant_pages'],
                    'Mejor Página': cluster['top_pages'][0]['url'] if cluster['top_pages'] else '',
                    'Keywords': cluster.get('keywords_sample', '')
                })

        df = pd.DataFrame(rows)

        # Guardar a archivo temporal
        import tempfile
        import os

        temp_file = os.path.join(tempfile.gettempdir(), 'auditoria_seo.xlsx')
        df.to_excel(temp_file, index=False, sheet_name='Auditoría Cobertura')

        return send_from_directory(
            tempfile.gettempdir(),
            'auditoria_seo.xlsx',
            as_attachment=True,
            download_name=f'auditoria_cobertura_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )

    except Exception as e:
        print(f"❌ Error exportando reporte: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 SERVIDOR DE CLUSTERING SEMÁNTICO")
    print("=" * 70)
    print("\n📍 Servidor iniciando en http://localhost:5000")
    print("📊 Endpoints disponibles:")
    print("   - POST /process_excel          → Procesar Excel formato Topic/Subtopic/Keyword")
    print("   - POST /process_excel_simple   → Procesar Excel formato Cluster/Keyword (SEMrush)")
    print("   - POST /cluster_keywords_multilevel → Clustering multinivel de keywords (GSC/SEMrush)")
    print("   - POST /cluster_by_url         → Clustering multinivel basado en URLs")
    print("   - POST /analyze_content_gaps   → Análisis de gaps de contenido vs competidor")
    print("   - POST /analyze_content_gaps_v2 → V2: Multi-competidor + Clustering + Scoring + Intención")
    print("   - POST /extract_url_content    → Extraer contenido de URLs (título, desc, H1-H3)")
    print("   - POST /analyze_topic_authority → Análisis de Topic Authority con contenido de URLs")
    print("   - POST /semantic_filter        → Filtrar keywords por similitud semántica con una query")
    print("   - POST /project_queries        → Proyectar consultas temáticas en mapa de gaps")
    print("   - POST /vectorize_topics       → Vectorizar temas generales personalizados")
    print("   - POST /load_page_centroids    → Cargar centroides de páginas desde Supabase")
    print("   - POST /load_chunks            → Cargar chunks individuales desde Supabase")
    print("   - POST /cluster_titles_h1      → Clustering de títulos/H1s desde Supabase")
    print("   - POST /hierarchical_clustering → Clustering jerárquico (meta-clusters + clusters)")
    print("   - POST /generate_heatmap       → Mapa de calor clusters vs topics personalizados")
    print("   - GET  /health                 → Verificar estado del servidor")
    print("\n🗄️  Conexión Supabase configurada:")
    print(f"   - Host: {SUPABASE_CONFIG['host']}:{SUPABASE_CONFIG['port']}")
    print(f"   - Database: {SUPABASE_CONFIG['database']}")
    print("\n✅ Listo para recibir requests\n")

    app.run(debug=True, port=5000, host='0.0.0.0')
