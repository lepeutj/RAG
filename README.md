# RAG System

Système de Retrieval-Augmented Generation (RAG) prêt pour déploiement sur VPS.
Projet de démonstration technique : architecture modulaire, providers
interchangeables (embeddings/LLM), API REST, tests, et scripts de déploiement.

## Pourquoi ce projet

Le but est de montrer une compréhension du fonctionnement interne d'un
pipeline RAG plutôt qu'un simple assemblage de librairies :

- **Chunking implémenté "from scratch"** (découpe récursive avec overlap),
  pas juste un appel à un splitter tout fait.
- **Architecture en couches avec interfaces abstraites** (`EmbeddingProvider`,
  `LLMProvider`) : on peut changer de modèle d'embedding ou de LLM en changeant
  une seule variable d'environnement, sans toucher au code métier.
- **Embeddings locaux par défaut** (sentence-transformers) : pas de coût API
  ni de dépendance réseau pour l'étape la plus fréquemment appelée, ce qui a
  du sens sur un petit VPS.
- **Tests unitaires et d'intégration**, avec mocks pour isoler les couches.
- **Deux chemins de déploiement** : Docker (recommandé) ou systemd + nginx.

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌───────────────┐
│   Documents │ ───▶ │   Chunking   │ ───▶ │   Embeddings   │
│ (pdf/md/txt)│      │ (overlap)    │      │ (local/OpenAI) │
└─────────────┘      └──────────────┘      └───────┬────────┘
                                                     ▼
                                            ┌────────────────┐
                                            │  ChromaDB       │
                                            │  (persistant)   │
                                            └───────┬────────┘
                                                     │
Question utilisateur ──▶ Embedding query ──▶ Recherche vectorielle
                                                     │
                                                     ▼
                                            ┌────────────────┐
                                            │  LLM Provider   │
                                            │ (Anthropic/     │
                                            │  OpenAI)        │
                                            └───────┬────────┘
                                                     ▼
                                              Réponse + sources
```

```
rag-system/
├── src/
│   ├── config.py              # config centralisée (pydantic-settings)
│   ├── main.py                # app FastAPI
│   ├── pipeline.py            # orchestrateur RAG (façade)
│   ├── ingestion/
│   │   ├── loader.py          # chargement txt/md/pdf -> Document
│   │   └── chunker.py         # découpage récursif + overlap -> Chunk
│   ├── embeddings/
│   │   └── embedder.py        # EmbeddingProvider (local / OpenAI)
│   ├── vectorstore/
│   │   └── chroma_store.py    # wrapper ChromaDB
│   ├── retrieval/
│   │   └── retriever.py       # query -> chunks pertinents
│   ├── generation/
│   │   └── llm.py             # LLMProvider (Anthropic / OpenAI) + prompt
│   └── api/
│       ├── routes.py          # endpoints /query /ingest /stats /health
│       └── schemas.py         # schémas Pydantic
├── scripts/ingest.py          # ingestion CLI en masse
├── tests/                     # pytest (chunker, retriever, api)
├── deploy/                    # nginx.conf, rag-system.service (systemd)
├── Dockerfile / docker-compose.yml
└── data/documents/            # dossier d'exemple à ingérer
```

## Installation locale

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # puis renseigner ANTHROPIC_API_KEY (ou OPENAI_API_KEY)
```

## Utilisation

### 1. Ingérer des documents

```bash
python scripts/ingest.py --path data/documents
```

### 2. Lancer l'API

```bash
uvicorn src.main:app --reload
```

Documentation interactive disponible sur `http://localhost:8000/docs`.

### 3. Interroger le système

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Quel est le délai pour un remboursement ?"}'
```

Réponse :
```json
{
  "answer": "Le délai pour un remboursement est de 30 jours après l'achat...",
  "sources": ["exemple_politique_remboursement.md"],
  "chunks": [...]
}
```

Ou ajouter un document via l'API :
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@mon_document.pdf"
```

## Tests

```bash
pytest -v
```

## Déploiement sur VPS

### Option A — Docker (recommandé)

```bash
git clone <votre-repo> && cd rag-system
cp .env.example .env   # renseigner les clés API
docker compose up -d --build
```

L'API écoute sur le port 8000. Mettre `deploy/nginx.conf` en reverse proxy
(avec `certbot` pour le HTTPS) pour l'exposer proprement sur un nom de domaine.

### Option B — systemd (sans Docker)

```bash
cd /opt && git clone <votre-repo> rag-system && cd rag-system
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # renseigner les clés
sudo cp deploy/rag-system.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rag-system
```

## Choix techniques et limites (à discuter en entretien)

- **ChromaDB embedded** plutôt qu'un service séparé (Qdrant/Weaviate) :
  suffisant jusqu'à quelques centaines de milliers de chunks, sans
  complexité opérationnelle additionnelle sur un petit VPS.
- **Pas de reranking ni de requête hybride (BM25 + dense)** dans cette
  version : piste d'amélioration évidente pour augmenter la précision du
  retrieval sur des corpus plus larges ou plus techniques.
- **Pas de gestion de l'historique de conversation** (chaque requête est
  indépendante) : à ajouter via une mémoire de session si besoin de RAG
  conversationnel.
- **Sécurité API minimale** (clé API statique en header) : à remplacer par
  un vrai système d'auth (JWT/OAuth) en contexte multi-utilisateurs.
