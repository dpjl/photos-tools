# Comparateur de photos — interface web

Portage web responsive de l'application de bureau PySide6 (`../app`). Mêmes
fonctionnalités, répertoires **côté serveur**, utilisable au clavier/souris sur
desktop et au tactile sur mobile.

## Architecture

- **`server/`** — backend FastAPI. Réutilise le cœur Python de l'app de bureau
  (`app/models`, `app/utils`) et la logique d'alignement LCS extraite dans
  `server/core/json_diff.py`. La config et les notes sont partagées avec l'app de
  bureau (`~/.photo_comparer_config.json`, `~/.photo_comparer_notes.json`).
- **`frontend/`** — SPA Svelte + Vite. Le viewport zoom/pan normalisé
  (`src/lib/viewport.ts`) est un port fidèle de `zoom_graphics_view.py`.

## Installation

```bash
# Backend (dans le venv de l'app)
cd ..
.venv/bin/python -m pip install -r web/requirements-web.txt

# Frontend (Node 18+)
cd web/frontend
npm install
npm run build      # produit web/frontend/dist/ servi par FastAPI
```

## Lancement

```bash
./run_web.sh          # port 8000 par défaut, écoute sur 0.0.0.0
./run_web.sh 8080
```

Ouvrir `http://<ip-serveur>:8000/` — sur desktop : grille synchronisée ;
sur mobile (< 768 px) : mode comparaison plein écran tactile.

## Développement front (hot reload)

```bash
# Terminal 1 : API
.venv/bin/python -m uvicorn web.server.main:app --port 8077
# Terminal 2 : Vite (proxifie /api vers 8077)
cd web/frontend && npm run dev
```

## Mémo gestes mobile

- **Pinch** : zoom · **glisser (zoomé)** : pan
- **Swipe horizontal (ajusté)** : version suivante/précédente au même cadrage
- **Swipe vertical (ajusté)** : groupe suivant/précédent
- **Appui maintenu** : aperçu (peek) de la version précédente
- **Double-tap** : bascule ajusté ↔ 100 %
- **★ Meilleure** / **Copier →** : sélection et export
