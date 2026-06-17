/**
 * Viewport normalisé partagé — port fidèle de `zoom_graphics_view.py`.
 *
 * L'état `{cx, cy, zoom}` est exprimé en coordonnées normalisées de l'image :
 *   cx, cy ∈ [0,1]  centre de la zone visible (0.5,0.5 = centre image)
 *   zoom           1.0 = ajusté à la vue (fit)
 * Deux viewers partageant le même (cx,cy,zoom) montrent la même région
 * relative, quelle que soit la résolution de chaque fichier.
 */
import { writable } from "svelte/store";

export interface Viewport {
  cx: number;
  cy: number;
  zoom: number;
}

export const ZOOM_STEP = 1.15;
export const MIN_ZOOM = 0.05;
export const MAX_ZOOM = 64.0;

export const viewport = writable<Viewport>({ cx: 0.5, cy: 0.5, zoom: 1.0 });

/** Géométrie d'une tuile de référence (dernière chargée), pour le zoom 1:1 global. */
export interface Geom {
  imgW: number;
  imgH: number;
  vw: number;
  vh: number;
}
export const refGeom = writable<Geom | null>(null);

export function resetViewport(): void {
  viewport.set({ cx: 0.5, cy: 0.5, zoom: 1.0 });
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const clampZoom = (z: number) => clamp(z, MIN_ZOOM, MAX_ZOOM);
const clampCenter = (c: number) => clamp(c, -0.5, 1.5);

export function fitScale(imgW: number, imgH: number, vw: number, vh: number): number {
  if (vw <= 0 || vh <= 0 || imgW <= 0 || imgH <= 0) return 1.0;
  return Math.min(vw / imgW, vh / imgH);
}

/** Transform CSS (origin 0 0) pour une tuile donnée. */
export function computeTransform(
  vp: Viewport,
  imgW: number,
  imgH: number,
  vw: number,
  vh: number,
): { scale: number; tx: number; ty: number } {
  const scale = fitScale(imgW, imgH, vw, vh) * vp.zoom;
  const tx = vw / 2 - vp.cx * imgW * scale;
  const ty = vh / 2 - vp.cy * imgH * scale;
  return { scale, tx, ty };
}

/** Zoom molette centré sur le curseur (port de `wheelEvent`). */
export function applyWheel(
  vp: Viewport,
  deltaY: number,
  vx: number,
  vy: number,
  vw: number,
  vh: number,
  imgW: number,
  imgH: number,
): Viewport {
  const fit = fitScale(imgW, imgH, vw, vh);
  const actual = fit * vp.zoom;
  const cursorImgX = vp.cx + (vx - vw / 2) / (actual * imgW);
  const cursorImgY = vp.cy + (vy - vh / 2) / (actual * imgH);

  const factor = deltaY < 0 ? ZOOM_STEP : 1.0 / ZOOM_STEP;
  const newZoom = clampZoom(vp.zoom * factor);
  if (newZoom === vp.zoom) return vp;

  const newActual = fit * newZoom;
  const newCx = cursorImgX - (vx - vw / 2) / (newActual * imgW);
  const newCy = cursorImgY - (vy - vh / 2) / (newActual * imgH);
  return { cx: clampCenter(newCx), cy: clampCenter(newCy), zoom: newZoom };
}

/** Déplacement (pan) en pixels écran (port de `mouseMoveEvent`). */
export function applyPan(
  vp: Viewport,
  dxPx: number,
  dyPx: number,
  vw: number,
  vh: number,
  imgW: number,
  imgH: number,
): Viewport {
  const actual = fitScale(imgW, imgH, vw, vh) * vp.zoom;
  if (actual <= 0) return vp;
  const dx = -dxPx / (actual * imgW);
  const dy = -dyPx / (actual * imgH);
  return { cx: clampCenter(vp.cx + dx), cy: clampCenter(vp.cy + dy), zoom: vp.zoom };
}

/** Zoom autour d'un point écran arbitraire (pinch tactile). */
export function applyZoomAt(
  vp: Viewport,
  factor: number,
  vx: number,
  vy: number,
  vw: number,
  vh: number,
  imgW: number,
  imgH: number,
): Viewport {
  const fit = fitScale(imgW, imgH, vw, vh);
  const actual = fit * vp.zoom;
  const cursorImgX = vp.cx + (vx - vw / 2) / (actual * imgW);
  const cursorImgY = vp.cy + (vy - vh / 2) / (actual * imgH);
  const newZoom = clampZoom(vp.zoom * factor);
  if (newZoom === vp.zoom) return vp;
  const newActual = fit * newZoom;
  const newCx = cursorImgX - (vx - vw / 2) / (newActual * imgW);
  const newCy = cursorImgY - (vy - vh / 2) / (newActual * imgH);
  return { cx: clampCenter(newCx), cy: clampCenter(newCy), zoom: newZoom };
}

export function adjustZoom(vp: Viewport, direction: number): Viewport {
  const factor = direction > 0 ? ZOOM_STEP : 1.0 / ZOOM_STEP;
  return { ...vp, zoom: clampZoom(vp.zoom * factor) };
}

/** 1:1 — 1 pixel image = 1 pixel écran, centré. */
export function zoomToActual(imgW: number, imgH: number, vw: number, vh: number): Viewport {
  const fit = fitScale(imgW, imgH, vw, vh);
  const zoom = fit > 0 ? 1.0 / fit : 1.0;
  return { cx: 0.5, cy: 0.5, zoom: clampZoom(zoom) };
}

export const FIT: Viewport = { cx: 0.5, cy: 0.5, zoom: 1.0 };
