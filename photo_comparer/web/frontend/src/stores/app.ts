/** Stores applicatifs centraux + actions. */
import { writable, get } from "svelte/store";
import {
  api,
  type SessionState,
  type GroupSummary,
  type GroupDetail,
} from "../lib/api";
import { resetViewport } from "../lib/viewport";

export const session = writable<SessionState | null>(null);
export const groups = writable<GroupSummary[]>([]);
export const currentKey = writable<string | null>(null);
export const currentDetail = writable<GroupDetail | null>(null);
export const mode = writable<"images" | "json">("images");
export const soloIndex = writable<number | null>(null);
export const filter = writable<string>("");
export const showNotes = writable<boolean>(false);
export const status = writable<string>("");

let _statusTimer: ReturnType<typeof setTimeout> | undefined;
/** Affiche un message de statut transitoire qui s'efface seul après `ms`. */
export function flashStatus(msg: string, ms = 2500): void {
  status.set(msg);
  clearTimeout(_statusTimer);
  if (msg) _statusTimer = setTimeout(() => status.set(""), ms);
}
/** Incrémenté pour forcer le rechargement des images/vignettes affichées. */
export const refreshNonce = writable<number>(0);
/** Index du répertoire d'entrée utilisé pour les vignettes (0 = premier). */
export const thumbDir = writable<number>(0);

export async function loadSession(): Promise<void> {
  const s = await api.getSession();
  session.set(s);
  if (s.group_count > 0) {
    await loadGroups();
    const gs = get(groups);
    if (gs.length > 0) await selectGroup(gs[0].key);
  } else {
    groups.set([]);
    currentKey.set(null);
    currentDetail.set(null);
  }
}

export async function loadGroups(): Promise<void> {
  const { groups: g } = await api.getGroups();
  groups.set(g);
}

export async function selectGroup(key: string): Promise<void> {
  currentKey.set(key);
  soloIndex.set(null);
  resetViewport();
  const detail = await api.getGroup(key);
  currentDetail.set(detail);
}

/** Recharge le groupe courant : re-lit le détail (nouveaux mtimes/noms) et
 *  incrémente le nonce pour forcer le rechargement des images affichées. */
export async function refreshCurrent(): Promise<void> {
  const key = get(currentKey);
  if (!key) return;
  refreshNonce.update((n) => n + 1);
  try {
    const detail = await api.getGroup(key);
    currentDetail.set(detail);
    flashStatus("↻ Rechargé");
  } catch (e: any) {
    flashStatus(`Erreur de rechargement : ${e.message || e}`, 4000);
  }
}

export function navigate(delta: number): void {
  const gs = get(groups).filter((g) => matchesFilter(g.key));
  const key = get(currentKey);
  const idx = gs.findIndex((g) => g.key === key);
  if (idx < 0) {
    if (gs.length) void selectGroup(gs[0].key);
    return;
  }
  const next = Math.max(0, Math.min(gs.length - 1, idx + delta));
  if (next !== idx) void selectGroup(gs[next].key);
}

function matchesFilter(key: string): boolean {
  const f = get(filter).trim().toLowerCase();
  return !f || key.toLowerCase().includes(f);
}

export async function setBest(dirIndex: number | null): Promise<void> {
  const key = get(currentKey);
  if (!key) return;
  const detail = get(currentDetail);
  // Toggle : re-cliquer la même tuile désélectionne.
  const newBest = detail && detail.best === dirIndex ? null : dirIndex;
  await api.setBest(key, newBest);
  if (detail) {
    detail.best = newBest;
    detail.tiles = detail.tiles.map((t) => ({ ...t, is_best: t.dir_index === newBest }));
    currentDetail.set(detail);
  }
}

export async function exportBest(): Promise<void> {
  const key = get(currentKey);
  const detail = get(currentDetail);
  if (!key || !detail || detail.best == null) return;
  const s = get(session);
  if (!s || !s.output_dir) {
    flashStatus("Aucun répertoire de sortie défini.", 4000);
    return;
  }
  try {
    let res = await api.exportBest(key, detail.best, false);
    if (res.conflict) {
      if (!confirm(`« ${res.filename} » existe déjà.\nRemplacer ?`)) return;
      res = await api.exportBest(key, detail.best, true);
    }
    if (res.ok && res.export) {
      detail.export = res.export;
      currentDetail.set(detail);
      groups.update((gs) => gs.map((g) => (g.key === key ? { ...g, copied: true } : g)));
      flashStatus(`✓ Copié : ${res.export.output_filename}`);
    }
  } catch (e: any) {
    flashStatus(`Erreur de copie : ${e.message || e}`, 4000);
  }
}

export function setMode(m: "images" | "json"): void {
  mode.set(m);
}
