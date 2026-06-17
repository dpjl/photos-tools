/** Client API typé. */

export interface DirData {
  path: string;
  enabled: boolean;
}

export interface SessionState {
  dir_data: DirData[];
  output_dir: string;
  dir_names: string[];
  mode: string;
  group_count: number;
}

export interface GroupSummary {
  key: string;
  photo_count: number;
  total_dirs: number;
  copied: boolean;
}

export interface Tile {
  dir_index: number;
  dir_name: string;
  present: boolean;
  filename: string | null;
  size: number | null;
  mtime: number | null;
  is_best: boolean;
  has_json: boolean;
}

export interface ExportRecord {
  group_key: string;
  source_path: string;
  dir_name: string;
  output_filename: string;
}

export interface GroupDetail {
  key: string;
  tiles: Tile[];
  best: number | null;
  note: string;
  export: ExportRecord | null;
}

export interface JsonTile {
  dir_index: number;
  dir_name: string;
  present: boolean;
  filename?: string;
  size?: number | null;
  lines?: string[];
  diff?: number[];
  gap?: number[];
  is_best: boolean;
}

export interface FsEntry {
  name: string;
  path: string;
  image_count: number | null;
}

export interface FsList {
  path: string;
  parent: string | null;
  entries: FsEntry[];
  image_count: number;
}

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function send<T>(method: string, url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const err: any = new Error((data && data.error) || `${r.status}`);
    err.status = r.status;
    err.data = data;
    throw err;
  }
  return data as T;
}

export const api = {
  getSession: () => getJSON<SessionState>("/api/session"),
  setDirs: (dirs: DirData[], output_dir: string) =>
    send<SessionState>("POST", "/api/session/dirs", { dirs, output_dir }),
  fsList: (path: string) => getJSON<FsList>(`/api/fs/list?path=${encodeURIComponent(path)}`),
  getGroups: () => getJSON<{ groups: GroupSummary[] }>("/api/groups"),
  getGroup: (key: string) => getJSON<GroupDetail>(`/api/group/${encodeURIComponent(key)}`),
  getJson: (key: string) => getJSON<{ key: string; tiles: JsonTile[] }>(`/api/json/${encodeURIComponent(key)}`),
  setBest: (key: string, dir_index: number | null) =>
    send<{ ok: boolean; best: number | null }>("POST", "/api/best", { key, dir_index }),
  exportBest: (key: string, dir_index: number, overwrite: boolean) =>
    send<{ ok: boolean; export?: ExportRecord; conflict?: boolean; filename?: string }>(
      "POST", "/api/export", { key, dir_index, overwrite },
    ),
  getNotes: () => getJSON<{ notes: Record<string, string> }>("/api/notes"),
  putNote: (key: string, text: string) =>
    send<{ ok: boolean }>("PUT", `/api/notes/${encodeURIComponent(key)}`, { text }),
};

export function imageUrl(
  key: string,
  dir: number,
  mtime: number | null,
  nonce = 0,
): string {
  const v = mtime != null ? `&v=${Math.floor(mtime)}` : "";
  const r = nonce ? `&r=${nonce}` : "";
  return `/api/image?key=${encodeURIComponent(key)}&dir=${dir}${v}${r}`;
}

export function thumbUrl(key: string, nonce = 0, dir: number | null = null): string {
  const r = nonce ? `&r=${nonce}` : "";
  const d = dir != null ? `&dir=${dir}` : "";
  return `/api/thumb?key=${encodeURIComponent(key)}${d}${r}`;
}
