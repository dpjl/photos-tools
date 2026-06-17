<script lang="ts">
  import { currentKey, refreshNonce } from "../stores/app";
  import { api, type JsonTile } from "../lib/api";
  import { formatSize } from "../lib/format";

  let tiles: JsonTile[] = [];
  let loadingKey: string | null = null;
  let lastNonce = -1;
  let containers: HTMLDivElement[] = [];
  let syncing = false;

  $: present = tiles.filter((t) => t.present);

  $: if ($currentKey && ($currentKey !== loadingKey || $refreshNonce !== lastNonce)) {
    loadingKey = $currentKey;
    lastNonce = $refreshNonce;
    load($currentKey);
  }

  async function load(key: string) {
    try {
      const res = await api.getJson(key);
      if (loadingKey === key) tiles = res.tiles;
    } catch {
      tiles = [];
    }
  }

  function lineClass(t: JsonTile, i: number): string {
    if (t.gap && t.gap.includes(i)) return "gap";
    if (t.diff && t.diff.includes(i)) return "diff";
    return "";
  }

  function onScroll(e: Event) {
    if (syncing) return;
    syncing = true;
    const src = e.currentTarget as HTMLDivElement;
    for (const c of containers) {
      if (c && c !== src) c.scrollTop = src.scrollTop;
    }
    syncing = false;
  }
</script>

<div class="json-grid" style="grid-template-columns: repeat({present.length || 1}, 1fr);">
  {#each present as t, ci (t.dir_index)}
    <div class="col">
      <div class="hdr" class:best={t.is_best}>
        <div class="dir">{t.dir_name}</div>
        <div class="file">{t.filename}</div>
        {#if t.size != null}<div class="size">{formatSize(t.size)}</div>{/if}
      </div>
      <div class="code" bind:this={containers[ci]} on:scroll={onScroll}>
        {#each t.lines ?? [] as line, i}
          <div class="ln {lineClass(t, i)}">{line || " "}</div>
        {/each}
      </div>
    </div>
  {/each}
  {#if present.length === 0}
    <div class="empty">Aucun JSON pour ce groupe</div>
  {/if}
</div>

<style>
  .json-grid {
    display: grid;
    gap: 4px;
    height: 100%;
    padding: 4px;
    box-sizing: border-box;
  }
  .col {
    display: flex;
    flex-direction: column;
    min-width: 0;
    border: 1px solid #3a3a3a;
    background: #0d0d0d;
  }
  .hdr {
    flex: 0 0 auto;
    text-align: center;
    padding: 2px;
    border-bottom: 1px solid #2a2a2a;
  }
  .hdr.best {
    border: 2px solid #4caf50;
  }
  .dir {
    color: #888;
    font-size: 10px;
  }
  .file {
    color: #5a9fd4;
    font-size: 9px;
  }
  .size {
    color: #fff;
    font-size: 8px;
  }
  .code {
    flex: 1 1 auto;
    overflow: auto;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 11px;
    color: #c8c8c8;
    line-height: 1.35;
  }
  .ln {
    white-space: pre;
    padding: 0 4px;
  }
  .ln.diff {
    background: rgba(255, 200, 0, 0.27);
  }
  .ln.gap {
    background: #282828;
  }
  .empty {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #555;
  }
</style>
