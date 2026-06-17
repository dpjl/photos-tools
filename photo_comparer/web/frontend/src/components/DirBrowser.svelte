<script lang="ts">
  import { createEventDispatcher, onMount } from "svelte";
  import { api, type DirData, type FsList } from "../lib/api";

  const dispatch = createEventDispatcher();
  const MAX_DIRS = 10;

  let dirs: DirData[] = [];
  let outputDir = "";
  let fs: FsList | null = null;
  let busy = false;
  $: parentPath = fs?.parent ?? "";
  $: curPath = fs?.path ?? "";
  $: imgHere = fs?.image_count ?? 0;

  onMount(async () => {
    const s = await api.getSession();
    dirs = s.dir_data.map((d) => ({ ...d }));
    outputDir = s.output_dir;
    await browse("");
  });

  async function browse(path: string) {
    try {
      fs = await api.fsList(path);
    } catch (e: any) {
      fs = null;
    }
  }

  function addDir(path: string) {
    if (dirs.length >= MAX_DIRS) return;
    if (dirs.some((d) => d.path === path)) return;
    dirs = [...dirs, { path, enabled: true }];
  }
  function removeDir(i: number) {
    dirs = dirs.filter((_, j) => j !== i);
  }
  function move(i: number, delta: number) {
    const j = i + delta;
    if (j < 0 || j >= dirs.length) return;
    const copy = [...dirs];
    [copy[i], copy[j]] = [copy[j], copy[i]];
    dirs = copy;
  }
  function toggle(i: number) {
    dirs = dirs.map((d, j) => (j === i ? { ...d, enabled: !d.enabled } : d));
  }

  async function apply() {
    busy = true;
    try {
      await api.setDirs(dirs, outputDir);
      dispatch("applied");
    } finally {
      busy = false;
    }
  }
</script>

<div class="backdrop" on:click|self={() => dispatch("close")}>
  <div class="dialog">
    <h2>Sélection des répertoires</h2>

    <div class="cols">
      <!-- Répertoires sélectionnés -->
      <div class="selected">
        <div class="lbl">Répertoires source ({dirs.length}/{MAX_DIRS})</div>
        {#each dirs as d, i (d.path)}
          <div class="row" class:off={!d.enabled}>
            <input type="checkbox" checked={d.enabled} on:change={() => toggle(i)} />
            <span class="path" title={d.path}>{d.path}</span>
            <button on:click={() => move(i, -1)} disabled={i === 0}>▲</button>
            <button on:click={() => move(i, +1)} disabled={i === dirs.length - 1}>▼</button>
            <button class="rm" on:click={() => removeDir(i)}>✕</button>
          </div>
        {/each}
        {#if dirs.length === 0}<div class="hint">Ajoutez des dossiers depuis l'explorateur →</div>{/if}

        <div class="lbl out">Répertoire de sortie</div>
        <div class="row">
          <input class="outedit" bind:value={outputDir} placeholder="(aucun)" />
        </div>
      </div>

      <!-- Explorateur -->
      <div class="browser">
        <div class="crumbs">
          <button on:click={() => browse("")}>⌂ racines</button>
          {#if fs && fs.parent != null}
            <button on:click={() => browse(parentPath)}>⬑ parent</button>
          {/if}
          <span class="here">{fs ? fs.path || "racines" : "…"}</span>
        </div>
        {#if fs}
          {#if curPath}
            <div class="curdir">
              <span>{imgHere} images ici</span>
              <button class="add" on:click={() => addDir(curPath)} disabled={dirs.length >= MAX_DIRS}>+ source</button>
              <button class="add" on:click={() => (outputDir = curPath)}>→ sortie</button>
            </div>
          {/if}
          <div class="entries">
            {#each fs.entries as e (e.path)}
              <div class="entry">
                <button class="nav" on:click={() => browse(e.path)} title={e.path}>
                  📁 {e.name}
                  {#if e.image_count != null}<span class="cnt">{e.image_count}</span>{/if}
                </button>
                <button class="add" on:click={() => addDir(e.path)} disabled={dirs.length >= MAX_DIRS}>+</button>
              </div>
            {/each}
          </div>
        {:else}
          <div class="hint">Dossier inaccessible.</div>
        {/if}
      </div>
    </div>

    <div class="actions">
      <button on:click={() => dispatch("close")}>Annuler</button>
      <button class="ok" on:click={apply} disabled={busy}>Valider</button>
    </div>
  </div>
</div>

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 50;
  }
  .dialog {
    background: #262626;
    color: #ccc;
    border: 1px solid #444;
    border-radius: 6px;
    width: min(900px, 94vw);
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    padding: 12px 14px;
  }
  h2 {
    margin: 0 0 8px;
    font-size: 16px;
    color: #ddd;
  }
  .cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    min-height: 0;
    flex: 1 1 auto;
  }
  .selected,
  .browser {
    display: flex;
    flex-direction: column;
    min-height: 0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 6px;
    overflow: auto;
    max-height: 60vh;
  }
  .lbl {
    color: #999;
    font-size: 12px;
    margin-bottom: 4px;
  }
  .lbl.out {
    margin-top: 10px;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-bottom: 3px;
  }
  .row.off .path {
    color: #555;
  }
  .path {
    flex: 1 1 auto;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 12px;
  }
  .outedit {
    flex: 1 1 auto;
    background: #1e1e1e;
    color: #ccc;
    border: 1px solid #444;
    border-radius: 3px;
    padding: 4px;
  }
  button {
    background: #3a3a3a;
    color: #ccc;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 2px 7px;
    cursor: pointer;
    font-size: 12px;
  }
  button:hover {
    background: #4a4a4a;
  }
  button:disabled {
    color: #555;
    cursor: default;
  }
  .rm {
    color: #d77;
  }
  .crumbs {
    display: flex;
    gap: 4px;
    align-items: center;
    margin-bottom: 6px;
    flex-wrap: wrap;
  }
  .here {
    font-size: 11px;
    color: #888;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .curdir {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: #7ec896;
    margin-bottom: 6px;
  }
  .entries {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .entry {
    display: flex;
    gap: 4px;
  }
  .entry .nav {
    flex: 1 1 auto;
    text-align: left;
    display: flex;
    justify-content: space-between;
    gap: 8px;
  }
  .cnt {
    color: #4caf50;
    font-size: 11px;
  }
  .add {
    color: #8ac4ff;
  }
  .hint {
    color: #777;
    font-size: 12px;
    padding: 6px 0;
  }
  .actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-top: 10px;
  }
  .ok {
    border-color: #5a9fd4;
    color: #8ac4ff;
  }
</style>
