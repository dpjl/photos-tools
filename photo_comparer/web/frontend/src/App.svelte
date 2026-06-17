<script lang="ts">
  import { onMount } from "svelte";
  import NavPanel from "./components/NavPanel.svelte";
  import ComparisonGrid from "./components/ComparisonGrid.svelte";
  import NotePanel from "./components/NotePanel.svelte";
  import ExportedBar from "./components/ExportedBar.svelte";
  import DirBrowser from "./components/DirBrowser.svelte";
  import JsonDiff from "./components/JsonDiff.svelte";
  import MobileCompare from "./components/MobileCompare.svelte";
  import {
    loadSession,
    groups,
    currentKey,
    currentDetail,
    mode,
    soloIndex,
    filter,
    showNotes,
    status,
    navigate,
    setMode,
    exportBest,
    refreshCurrent,
  } from "./stores/app";
  import {
    viewport,
    refGeom,
    adjustZoom,
    zoomToActual,
    FIT,
    type Geom,
  } from "./lib/viewport";
  import { get } from "svelte/store";

  let filterInput: HTMLInputElement | undefined;
  let showDirs = false;
  let isMobile = false;
  let geom: Geom | null = null;
  refGeom.subscribe((g) => (geom = g));

  onMount(() => {
    loadSession().catch((e) => status.set(`Erreur : ${e.message || e}`));
    const mq = window.matchMedia("(max-width: 768px)");
    const upd = () => (isMobile = mq.matches);
    upd();
    mq.addEventListener("change", upd);
    return () => mq.removeEventListener("change", upd);
  });

  $: detail = $currentDetail;
  $: filtered = $groups.filter(
    (g) => !$filter.trim() || g.key.toLowerCase().includes($filter.trim().toLowerCase()),
  );
  $: idx = filtered.findIndex((g) => g.key === $currentKey);
  $: info = detail
    ? `${detail.key}   (${idx + 1} / ${filtered.length})`
    : "Aucune photo — ouvrir des répertoires";
  $: canCopy = !!detail && detail.best != null;

  function zoomIn() {
    viewport.update((vp) => adjustZoom(vp, +1));
  }
  function zoomOut() {
    viewport.update((vp) => adjustZoom(vp, -1));
  }
  function zoomFit() {
    viewport.set({ ...FIT });
  }
  function zoom100() {
    if (geom) viewport.set(zoomToActual(geom.imgW, geom.imgH, geom.vw, geom.vh));
  }

  function onKey(e: KeyboardEvent) {
    const t = e.target as HTMLElement;
    const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA");
    if (e.key === "F5") {
      e.preventDefault();
      void refreshCurrent();
      return;
    }
    if (e.ctrlKey && e.key === "f") {
      e.preventDefault();
      filterInput?.focus();
      return;
    }
    if (e.ctrlKey && e.key === "Enter") {
      e.preventDefault();
      void exportBest();
      return;
    }
    if (e.ctrlKey && (e.key === "=" || e.key === "+")) {
      e.preventDefault();
      zoomIn();
      return;
    }
    if (e.ctrlKey && e.key === "-") {
      e.preventDefault();
      zoomOut();
      return;
    }
    if (e.ctrlKey && e.key === "0") {
      e.preventDefault();
      zoomFit();
      return;
    }
    if (e.ctrlKey && (e.key === "i" || e.key === "j")) {
      e.preventDefault();
      setMode(e.key === "j" ? "json" : "images");
      return;
    }
    if (typing) return;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") navigate(+1);
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") navigate(-1);
    else if (e.key === "Escape") soloIndex.set(null);
    else if (e.key === "1") zoom100();
    else if (e.key === "r" || e.key === "R") void refreshCurrent();
    else if (e.key === "n" || e.key === "N") showNotes.update((v) => !v);
  }

  async function onDirsApplied() {
    showDirs = false;
    await loadSession();
  }
</script>

<svelte:window on:keydown={onKey} />

{#if isMobile}
  <MobileCompare on:opendirs={() => (showDirs = true)} />
{:else}
  <div class="app">
    <div class="toolbar">
      <button on:click={() => (showDirs = true)}>Répertoires…</button>
      <span class="sep" />
      <button class:active={$mode === "images"} on:click={() => setMode("images")}>Images</button>
      <button class:active={$mode === "json"} on:click={() => setMode("json")}>{"{ } JSON"}</button>
      <span class="sep" />
      <button on:click={zoomOut} title="Zoom arrière">−</button>
      <button on:click={zoomIn} title="Zoom avant">+</button>
      <button on:click={zoomFit} title="Ajuster">Fit</button>
      <button on:click={zoom100} title="100 %">1:1</button>
      <span class="sep" />
      <button on:click={refreshCurrent} title="Recharger les images (R / F5)">↻</button>
      <span class="info">{info}</span>
      <button class:active={$showNotes} on:click={() => showNotes.update((v) => !v)}>📝 Notes</button>
      <button class="copy" class:on={canCopy} disabled={!canCopy} on:click={exportBest}>
        Copier la meilleure →
      </button>
    </div>

    <div class="body">
      <div class="nav-col"><NavPanel bind:filterInput /></div>
      <div class="main-col">
        <div class="view">
          {#if $mode === "json"}
            <JsonDiff />
          {:else}
            <ComparisonGrid />
          {/if}
        </div>
        <ExportedBar />
      </div>
    </div>

    {#if $showNotes}<NotePanel />{/if}
    <div class="status">{$status}</div>
  </div>
{/if}

{#if showDirs}
  <DirBrowser on:close={() => (showDirs = false)} on:applied={onDirsApplied} />
{/if}

<style>
  .app {
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }
  .toolbar {
    display: flex;
    align-items: center;
    gap: 4px;
    background: #242424;
    border-bottom: 1px solid #3a3a3a;
    padding: 4px 6px;
    flex: 0 0 auto;
  }
  .toolbar button {
    background: #363636;
    color: #ccc;
    border: 1px solid #505050;
    border-radius: 3px;
    padding: 4px 10px;
    cursor: pointer;
    font-size: 13px;
  }
  .toolbar button:hover {
    background: #464646;
  }
  .toolbar button.active {
    background: #1a3a6a;
    color: #8ac4ff;
    border-color: #2a5a9a;
    font-weight: bold;
  }
  .toolbar button:disabled {
    color: #555;
    background: #2a2a2a;
    border-color: #3a3a3a;
    cursor: default;
  }
  .copy.on {
    background: #1a5c1a;
    color: #ddd;
    border-color: #2e7a2e;
    font-weight: bold;
  }
  .sep {
    width: 1px;
    align-self: stretch;
    background: #3a3a3a;
    margin: 2px 4px;
  }
  .info {
    flex: 1 1 auto;
    text-align: center;
    color: #aaa;
    font-size: 13px;
  }
  .body {
    flex: 1 1 auto;
    display: flex;
    min-height: 0;
  }
  .nav-col {
    flex: 0 0 200px;
    min-width: 0;
    border-right: 1px solid #2a2a2a;
  }
  .main-col {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
  }
  .view {
    flex: 1 1 auto;
    min-height: 0;
  }
  .status {
    flex: 0 0 auto;
    color: #888;
    font-size: 11px;
    padding: 2px 8px;
    background: #1c1c1c;
    border-top: 1px solid #2a2a2a;
    min-height: 16px;
  }
</style>
