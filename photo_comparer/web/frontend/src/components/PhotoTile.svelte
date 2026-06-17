<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import { viewport, refGeom, computeTransform, applyWheel, applyPan } from "../lib/viewport";
  import { imageUrl } from "../lib/api";
  import { formatSize } from "../lib/format";
  import { refreshNonce } from "../stores/app";
  import type { Tile } from "../lib/api";

  export let groupKey: string;
  export let tile: Tile;

  const dispatch = createEventDispatcher();

  let container: HTMLDivElement;
  let cw = 0;
  let ch = 0;
  let imgW = 0;
  let imgH = 0;
  let loaded = false;
  let errored = false;

  // Pan
  let panning = false;
  let lastX = 0;
  let lastY = 0;

  $: src = tile.present ? imageUrl(groupKey, tile.dir_index, tile.mtime, $refreshNonce) : "";
  $: transform =
    loaded && cw > 0 && ch > 0
      ? computeTransform($viewport, imgW, imgH, cw, ch)
      : { scale: 1, tx: 0, ty: 0 };

  // Recharger quand la tuile change de groupe/dir.
  $: if (src) {
    loaded = false;
    errored = false;
  }

  function onImgLoad(e: Event) {
    const img = e.currentTarget as HTMLImageElement;
    imgW = img.naturalWidth;
    imgH = img.naturalHeight;
    loaded = true;
    refGeom.set({ imgW, imgH, vw: cw, vh: ch });
  }

  function onWheel(e: WheelEvent) {
    if (!loaded) return;
    e.preventDefault();
    const rect = container.getBoundingClientRect();
    viewport.update((vp) =>
      applyWheel(vp, e.deltaY, e.clientX - rect.left, e.clientY - rect.top, cw, ch, imgW, imgH),
    );
  }

  function onPointerDown(e: PointerEvent) {
    if (e.button !== 0) return;
    if (e.ctrlKey) {
      dispatch("best", tile.dir_index);
      return;
    }
    panning = true;
    lastX = e.clientX;
    lastY = e.clientY;
    container.setPointerCapture(e.pointerId);
  }

  function onPointerMove(e: PointerEvent) {
    if (!panning || !loaded) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    viewport.update((vp) => applyPan(vp, dx, dy, cw, ch, imgW, imgH));
  }

  function onPointerUp(e: PointerEvent) {
    panning = false;
    try {
      container.releasePointerCapture(e.pointerId);
    } catch {}
  }

  function onDblClick(e: MouseEvent) {
    if (e.ctrlKey) return;
    if (tile.present) dispatch("solo", tile.dir_index);
  }
</script>

<div class="tile" class:best={tile.is_best} on:dblclick={onDblClick}>
  <div class="hdr">
    <div class="dir">{tile.dir_name}</div>
    {#if tile.present}
      <div class="file">{tile.filename}</div>
      {#if tile.size != null}<div class="size">{formatSize(tile.size)}</div>{/if}
    {/if}
  </div>

  <div
    class="viewport"
    bind:this={container}
    bind:clientWidth={cw}
    bind:clientHeight={ch}
    on:wheel={onWheel}
    on:pointerdown={onPointerDown}
    on:pointermove={onPointerMove}
    on:pointerup={onPointerUp}
    on:pointercancel={onPointerUp}
  >
    {#if !tile.present}
      <div class="overlay">Absent</div>
    {:else if errored}
      <div class="overlay">Erreur</div>
    {:else}
      {#if !loaded}<div class="overlay">Chargement…</div>{/if}
      <img
        {src}
        alt={tile.filename}
        draggable="false"
        on:load={onImgLoad}
        on:error={() => (errored = true)}
        style="transform: translate({transform.tx}px, {transform.ty}px) scale({transform.scale}); opacity:{loaded ? 1 : 0};"
      />
    {/if}
  </div>
</div>

<style>
  .tile {
    display: flex;
    flex-direction: column;
    border: 1px solid #3a3a3a;
    background: #181818;
    min-height: 0;
    min-width: 0;
    overflow: hidden;
  }
  .tile.best {
    border: 2px solid #4caf50;
  }
  .hdr {
    flex: 0 0 auto;
    text-align: center;
    padding: 2px;
    line-height: 1.15;
  }
  .dir {
    color: #888;
    font-size: 10px;
  }
  .file {
    color: #5a9fd4;
    font-size: 9px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .size {
    color: #fff;
    font-size: 8px;
  }
  .viewport {
    position: relative;
    flex: 1 1 auto;
    min-height: 0;
    overflow: hidden;
    background: #161616;
    cursor: grab;
    touch-action: none;
  }
  .viewport:active {
    cursor: grabbing;
  }
  img {
    position: absolute;
    top: 0;
    left: 0;
    transform-origin: 0 0;
    will-change: transform;
    user-select: none;
    -webkit-user-drag: none;
    max-width: none;
  }
  .overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #555;
    font-size: 16px;
    background: #141414;
    pointer-events: none;
  }
</style>
