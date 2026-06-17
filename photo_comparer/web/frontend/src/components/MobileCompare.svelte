<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import {
    currentDetail,
    currentKey,
    groups,
    filter,
    navigate,
    setBest,
    exportBest,
    selectGroup,
    status,
    refreshCurrent,
    refreshNonce,
    thumbDir,
    session,
  } from "../stores/app";
  import {
    viewport,
    computeTransform,
    applyPan,
    applyZoomAt,
    zoomToActual,
    FIT,
  } from "../lib/viewport";
  import { imageUrl, thumbUrl } from "../lib/api";
  import { get } from "svelte/store";

  const dispatch = createEventDispatcher();

  let detail = get(currentDetail);
  let activeDir = 0;
  let peekDir: number | null = null;
  let drawerOpen = false;

  let container: HTMLDivElement;
  let cw = 0;
  let ch = 0;
  let imgW = 0;
  let imgH = 0;
  let loaded = false;

  $: detail = $currentDetail;
  $: present = detail ? detail.tiles.filter((t) => t.present) : [];
  // Recadrer activeDir sur le groupe courant (best ou première version présente).
  $: if (detail) ensureActive(detail.key);

  let lastKey: string | null = null;
  // Répertoire « collant » : conservé en changeant de photo (groupe).
  let preferredDir: number | null = null;
  function ensureActive(key: string) {
    if (key === lastKey) return;
    lastKey = key;
    const d = get(currentDetail);
    if (!d) return;
    const presentIdx = d.tiles.filter((t) => t.present).map((t) => t.dir_index);
    if (presentIdx.length === 0) {
      activeDir = 0;
      loaded = false;
      return;
    }
    if (preferredDir != null && presentIdx.includes(preferredDir)) {
      // On reste sur le même folder que la photo précédente.
      activeDir = preferredDir;
    } else if (preferredDir == null && d.best != null && presentIdx.includes(d.best)) {
      // Premier affichage : on part du best s'il existe.
      activeDir = d.best;
      preferredDir = activeDir;
    } else {
      // Le folder préféré n'existe pas ici : on montre la 1re version,
      // sans oublier la préférence (elle reprendra dès qu'un groupe la contient).
      activeDir = presentIdx[0];
      if (preferredDir == null) preferredDir = activeDir;
    }
    loaded = false;
  }

  $: activeTile = present.find((t) => t.dir_index === activeDir) || present[0];
  $: shownDir = peekDir != null ? peekDir : activeDir;
  $: shownTile = present.find((t) => t.dir_index === shownDir) || activeTile;
  $: src = shownTile ? imageUrl(detail!.key, shownTile.dir_index, shownTile.mtime, $refreshNonce) : "";
  $: transform =
    loaded && cw > 0 && ch > 0
      ? computeTransform($viewport, imgW, imgH, cw, ch)
      : { scale: 1, tx: 0, ty: 0 };
  $: bestDir = detail ? detail.best : null;

  function onImgLoad(e: Event) {
    const img = e.currentTarget as HTMLImageElement;
    imgW = img.naturalWidth;
    imgH = img.naturalHeight;
    loaded = true;
  }

  function switchVersion(delta: number) {
    if (present.length < 2) return;
    const order = present.map((t) => t.dir_index);
    const i = order.indexOf(activeDir);
    const next = (i + delta + order.length) % order.length;
    activeDir = order[next];
    preferredDir = activeDir;
  }

  function gotoDir(dir: number) {
    activeDir = dir;
    preferredDir = dir;
  }

  // --------------------------------------------------------------
  // Gestes (Pointer Events)
  // --------------------------------------------------------------
  const pointers = new Map<number, { x: number; y: number }>();
  let pinchDist = 0;
  let pinchMid = { x: 0, y: 0 };
  let panLast = { x: 0, y: 0 };
  let downAt = { x: 0, y: 0, t: 0 };
  let moved = false;
  let longTimer: ReturnType<typeof setTimeout> | undefined;
  let lastTap = 0;

  function rectXY(e: PointerEvent) {
    const r = container.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  function onDown(e: PointerEvent) {
    container.setPointerCapture(e.pointerId);
    const p = rectXY(e);
    pointers.set(e.pointerId, p);
    moved = false;
    if (pointers.size === 1) {
      panLast = p;
      downAt = { x: p.x, y: p.y, t: Date.now() };
      // Appui long → peek de la version précédente.
      longTimer = setTimeout(() => {
        if (!moved && present.length >= 2) {
          const order = present.map((t) => t.dir_index);
          const i = order.indexOf(activeDir);
          peekDir = order[(i - 1 + order.length) % order.length];
          loaded = false;
        }
      }, 450);
    } else if (pointers.size === 2) {
      clearTimeout(longTimer);
      const pts = [...pointers.values()];
      pinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
      pinchMid = { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 };
    }
  }

  function onMove(e: PointerEvent) {
    if (!pointers.has(e.pointerId)) return;
    const p = rectXY(e);
    pointers.set(e.pointerId, p);

    if (pointers.size === 2) {
      const pts = [...pointers.values()];
      const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
      if (pinchDist > 0 && loaded) {
        const factor = dist / pinchDist;
        viewport.update((vp) => applyZoomAt(vp, factor, pinchMid.x, pinchMid.y, cw, ch, imgW, imgH));
      }
      pinchDist = dist;
      moved = true;
      return;
    }

    // Un doigt.
    const dx = p.x - panLast.x;
    const dy = p.y - panLast.y;
    panLast = p;
    if (Math.abs(p.x - downAt.x) > 8 || Math.abs(p.y - downAt.y) > 8) {
      moved = true;
      clearTimeout(longTimer);
    }
    const vp = get(viewport);
    if (vp.zoom > 1.001) {
      // Zoomé : pan.
      viewport.update((v) => applyPan(v, dx, dy, cw, ch, imgW, imgH));
    }
    // Sinon (ajusté) : on laisse le geste à la fin décider swipe version/groupe.
  }

  function onUp(e: PointerEvent) {
    clearTimeout(longTimer);
    const wasPeek = peekDir != null;
    if (wasPeek) {
      peekDir = null;
      loaded = false;
      pointers.delete(e.pointerId);
      return;
    }
    const start = pointers.get(e.pointerId);
    pointers.delete(e.pointerId);
    try { container.releasePointerCapture(e.pointerId); } catch {}

    const vp = get(viewport);
    if (!start) return;

    // Tap (pas de mouvement) → détecter double-tap.
    if (!moved && pointers.size === 0) {
      const now = Date.now();
      if (now - lastTap < 300) {
        lastTap = 0;
        // Double-tap : bascule fit ↔ 1:1 au point touché.
        if (vp.zoom > 1.001) viewport.set({ ...FIT });
        else viewport.set(zoomToActual(imgW, imgH, cw, ch));
      } else {
        lastTap = now;
      }
      return;
    }

    // Swipe en mode ajusté (zoom ≈ 1) : horizontal = version, vertical = groupe.
    if (vp.zoom <= 1.001 && pointers.size === 0) {
      const tdx = panLast.x - downAt.x;
      const tdy = panLast.y - downAt.y;
      if (Math.abs(tdx) > 50 && Math.abs(tdx) > Math.abs(tdy)) {
        switchVersion(tdx < 0 ? +1 : -1);
      } else if (Math.abs(tdy) > 60 && Math.abs(tdy) > Math.abs(tdx)) {
        navigate(tdy < 0 ? +1 : -1);
      }
    }
  }

  // Pré-chargement des versions adjacentes.
  $: preloadSrcs = detail
    ? present
        .filter((t) => t.dir_index !== shownDir)
        .map((t) => imageUrl(detail!.key, t.dir_index, t.mtime, $refreshNonce))
    : [];

  $: gIdx = (() => {
    const f = get(filter).trim().toLowerCase();
    const list = $groups.filter((g) => !f || g.key.toLowerCase().includes(f));
    return list.findIndex((g) => g.key === $currentKey);
  })();
</script>

<div class="mc">
  <div class="topbar">
    <button on:click={() => (drawerOpen = !drawerOpen)}>☰</button>
    <span class="title">{detail ? detail.key : "—"} ({gIdx + 1}/{$groups.length})</span>
    <button on:click={refreshCurrent} title="Recharger">↻</button>
    <button on:click={() => navigate(-1)}>‹</button>
    <button on:click={() => navigate(+1)}>›</button>
  </div>

  <div
    class="stage"
    bind:this={container}
    bind:clientWidth={cw}
    bind:clientHeight={ch}
    on:pointerdown={onDown}
    on:pointermove={onMove}
    on:pointerup={onUp}
    on:pointercancel={onUp}
  >
    {#if !shownTile}
      <div class="overlay">Aucune image</div>
    {:else}
      {#if !loaded}<div class="overlay">Chargement…</div>{/if}
      <img
        {src}
        alt=""
        draggable="false"
        on:load={onImgLoad}
        style="transform: translate({transform.tx}px,{transform.ty}px) scale({transform.scale}); opacity:{loaded ? 1 : 0};"
      />
      {#if peekDir != null}<div class="peek-tag">aperçu : {shownTile.dir_name}</div>{/if}
    {/if}

    {#each preloadSrcs as ps}<img class="preload" src={ps} alt="" />{/each}
  </div>

  <div class="chips">
    {#each present as t (t.dir_index)}
      <button
        class="chip"
        class:active={t.dir_index === activeDir}
        class:best={t.dir_index === bestDir}
        on:click={() => gotoDir(t.dir_index)}
      >
        {t.dir_index === bestDir ? "✓ " : ""}{t.dir_name}
      </button>
    {/each}
  </div>

  <div class="fab">
    <button class="star" class:on={activeDir === bestDir} on:click={() => setBest(activeDir)}>
      ★ Meilleure
    </button>
    <button class="copy" disabled={bestDir == null} on:click={exportBest}>Copier →</button>
  </div>

  {#if drawerOpen}
    <div class="drawer">
      <div class="dhead">
        <input class="filter" placeholder="Filtrer…" bind:value={$filter} />
        <button class="dirbtn" on:click={() => { drawerOpen = false; dispatch("opendirs"); }}>📁</button>
        <button class="dirbtn" on:click={() => (drawerOpen = false)}>✕</button>
      </div>
      {#if ($session?.dir_names ?? []).length > 1}
        <div class="dirchips">
          <span class="dirchips-lbl">Vignettes :</span>
          {#each $session?.dir_names ?? [] as name, di}
            <button
              class="dirchip"
              class:active={$thumbDir === di}
              on:click={() => ($thumbDir = di)}
            >{name}</button>
          {/each}
        </div>
      {/if}
      <div class="ggrid">
        {#each $groups.filter((g) => !$filter.trim() || g.key.toLowerCase().includes($filter.trim().toLowerCase())) as g (g.key)}
          <button
            class="gcell"
            class:sel={g.key === $currentKey}
            on:click={() => { drawerOpen = false; selectGroup(g.key); }}
          >
            <img src={thumbUrl(g.key, $refreshNonce, $thumbDir)} alt={g.key} loading="lazy" />
            {#if g.copied}
              <span class="gbadge ok">✓</span>
            {:else if g.photo_count < g.total_dirs}
              <span class="gbadge warn">{g.photo_count}/{g.total_dirs}</span>
            {/if}
          </button>
        {/each}
      </div>
    </div>
  {/if}

  <div class="status">{$status}</div>
</div>

<style>
  .mc {
    position: fixed;
    inset: 0;
    display: flex;
    flex-direction: column;
    background: #111;
    touch-action: none;
    overflow: hidden;
  }
  .topbar {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 8px;
    background: #1e1e1e;
    border-bottom: 1px solid #333;
  }
  .topbar .title {
    flex: 1 1 auto;
    text-align: center;
    font-size: 14px;
    color: #ddd;
  }
  .topbar button {
    background: #333;
    color: #ccc;
    border: 1px solid #505050;
    border-radius: 5px;
    padding: 6px 12px;
    font-size: 16px;
  }
  .stage {
    flex: 1 1 auto;
    position: relative;
    overflow: hidden;
    background: #0e0e0e;
  }
  .stage img {
    position: absolute;
    top: 0;
    left: 0;
    transform-origin: 0 0;
    will-change: transform;
    -webkit-user-drag: none;
    max-width: none;
  }
  .preload {
    width: 1px;
    height: 1px;
    opacity: 0;
    pointer-events: none;
  }
  .overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #666;
    pointer-events: none;
  }
  .peek-tag {
    position: absolute;
    top: 8px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(0, 0, 0, 0.7);
    color: #8ac4ff;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
  }
  .chips {
    flex: 0 0 auto;
    display: flex;
    gap: 6px;
    overflow-x: auto;
    padding: 6px 8px;
    background: #181818;
    border-top: 1px solid #2a2a2a;
  }
  .chip {
    flex: 0 0 auto;
    background: #2a2a2a;
    color: #aaa;
    border: 1px solid #444;
    border-radius: 14px;
    padding: 6px 12px;
    font-size: 12px;
    white-space: nowrap;
  }
  .chip.active {
    background: #1a3a6a;
    color: #8ac4ff;
    border-color: #2a5a9a;
  }
  .chip.best {
    border-color: #4caf50;
  }
  .fab {
    flex: 0 0 auto;
    display: flex;
    gap: 8px;
    padding: 8px;
    background: #181818;
  }
  .fab button {
    flex: 1 1 0;
    padding: 12px;
    border-radius: 8px;
    font-size: 15px;
    border: 1px solid #505050;
    background: #333;
    color: #ddd;
  }
  .fab .star.on {
    background: #1a5c1a;
    border-color: #2e7a2e;
    color: #fff;
  }
  .fab .copy:disabled {
    color: #555;
    background: #222;
  }
  .drawer {
    position: absolute;
    inset: 0;
    background: #141414;
    display: flex;
    flex-direction: column;
    z-index: 20;
  }
  .dhead {
    flex: 0 0 auto;
    display: flex;
    gap: 6px;
    padding: 8px;
    background: #1e1e1e;
    border-bottom: 1px solid #333;
  }
  .dhead .filter {
    flex: 1 1 auto;
    min-width: 0;
    background: #252525;
    color: #ccc;
    border: 1px solid #555;
    border-radius: 6px;
    padding: 10px;
    font-size: 15px;
  }
  .dirbtn {
    flex: 0 0 auto;
    background: #333;
    color: #ccc;
    border: 1px solid #505050;
    border-radius: 6px;
    padding: 0 14px;
    font-size: 17px;
  }
  .dirchips {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 5px;
    overflow-x: auto;
    padding: 5px 8px;
    background: #161616;
    border-bottom: 1px solid #2a2a2a;
    touch-action: pan-x;
  }
  .dirchips-lbl {
    flex: 0 0 auto;
    color: #777;
    font-size: 11px;
  }
  .dirchip {
    flex: 0 0 auto;
    background: #2a2a2a;
    color: #aaa;
    border: 1px solid #444;
    border-radius: 13px;
    padding: 5px 11px;
    font-size: 12px;
    white-space: nowrap;
  }
  .dirchip.active {
    background: #1a3a6a;
    color: #8ac4ff;
    border-color: #2a5a9a;
  }
  .ggrid {
    flex: 1 1 auto;
    overflow-y: auto;
    touch-action: pan-y;
    -webkit-overflow-scrolling: touch;
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    grid-auto-rows: min-content;
    align-items: start;
    gap: 2px;
    padding: 2px;
    align-content: start;
  }
  .gcell {
    display: block;
    position: relative;
    padding: 0;
    margin: 0;
    border: 2px solid transparent;
    border-radius: 4px;
    background: #222;
    overflow: hidden;
    line-height: 0;
  }
  .gcell img {
    width: 100%;
    height: auto;
    aspect-ratio: 5 / 4;
    object-fit: cover;
    display: block;
  }
  .gcell.sel {
    border-color: #4aa3ff;
  }
  .gbadge {
    position: absolute;
    top: 1px;
    right: 1px;
    font-size: 9px;
    font-weight: bold;
    padding: 0 4px;
    border-radius: 7px;
    line-height: 1.45;
  }
  .gbadge.ok {
    background: rgba(26, 92, 26, 0.92);
    color: #aaeaaa;
  }
  .gbadge.warn {
    background: rgba(0, 0, 0, 0.65);
    color: #e0b060;
  }
  .status {
    position: absolute;
    bottom: 76px;
    left: 50%;
    transform: translateX(-50%);
    color: #8ac4ff;
    font-size: 12px;
    pointer-events: none;
    text-shadow: 0 1px 3px #000;
  }
</style>
