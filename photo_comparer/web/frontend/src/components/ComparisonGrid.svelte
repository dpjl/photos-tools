<script lang="ts">
  import PhotoTile from "./PhotoTile.svelte";
  import { currentDetail, soloIndex, setBest } from "../stores/app";

  $: detail = $currentDetail;
  $: tiles = detail ? detail.tiles : [];
  $: n = tiles.length;
  $: cols = n > 0 ? Math.ceil(Math.sqrt(n)) : 1;
  $: visibleTiles =
    $soloIndex != null ? tiles.filter((t) => t.dir_index === $soloIndex) : tiles;
  $: effectiveCols = $soloIndex != null ? 1 : cols;

  function onSolo(e: CustomEvent<number>) {
    soloIndex.update((s) => (s === e.detail ? null : e.detail));
  }
  function onBest(e: CustomEvent<number>) {
    void setBest(e.detail);
  }
</script>

{#if !detail}
  <div class="empty">Aucune photo — ouvrir des répertoires</div>
{:else}
  <div class="grid" style="grid-template-columns: repeat({effectiveCols}, 1fr);">
    {#each visibleTiles as tile (tile.dir_index)}
      <PhotoTile
        groupKey={detail.key}
        {tile}
        on:solo={onSolo}
        on:best={onBest}
      />
    {/each}
  </div>
{/if}

<style>
  .grid {
    display: grid;
    gap: 4px;
    padding: 4px;
    width: 100%;
    height: 100%;
    box-sizing: border-box;
    grid-auto-rows: 1fr;
  }
  .empty {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: #555;
    font-size: 14px;
  }
</style>
