<script lang="ts">
  import { currentDetail, setBest } from "../stores/app";
  import { thumbUrl } from "../lib/api";

  $: rec = $currentDetail ? $currentDetail.export : null;

  function restore() {
    const detail = $currentDetail;
    if (!detail || !rec) return;
    const tile = detail.tiles.find((t) => t.dir_name === rec!.dir_name);
    if (tile) void setBest(tile.dir_index);
  }
</script>

{#if rec}
  <div class="bar">
    <img class="thumb" src={thumbUrl(rec.group_key)} alt="" />
    <span class="info">
      <b>✓ Exportée</b> · {rec.output_filename}
      <span class="dim">depuis</span> <i>{rec.dir_name}</i>
    </span>
    <button on:click={restore}>↩ Restaurer sélection</button>
  </div>
{/if}

<style>
  .bar {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 12px;
    background: #17241a;
    border-top: 1px solid #2a5c35;
    padding: 4px 10px;
    height: 56px;
  }
  .thumb {
    width: 64px;
    height: 46px;
    object-fit: cover;
    border: 1px solid #2a5c35;
    background: #0e1a12;
  }
  .info {
    flex: 1 1 auto;
    color: #7ec896;
    font-size: 12px;
  }
  .info b {
    color: #aaeaaa;
  }
  .dim {
    color: #3a7a50;
  }
  button {
    background: #1c3d24;
    color: #7ec896;
    border: 1px solid #2a5c35;
    border-radius: 3px;
    padding: 4px 10px;
    cursor: pointer;
    font-size: 12px;
  }
  button:hover {
    background: #265c32;
    color: #aaeaaa;
  }
</style>
