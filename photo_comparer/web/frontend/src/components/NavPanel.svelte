<script lang="ts">
  import { groups, currentKey, filter, selectGroup, refreshNonce } from "../stores/app";
  import { thumbUrl } from "../lib/api";

  export let filterInput: HTMLInputElement | undefined = undefined;

  $: f = $filter.trim().toLowerCase();
  $: visible = $groups.filter((g) => !f || g.key.toLowerCase().includes(f));

  function badge(g: { copied: boolean; photo_count: number; total_dirs: number }): string {
    if (g.copied) return "✓";
    if (g.photo_count < g.total_dirs) return `(${g.photo_count}/${g.total_dirs})`;
    return "";
  }
</script>

<div class="nav">
  <input
    class="filter"
    placeholder="Filtrer… (ex: 0077)"
    bind:this={filterInput}
    bind:value={$filter}
  />
  <div class="count">{$groups.length} groupes</div>
  <div class="list">
    {#each visible as g (g.key)}
      <button
        class="item"
        class:sel={$currentKey === g.key}
        on:click={() => selectGroup(g.key)}
      >
        <img class="thumb" src={thumbUrl(g.key, $refreshNonce)} alt="" loading="lazy" />
        <span class="key">{g.key}</span>
        <span class="badge" class:copied={g.copied}>{badge(g)}</span>
      </button>
    {/each}
  </div>
</div>

<style>
  .nav {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: #1a1a1a;
    min-height: 0;
  }
  .filter {
    margin: 6px 4px 4px;
    background: #252525;
    color: #ccc;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 4px;
    font-size: 13px;
  }
  .count {
    color: #666;
    font-size: 10px;
    text-align: center;
    padding: 2px;
  }
  .list {
    flex: 1 1 auto;
    overflow-y: auto;
    min-height: 0;
  }
  .item {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    background: none;
    border: none;
    border-bottom: 1px solid #2a2a2a;
    color: #bbb;
    padding: 3px 6px;
    cursor: pointer;
    text-align: left;
    font-size: 12px;
  }
  .item:hover {
    background: #252525;
  }
  .item.sel {
    background: #1e4a7a;
    color: #fff;
  }
  .thumb {
    width: 56px;
    height: 42px;
    object-fit: cover;
    background: #232323;
    flex: 0 0 auto;
    border-radius: 2px;
  }
  .key {
    flex: 1 1 auto;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .badge {
    color: #888;
    font-size: 11px;
    flex: 0 0 auto;
  }
  .badge.copied {
    color: #4caf50;
    font-weight: bold;
  }
</style>
