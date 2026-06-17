<script lang="ts">
  import { currentKey, currentDetail } from "../stores/app";
  import { api } from "../lib/api";

  let text = "";
  let saved = "";
  let timer: ReturnType<typeof setTimeout> | undefined;
  let key: string | null = null;

  // Charger la note quand le groupe change.
  $: if ($currentDetail && $currentDetail.key !== key) {
    key = $currentDetail.key;
    text = $currentDetail.note || "";
    saved = "";
  }

  function onInput() {
    saved = "";
    clearTimeout(timer);
    timer = setTimeout(save, 600);
  }

  async function save() {
    const k = $currentKey;
    if (!k) return;
    await api.putNote(k, text);
    if ($currentDetail) $currentDetail.note = text;
    saved = "✓ Enregistré";
  }
</script>

<div class="notes">
  <div class="hdr">
    <span class="title">Notes — {key ?? ""}</span>
    <span class="saved">{saved}</span>
  </div>
  <textarea
    placeholder="Tapez une note pour ce groupe de photos…"
    bind:value={text}
    on:input={onInput}
  ></textarea>
</div>

<style>
  .notes {
    flex: 0 0 auto;
    background: #18182e;
    border-top: 1px solid #3a3a6a;
    padding: 4px 8px 6px;
    height: 115px;
    display: flex;
    flex-direction: column;
  }
  .hdr {
    display: flex;
    justify-content: space-between;
    margin-bottom: 3px;
  }
  .title {
    color: #8ac4ff;
    font-size: 11px;
    font-weight: bold;
  }
  .saved {
    color: #4caf50;
    font-size: 10px;
  }
  textarea {
    flex: 1 1 auto;
    background: #111128;
    color: #ccc;
    border: 1px solid #3a3a6a;
    border-radius: 3px;
    padding: 3px;
    font-size: 12px;
    resize: none;
    font-family: inherit;
  }
</style>
