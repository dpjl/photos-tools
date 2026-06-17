/** Taille de fichier lisible — port de `PhotoTile._format_size` (espaces fines FR). */
const NB = " "; // narrow no-break space

function fmtBytes(v: number): string {
  const s = v.toLocaleString("fr-FR").replace(/ |,|\s/g, NB);
  return `${s}${NB}o`;
}

export function formatSize(n: number | null | undefined): string {
  if (n == null) return "";
  if (n < 1024) return fmtBytes(n);
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}${NB}Ko (${fmtBytes(n)})`;
  return `${(n / (1024 * 1024)).toFixed(1)}${NB}Mo (${fmtBytes(n)})`;
}
