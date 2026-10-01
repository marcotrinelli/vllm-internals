const bad = (v: number | null | undefined): v is null | undefined => v == null || Number.isNaN(v);

export const fmt = (v: number | null | undefined, d = 2) => (bad(v) ? '—' : v.toFixed(d));
export const pct = (v: number | null | undefined, d = 1) => (bad(v) ? '—' : `${(v * 100).toFixed(d)}%`);
export const int = (v: number | null | undefined) => (bad(v) ? '—' : Math.round(v).toLocaleString('en-US'));
export const ms = (v: number | null | undefined) =>
  bad(v) ? '—' : `${(v * 1e3).toFixed(v * 1e3 >= 100 ? 0 : 1)} ms`;
export const secs = (v: number | null | undefined) =>
  bad(v) ? '—' : v >= 10 ? `${v.toFixed(1)} s` : `${v.toFixed(3)} s`;
