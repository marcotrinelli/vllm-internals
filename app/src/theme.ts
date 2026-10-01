/* Single source for colour: the CSS reads these as custom properties, the canvas and SVG
 * charts read them directly. Prefill and decode are close under deuteranopia, so wherever they
 * meet a legend names them */
export const PALETTE = {
  page: '#0a0d13',
  panel: '#111620',
  panel2: '#161c28',
  line: '#232b3c',
  axis: '#2f3a50',
  text: '#e6e9f0',
  text2: '#b8c0d0',
  muted: '#8994a9',
  accent: '#4c8dff',

  running: '#3ddc97',
  waiting: '#ffb454',
  prefill: '#a78bfa',
  decode: '#2fb8ff',
  kv: '#ff9f5a',
  hit: '#3ddc97',
  // status critical: always shown with a label, never alone
  preempted: '#ff7b72',

  good: '#3ddc97',
  warning: '#ffb454',
  critical: '#ff7b72',

  prompt: '#4c8dff',
  cached: '#1fb5a8',
  generated: '#d9529a',

  free: '#1a2130',
} as const;

// the request labels' column, which the engine timeline leaves empty too so the x axes line up
export const TRACK_LEFT = 40;

// one hue per request, cycled, so a request reads the same in every lane and tooltip
const REQ = ['#5aa2ff', '#37d399', '#ffb454', '#ff6fae', '#2ad4c8', '#7c5cff', '#ff6b6b', '#9ecbff'];
export const reqColor = (id: number): string => REQ[((id % REQ.length) + REQ.length) % REQ.length];

export const alpha = (hex: string, a: number): string => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
};

export function applyPalette(root: HTMLElement): void {
  for (const [name, value] of Object.entries(PALETTE)) {
    root.style.setProperty(`--${name}`, value);
  }
}
