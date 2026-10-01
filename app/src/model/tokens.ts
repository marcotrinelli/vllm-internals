import type { Request } from './types';

/* One token of a sequence. `kind`: prompt served from the prefix cache, prompt, generated */
export interface Token {
  pos: number;
  kind: 'cached' | 'prompt' | 'generated';
  id: number | null;
  text: string | null;
  // generated tokens only
  j?: number;
  logprob?: number | null;
  t?: number;
  gap?: number | null;
}

export interface Elided {
  elided: number;
}

/* Token at sequence position `pos`: prompt positions from the ids + vocab, output positions
 * from the streamed text */
export function tokenAt(r: Request, vocab: Record<string, string>, pos: number): Token {
  if (pos < r.promptTokens) {
    const id = r.promptIds && pos < r.promptIds.length ? r.promptIds[pos] : null;
    return {
      pos, id, kind: pos < r.cachedTokens ? 'cached' : 'prompt',
      text: id == null ? null : (vocab[String(id)] ?? '�'),
    };
  }
  const j = pos - r.promptTokens;
  return {
    pos, j, kind: 'generated',
    id: r.outputIds?.[j] ?? null,
    text: r.tokenText[j] ?? null,
    logprob: r.logprobs?.[j] ?? null,
    t: r.tokenT[j],
    gap: j ? r.tokenT[j] - r.tokenT[j - 1] : null,
  };
}

/* The tokens at positions [from, to), keeping both ends when there are more than `cap` */
export function tokenRange(
  r: Request, vocab: Record<string, string>, from: number, to: number, cap: number,
): Array<Token | Elided> {
  const out: Array<Token | Elided> = [];
  const push = (a: number, b: number) => { for (let p = a; p < b; p++) { out.push(tokenAt(r, vocab, p)); } };
  if (to - from <= cap) { push(from, to); return out; }
  const nHead = Math.round(cap * 0.6);
  const nTail = cap - nHead;
  push(from, from + nHead);
  out.push({ elided: to - from - nHead - nTail });
  push(to - nTail, to);
  return out;
}

/* Newlines and tabs would otherwise vanish into the layout */
export const visible = (s: string | null) => (s == null ? '?' : s.replace(/\n/g, '↵\n').replace(/\t/g, '→\t'));
