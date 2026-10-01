import { useEffect, useRef, useState, type MouseEvent, type RefObject } from 'react';

export interface TipAnchor {
  x: number;
  y: number;
  flipX: boolean;
  flipY: boolean;
}

/* A tooltip anchored inside a positioned wrapper, flipped away from the near edges */
export function useTip<T>(wrapRef: RefObject<HTMLElement | null>) {
  const [tip, setTip] = useState<(TipAnchor & { data: T }) | null>(null);
  const place = (e: MouseEvent, data: T) => {
    const box = wrapRef.current?.getBoundingClientRect();
    if (!box || !box.width) { return; }
    const x = e.clientX - box.left;
    const y = e.clientY - box.top;
    setTip({ x, y, flipX: x > box.width * 0.55, flipY: y > box.height * 0.6, data });
  };
  const clear = () => setTip(null);
  const style = tip ? {
    left: tip.x + 14,
    top: tip.y + 14,
    transform: `translate(${tip.flipX ? 'calc(-100% - 28px)' : '0'}, ${tip.flipY ? 'calc(-100% - 28px)' : '0'})`,
  } : undefined;
  return { tip: tip?.data ?? null, place, clear, style };
}

/* Width of an element, kept current as it resizes */
export function useWidth<T extends HTMLElement>(fallback = 600): [RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver !== 'function') { return; }
    const observer = new ResizeObserver(() => setWidth(el.clientWidth || fallback));
    observer.observe(el);
    setWidth(el.clientWidth || fallback);
    return () => observer.disconnect();
  }, [fallback]);
  return [ref, width];
}
