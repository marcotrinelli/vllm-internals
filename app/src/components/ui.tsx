import { useEffect, useRef, type ReactNode } from 'react';

export type Tone = 'plain' | 'good' | 'warn' | 'bad';

const toneClass = (tone?: Tone) => (tone && tone !== 'plain' ? `tone-${tone}` : undefined);

/* An (i) that explains what sits next to it, on hover or keyboard focus */
export const Info = ({ text }: { text: string }) => (
  <span className="info" tabIndex={0} aria-label={text}>
    <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">
      <circle cx="8" cy="8" r="6.6" fill="none" stroke="currentColor" strokeWidth="1.3" />
      <rect x="7.3" y="7" width="1.4" height="4.6" rx="0.7" fill="currentColor" />
      <circle cx="8" cy="4.9" r="0.9" fill="currentColor" />
    </svg>
    <span className="info-tip" role="tooltip">{text}</span>
  </span>
);

interface CardProps {
  title: string;
  count?: string;
  // what the card shows and where it comes from, behind an (i) next to the title
  note?: string;
  right?: ReactNode;
  children: ReactNode;
}

const Head = ({ title, count, note, right }: Omit<CardProps, 'children'>) => (
  <header>
    <h2>{title}{note && <Info text={note} />}</h2>
    <span className="grow" />
    {count && <span className="chip">{count}</span>}
    {right}
  </header>
);

export const Card = ({ children, ...head }: CardProps) => (
  <section className="card" role="region" aria-label={head.title}>
    <Head {...head} />
    <div className="card-body">{children}</div>
  </section>
);

/* A card in a modal dialog, open while mounted; Esc, × or a click on the backdrop closes it */
export const Dialog = ({ children, onClose, ...head }: Omit<CardProps, 'right'> & { onClose: () => void }) => {
  const ref = useRef<HTMLDialogElement>(null);
  // showModal focuses the first focusable element, the (i) in the header, which would open its
  // tooltip: focus the dialog instead, and Tab still reaches the (i)
  useEffect(() => {
    ref.current?.showModal();
    ref.current?.focus();
  }, []);
  const close = () => ref.current?.close();
  return (
    // keys stay in the dialog (Esc would also clear the range, Space play)
    <dialog ref={ref} className="card dialog" aria-label={head.title} tabIndex={-1} onClose={onClose}
            onKeyDown={(e) => e.stopPropagation()} onClick={(e) => { if (e.target === e.currentTarget) { close(); } }}>
      <Head {...head} right={<button type="button" className="icon" aria-label="Close" onClick={close}>×</button>} />
      <div className="card-body">{children}</div>
    </dialog>
  );
};

interface SegProps<T> {
  // undefined: no option pressed
  value: T | undefined;
  options: Array<[T, string, boolean?]>; // value, label, disabled
  onChange: (v: T) => void;
  label: string;
}

export const Seg = <T,>({ value, options, onChange, label }: SegProps<T>) => (
  <div className="seg" role="group" aria-label={label}>
    {options.map(([v, text, disabled]) => (
      <button key={text} type="button" aria-pressed={v === value} disabled={disabled}
              onClick={() => onChange(v)}>{text}</button>
    ))}
  </div>
);

export const Row = ({ k, v, tone }: { k: string; v: ReactNode; tone?: Tone }) => (
  <div className="kv"><span>{k}</span><b className={toneClass(tone)}>{v}</b></div>
);

export const Stat = ({ label, value, unit, tone }: { label: string; value: string; unit?: string; tone?: Tone }) => (
  <div className="stat">
    <strong className={toneClass(tone)}>{value}{unit && <small>{unit}</small>}</strong>
    <span>{label}</span>
  </div>
);

export const Legend = ({ items, top, right }: { items: Array<[string, string]>; top?: boolean; right?: ReactNode }) => (
  <div className={top ? 'legend top' : 'legend'}>
    {items.map(([color, label]) => <span key={label}><i style={{ background: color }} />{label}</span>)}
    {right && <span className="legend-right">{right}</span>}
  </div>
);

export const Empty = ({ children }: { children: ReactNode }) => <div className="empty">{children}</div>;

const ICON = { viewBox: '0 0 16 16', width: 14, height: 14, 'aria-hidden': true } as const;

export const PlayIcon = () => (
  <svg {...ICON}><path d="M4.5 2.8v10.4a.8.8 0 0 0 1.2.7l8.4-5.2a.8.8 0 0 0 0-1.4L5.7 2.1a.8.8 0 0 0-1.2.7z" fill="currentColor" /></svg>
);

export const PauseIcon = () => (
  <svg {...ICON}><rect x="3.5" y="2.5" width="3" height="11" rx="1" fill="currentColor" /><rect x="9.5" y="2.5" width="3" height="11" rx="1" fill="currentColor" /></svg>
);

export const ReplayIcon = () => (
  <svg {...ICON}><path d="M8 2.5a5.5 5.5 0 1 1-5.2 3.7" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /><path d="M2.2 2.6v3.9h3.9" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
);

export const StartIcon = () => (
  <svg {...ICON}><rect x="2.5" y="2.5" width="2" height="11" rx="1" fill="currentColor" /><path d="M13.5 2.8v10.4a.8.8 0 0 1-1.2.7L5.6 8.7a.8.8 0 0 1 0-1.4l6.7-5.2a.8.8 0 0 1 1.2.7z" fill="currentColor" /></svg>
);
