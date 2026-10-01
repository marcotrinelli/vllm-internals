interface SplashProps {
  hot: boolean;
  errors: string[];
  onPick: () => void;
}

/* Empty state: nothing loaded yet. Files only; this app never talks to a server */
export const Splash = ({ hot, errors, onPick }: SplashProps) => (
  <div className="splash">
    <div className={hot ? 'drop hot' : 'drop'}>
      <h1>vLLM internals</h1>
      <p>
        Drop the two files the notebook wrote, or pick them. Either one alone works: the
        trace gives the request timeline and the token content, the metrics give the KV cache
        and the scheduler over time, and together they give the reconstructed block pool.
      </p>
      <pre>{`runs/<name>.trace.json     vllm-traces/1
runs/<name>.metrics.json   vllm-metrics/1`}</pre>
      <button type="button" className="primary" onClick={onPick}>Open files…</button>
      <p className="fine">
        No server? <code>examples/</code> has a small run recorded against the mock server
        in <code>tests/mock_vllm.py</code>.
      </p>
      {errors.map((e) => <div key={e} className="err">{e}</div>)}
    </div>
  </div>
);
