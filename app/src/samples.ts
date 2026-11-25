import batchMetrics from '../../examples/batch.metrics.json?url';
import batchTrace from '../../examples/batch.trace.json?url';
import prefixMetrics from '../../examples/prefix.metrics.json?url';
import prefixTrace from '../../examples/prefix.trace.json?url';

export interface Sample {
  name: string;
  label: string;
  files: Record<string, string>;
}

/* Runs from examples/, recorded against tests/mock_vllm.py and emitted next to the build so the
 * viewer can be tried without recording one. `?sample=<name>` opens one on load */
export const SAMPLES: Sample[] = [
  { name: 'prefix', label: 'Prefix caching · 12 requests',
    files: { 'prefix.trace.json': prefixTrace, 'prefix.metrics.json': prefixMetrics } },
  { name: 'batch', label: 'Continuous batching · 8 requests',
    files: { 'batch.trace.json': batchTrace, 'batch.metrics.json': batchMetrics } },
];

/* As files under their original names, so a sample goes through the same validation as a drop */
export async function fetchSample(name: string): Promise<File[]> {
  const sample = SAMPLES.find((s) => s.name === name);
  if (!sample) {
    throw new Error(`no sample named "${name}" (one of ${SAMPLES.map((s) => s.name).join(', ')})`);
  }
  return Promise.all(Object.entries(sample.files).map(async ([file, url]) => {
    const res = await fetch(url);
    if (!res.ok) { throw new Error(`${file}: HTTP ${res.status}`); }
    return new File([await res.blob()], file);
  }));
}
