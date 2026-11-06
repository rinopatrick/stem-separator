export const STEMS = ["vocal", "drums", "bass", "piano", "guitar", "strings", "synth", "cue", "click"] as const;
export type Stem = (typeof STEMS)[number];
export type DownloadStem = Stem | "residual";

export interface JobStatus {
  job_id: string;
  status: "queued" | "running" | "done" | "error" | "cancelled";
  progress: number;
  stems?: string[];
  error?: string;
}

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function uploadJob(file: File): Promise<{ job_id: string }> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API}/jobs`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`upload failed: ${r.status}`);
  return r.json();
}

export function subscribeJob(jobId: string, onMsg: (s: JobStatus) => void): () => void {
  const es = new EventSource(`${API}/jobs/${jobId}`);
  es.onmessage = (e) => {
    const s = JSON.parse(e.data) as JobStatus;
    onMsg(s);
    if (s.status === "done" || s.status === "error") es.close();
  };
  es.onerror = () => es.close();
  return () => es.close();
}

export function stemUrl(jobId: string, stem: DownloadStem): string {
  return `${API}/jobs/${jobId}/stems/${stem}`;
}

export async function cancelJob(jobId: string): Promise<void> {
  await fetch(`${API}/jobs/${jobId}`, { method: "DELETE" });
}

export interface QaEntry {
  sisdr_db: number;
  spread_db: number;
  runs: number;
  limiter: string;
}

export interface QaClip {
  stem: Stem;
  peak: number;
  flagged: boolean;
}

export interface QaDoc {
  sr: number;
  stems: Record<Stem, QaEntry>;
  leakage: Record<string, number>;
  clips: QaClip[];
}

export function qaUrl(jobId: string): string {
  return `${API}/jobs/${jobId}/sidecars/qa.json`;
}

export function manifestUrl(jobId: string): string {
  return `${API}/jobs/${jobId}/sidecars/manifest.json`;
}

export interface ManifestStem {
  file: string;
  present: boolean;
  confidence: number;
  source: string;
  reason: string;
}

export interface ManifestDoc {
  model: string;
  tile: number;
  sr: number;
  stems: Record<Stem, ManifestStem>;
}

export async function fetchQa(jobId: string): Promise<QaDoc | null> {
  const r = await fetch(qaUrl(jobId));
  if (!r.ok) return null;
  return r.json();
}

export async function fetchManifest(jobId: string): Promise<ManifestDoc | null> {
  const r = await fetch(manifestUrl(jobId));
  if (!r.ok) return null;
  return r.json();
}

export function qaBadge(
  entry: QaEntry | undefined,
  clip: QaClip | undefined,
  manifest: ManifestStem | undefined,
): string {
  if (manifest && !manifest.present) return `absent (conf ${manifest.confidence})`;
  if (!entry) return "n/a";
  if (clip?.flagged) return `clip ${entry.sisdr_db.toFixed(1)} dB`;
  return `${entry.sisdr_db.toFixed(1)} dB`;
}

export async function listJobs(): Promise<{ jobs: JobStatus[] }> {
  const r = await fetch(`${API}/jobs`);
  return r.json();
}
