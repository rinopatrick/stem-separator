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

export async function listJobs(): Promise<{ jobs: JobStatus[] }> {
  const r = await fetch(`${API}/jobs`);
  return r.json();
}
