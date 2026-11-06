import { useEffect, useRef, useState } from "react";
import { STEMS, cancelJob, fetchManifest, fetchQa, fetchTempo, previewUrl, qaBadge, stemUrl, subscribeJob, tempoReadout, uploadJob, type JobStatus, type ManifestDoc, type QaDoc, type Stem, type TempoDoc } from "./api";

interface StemState {
  solo: boolean;
  muted: boolean;
  volume: number;
}

const initialStems = (): Record<Stem, StemState> =>
  Object.fromEntries(STEMS.map((s) => [s, { solo: false, muted: false, volume: 0.76 }])) as Record<
    Stem,
    StemState
  >;

export default function App() {
  const [job, setJob] = useState<JobStatus | null>(null);
  const [stems, setStems] = useState(initialStems);
  const [qa, setQa] = useState<QaDoc | null>(null);
  const [manifest, setManifest] = useState<ManifestDoc | null>(null);
  const [tempo, setTempo] = useState<TempoDoc | null>(null);
  const [playing, setPlaying] = useState<Stem | "all" | null>(null);
  const audioRefs = useRef<Record<string, HTMLAudioElement | null>>({});
  const [drag, setDrag] = useState(false);

  useEffect(() => () => Object.values(audioRefs.current).forEach((a) => a?.pause()), []);

  const startUpload = async (f: File | undefined) => {
    if (!f) return;
    const { job_id } = await uploadJob(f);
    setJob({ job_id, status: "queued", progress: 0 });
    setQa(null);
    setManifest(null);
    setTempo(null);
    subscribeJob(job_id, (s) => {
      setJob(s);
      if (s.status === "done") {
        void fetchQa(job_id).then(setQa);
        void fetchManifest(job_id).then(setManifest);
        void fetchTempo(job_id).then(setTempo);
      }
    });
  };

  const togglePlay = (s: Stem | "all") => {
    if (playing === s) {
      Object.values(audioRefs.current).forEach((a) => a?.pause());
      setPlaying(null);
      return;
    }
    const anySolo = (Object.keys(stems) as Stem[]).some((k) => stems[k].solo);
    const audible = (Object.keys(stems) as Stem[]).filter((k) =>
      s === "all" ? (anySolo ? stems[k].solo && !stems[k].muted : !stems[k].muted) : k === s && !stems[k].muted,
    );
    Object.entries(audioRefs.current).forEach(([k, a]) => {
      if (!a) return;
      if (audible.includes(k as Stem)) {
        a.volume = stems[k as Stem].volume;
        void a.play();
      } else a.pause();
    });
    setPlaying(s);
  };

  const flip = (s: Stem, key: "solo" | "muted") =>
    setStems((p) => ({ ...p, [s]: { ...p[s], [key]: !p[s][key] } }));

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">≋</span>
          <span className="brand-name">Stem Separator</span>
          <span className="workspace-name">Audio workspace</span>
        </div>
        <div className="header-right">
          <span className="engine-label">9-STEM ENGINE / WAV · MP3</span>
          {(() => {
            const readout = tempoReadout(tempo);
            return readout ? <span className="tempo-readout">♩ {readout}</span> : null;
          })()}
          <span className="job-state">
            <span className={`dot ${job && job.status !== "done" ? "working" : ""}`} />
            {job ? `${job.status} ${Math.round(job.progress * 100)}%` : "Ready"}
          </span>
        </div>
      </header>
      <main>
        <h1>Separate your mix</h1>
        <p className="intro">Split a track into nine editable stems, then audition and export each layer.</p>
        <section className="upload-row">
          <label
            className={`dropzone${drag ? " dragging" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); void startUpload(e.dataTransfer.files[0]); }}
          >
            <input type="file" accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg" hidden
              onChange={(e) => void startUpload(e.target.files?.[0])} />
            <span>Drop an audio file or <strong>browse files</strong></span>
            <small>WAV · MP3 · FLAC · M4A — up to ~10 min</small>
          </label>
          {job && job.status !== "done" && (
            <div className="progress">
              <div className="bar" style={{ width: `${job.progress * 100}%` }} />
              <button onClick={() => { void cancelJob(job.job_id); setJob(null); }}>Cancel</button>
              {job.status === "error" && <span className="err">{job.error}</span>}
            </div>
          )}
        </section>
        <section className="stems">
          <div className="stems-head">
            <h2>Separated stems <span className="pill">09</span></h2>
            <button disabled={!job || job.status !== "done"} onClick={() => togglePlay("all")}>
              {playing === "all" ? "Stop all" : "Play all"}
            </button>
          </div>
          {STEMS.map((s) => (
            <div className="stem-row" key={s}>
              <span className="stem-name">{s}{qa && (
                <span className={`qa-badge${qa.clips.find((c) => c.stem === s)?.flagged ? " clip" : ""}${manifest && !manifest.stems[s]?.present ? " absent" : ""}`} title={`SI-SDR estimate, ${qa.stems[s]?.runs ?? 0} runs, spread ${qa.stems[s]?.spread_db ?? 0} dB, limiter ${qa.stems[s]?.limiter ?? "none"}${manifest ? `, manifest present=${manifest.stems[s]?.present} conf=${manifest.stems[s]?.confidence}` : ""}`}>{qaBadge(qa.stems[s], qa.clips.find((c) => c.stem === s), manifest?.stems[s])}</span>
              )}</span>
              <div className="stem-btns">
                <button disabled={!job || job.status !== "done"} onClick={() => togglePlay(s)}>
                  {playing === s ? "Stop" : "Play"}
                </button>
                <button className={stems[s].solo ? "on" : ""} onClick={() => flip(s, "solo")}>S</button>
                <button className={stems[s].muted ? "on" : ""} onClick={() => flip(s, "muted")}>M</button>
              </div>
              <input type="range" min={0} max={1} step={0.01} value={stems[s].volume}
                onChange={(e) => setStems((p) => ({ ...p, [s]: { ...p[s], volume: Number(e.target.value) } }))} />
              <div className="exports">
                {job && job.status === "done" ? (
                  <>
                    <a href={stemUrl(job.job_id, s)} download={`${s}.wav`}>WAV</a>
                    <a href={previewUrl(job.job_id, s, "mp3")}>Preview</a>
                    <audio ref={(el) => { audioRefs.current[s] = el; }} src={previewUrl(job.job_id, s)} preload="none" />
                  </>
                ) : <span className="na">—</span>}
              </div>
            </div>
          ))}
        </section>
      </main>
    </div>
  );
}
