import { useState, useRef, useEffect, useCallback } from "react";

const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws/query";
const BUDGET_MS = 200;

const LOCAL = new Set(["safety", "embed", "search", "fetch", "relevance", "grounding"]);
const STAGE_LABEL = {
  stt: "Speech to text",
  safety: "Safety check",
  embed: "Embed question",
  search: "Vector search",
  fetch: "Fetch passages",
  relevance: "Relevance gate",
  generate: "First token",
  grounding: "Grounding check",
};

const pct = (arr, p) => {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
};

function Waterfall({ stages, totalMs }) {
  if (!stages.length) {
    return <div className="wf-empty">Ask something. Each stage appears here as it finishes.</div>;
  }
  // Scale from the bars themselves. totalMs is 0 while stages stream in, and
  // even once `done` lands it covers only the pipeline — `stt` is timed in the
  // websocket handler outside run_pipeline, so barsTotal legitimately exceeds
  // totalMs. Taking the max of both is correct in every phase.
  const barsTotal = stages.reduce((a, s) => a + s.ms, 0);
  const scale = Math.max(barsTotal, totalMs, BUDGET_MS * 1.25);
  let offset = 0;
  const rows = stages.map((s) => {
    const row = { ...s, start: offset };
    offset += s.ms;
    return row;
  });
  const localTotal = rows.filter((r) => LOCAL.has(r.stage)).reduce((a, r) => a + r.ms, 0);

  return (
    <div className="wf">
      <div className="wf-rows">
        {/* Overlay spans exactly the track column, so the dashed line and the
            bars share one coordinate space and cannot drift apart. */}
        <div className="wf-overlay">
          <span className="wf-budget"
                style={{ left: `${Math.min((BUDGET_MS / scale) * 100, 100)}%` }}>
            <span className="wf-budget-label">200ms target</span>
          </span>
        </div>
        {rows.map((r) => {
          const left = Math.min((r.start / scale) * 100, 100);
          const width = Math.min(Math.max((r.ms / scale) * 100, 0.4), 100 - left);
          return (
            <div className="wf-row" key={r.stage}>
              <div className="wf-name">{STAGE_LABEL[r.stage] || r.stage}</div>
              <div className="wf-track">
                <div
                  className={`wf-bar ${LOCAL.has(r.stage) ? "is-local" : "is-remote"}`}
                  style={{ left: `${left}%`, width: `${width}%` }}
                />
              </div>
              <div className="wf-ms">{r.ms.toFixed(1)}</div>
            </div>
          );
        })}
      </div>
      <div className="wf-sums">
        <div className={localTotal <= BUDGET_MS ? "sum ok" : "sum over"}>
          <span>Retrieval core</span><strong>{localTotal.toFixed(1)} ms</strong>
        </div>
        <div className="sum">
          <span>Full pipeline</span><strong>{totalMs.toFixed(1)} ms</strong>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [connected, setConnected] = useState(false);
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const [typed, setTyped] = useState("");
  const [transcript, setTranscript] = useState("");
  const [answer, setAnswer] = useState("");
  const [status, setStatus] = useState("idle");
  const [refusal, setRefusal] = useState(null);
  const [error, setError] = useState(null);
  const [stages, setStages] = useState([]);
  const [totalMs, setTotalMs] = useState(0);
  const [passages, setPassages] = useState([]);
  const [history, setHistory] = useState([]);

  const ws = useRef(null);
  const recorder = useRef(null);
  const chunks = useRef([]);

  const handleEvent = useCallback((ev) => {
    switch (ev.type) {
      case "stage":
        setStages((s) => [...s.filter((x) => x.stage !== ev.stage),
                          { stage: ev.stage, ms: ev.ms, detail: ev.detail }]);
        break;
      case "transcript": setTranscript(ev.text); break;
      case "passages": setPassages(ev.passages); break;
      case "token":
        setStatus("streaming");
        setAnswer((a) => a + ev.text);
        break;
      case "refusal":
        setRefusal({ reason: ev.reason, message: ev.message });
        setStatus((p) => (p === "streaming" || p === "done" ? "unverified" : "refused"));
        break;
      case "done":
        setTotalMs(ev.total_ms);
        setBusy(false);
        setStatus((p) => (p === "streaming" ? "done" : p));
        setHistory((h) => [...h, ev.total_ms].slice(-100));
        break;
      case "error":
        setError(ev.message); setBusy(false); setStatus("idle");
        break;
      default: break;
    }
  }, []);

  useEffect(() => {
    let alive = true, retry;
    const connect = () => {
      const sock = new WebSocket(WS_URL);
      ws.current = sock;
      sock.onopen = () => alive && setConnected(true);
      sock.onclose = () => {
        if (!alive) return;
        setConnected(false);
        retry = setTimeout(connect, 2000);
      };
      sock.onerror = () => sock.close();
      sock.onmessage = (e) => handleEvent(JSON.parse(e.data));
    };
    connect();
    return () => { alive = false; clearTimeout(retry); ws.current?.close(); };
  }, [handleEvent]);

  const reset = () => {
    setStages([]); setTotalMs(0); setAnswer(""); setTranscript("");
    setRefusal(null); setError(null); setPassages([]); setStatus("idle");
  };

  const askText = () => {
    const t = typed.trim();
    if (!t || !connected || busy) return;
    reset(); setTranscript(t); setBusy(true);
    ws.current.send(JSON.stringify({ type: "text", text: t }));
    setTyped("");
  };

  const startRec = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunks.current = [];
      const mr = new MediaRecorder(stream, { mimeType: "audio/webm" });
      mr.ondataavailable = (e) => e.data.size && chunks.current.push(e.data);
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const buf = await new Blob(chunks.current, { type: "audio/webm" }).arrayBuffer();
        let bin = "";
        new Uint8Array(buf).forEach((b) => (bin += String.fromCharCode(b)));
        reset(); setBusy(true);
        ws.current.send(JSON.stringify({ type: "audio", data: btoa(bin) }));
      };
      recorder.current = mr;
      mr.start();
      setRecording(true);
    } catch {
      setError("Microphone blocked. Allow access, or type your question below.");
    }
  };

  const stopRec = () => { recorder.current?.stop(); setRecording(false); };

  const p50 = pct(history, 50), p70 = pct(history, 70), p100 = pct(history, 100);

  return (
    <div className="app">
      <style>{CSS}</style>
      <header className="head">
        <div className="mark">
          <span className="mark-dot" data-on={connected} />
          <span className="mark-name">वाणी<span className="mark-sub">/ voice rag</span></span>
        </div>
        <div className="head-meta">1,485,330 passages · Hindi · Sarvam + HNSW + Groq</div>
      </header>

      <main className="grid">
        <section className="ask">
          <button className={`mic ${recording ? "is-rec" : ""}`}
                  onClick={recording ? stopRec : startRec}
                  disabled={!connected || busy}>
            <span className="mic-ring" />
            {recording ? "Stop and send" : "Hold a question"}
          </button>

          <div className="typebar">
            <input value={typed} onChange={(e) => setTyped(e.target.value)}
                   onKeyDown={(e) => e.key === "Enter" && askText()}
                   placeholder="or type — भारत की राजधानी क्या है"
                   disabled={!connected || busy} />
            <button onClick={askText} disabled={!connected || busy || !typed.trim()}>Ask</button>
          </div>

          {transcript && (
            <div className="heard"><span className="eyebrow">Heard</span><p>{transcript}</p></div>
          )}
          {error && <div className="alert">{error}</div>}
          {status === "refused" && refusal && (
            <div className="refusal">
              <span className="eyebrow">Declined · {refusal.reason}</span>
              <p>{refusal.message}</p>
            </div>
          )}
          {answer && status !== "refused" && (
            <div className={`answer ${status === "unverified" ? "is-unverified" : ""}`}>
              <span className="eyebrow">Answer{status === "streaming" ? " · writing" : ""}</span>
              <p>{answer}</p>
              {status === "unverified" && (
                <div className="unverified-note">
                  Not fully supported by the retrieved passages. Treat with caution.
                </div>
              )}
            </div>
          )}
          {passages.length > 0 && (
            <details className="sources">
              <summary>{passages.length} passages retrieved</summary>
              {passages.map((p) => (
                <div className="src" key={p.doc_id}>
                  <span className="src-score">{p.score.toFixed(3)}</span>
                  <p>{p.text}</p>
                </div>
              ))}
            </details>
          )}
        </section>

        <aside className="panel">
          <div className="panel-head"><span className="eyebrow">Where the time goes</span></div>
          <Waterfall stages={stages} totalMs={totalMs} />
          <div className="legend">
            <span><i className="sw local" /> our machine</span>
            <span><i className="sw remote" /> external API</span>
          </div>
          <div className="percentiles">
            <span className="eyebrow">This session · {history.length} queries</span>
            <div className="prow"><span>P50</span><strong>{p50.toFixed(0)}<em>ms</em></strong></div>
            <div className="prow"><span>P70</span><strong>{p70.toFixed(0)}<em>ms</em></strong></div>
            <div className="prow"><span>P100</span><strong>{p100.toFixed(0)}<em>ms</em></strong></div>
          </div>
        </aside>
      </main>
    </div>
  );
}

/* ==========================================================================
   HH Goa 2026 — neo-brutalist. Cream, pure black, zero-blur offset shadows.
   Every value below comes from the supplied tokens; none are invented.

   Devanagari: Imbue and Victor Mono cover NO Devanagari. Every stack that
   renders user content names 'Noto Sans Devanagari' explicitly, or Hindi
   falls back to tofu boxes.
   ========================================================================== */
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Imbue:opsz,wght@10..100,400;10..100,600;10..100,700&family=Victor+Mono:ital,wght@0,400;0,500;0,700;1,400&family=Noto+Sans+Devanagari:wght@400;500;700&display=swap');

:root {
  --bg:#fffbe8; --fg:#000000;
  --primary:#4c9d6a; --primary-hi:#18e780;
  --accent:#ff0080; --yellow:#fee101;
  --muted:#eeebdf; --muted-fg:#8d8a78;
  --line:#e1ded0;
  --destructive:#ef4444; --warning:#f59e0b;
  --r:0.375rem;
  --sh:rgba(0,0,0,0.25) 6px 8px 0px 0px;
  --sh-lg:rgba(0,0,0,0.25) 8px 10px 0px 0px;
  --sh-xs:rgba(0,0,0,0.25) 3px 4px 0px 0px;

  --ui:'Victor Mono','Noto Sans Devanagari',ui-monospace,monospace;
  --display:'Imbue','Noto Sans Devanagari',Georgia,serif;
}

*,*::before,*::after{box-sizing:border-box}
html,body,#root{margin:0;padding:0;min-height:100%}
body{background:var(--bg)}

.app{
  min-height:100vh;background:var(--bg);color:var(--fg);
  font-family:var(--ui);font-size:0.875rem;line-height:1.55;
  padding:0 0 3rem;
}
.app *:focus-visible{outline:2px solid var(--fg);outline-offset:2px}

/* ---------------------------------------------------------------- header */
.head{
  display:flex;flex-wrap:wrap;gap:0.75rem 1.5rem;
  align-items:center;justify-content:space-between;
  padding:1.25rem 1.5rem;border-bottom:2px solid var(--fg);
  background:var(--bg);
}
.mark{display:flex;align-items:center;gap:0.625rem;min-width:0}
.mark-dot{
  width:0.75rem;height:0.75rem;flex:none;border-radius:50%;
  border:2px solid var(--fg);background:var(--muted);
}
.mark-dot[data-on="true"]{background:var(--primary)}
.mark-name{
  font-family:var(--display);font-size:2rem;font-weight:700;line-height:1;
  letter-spacing:-0.01em;display:flex;align-items:baseline;gap:0.5rem;
  flex-wrap:wrap;
}
.mark-sub{
  font-family:var(--ui);font-size:0.6875rem;font-weight:500;
  letter-spacing:0.06em;text-transform:uppercase;color:var(--muted-fg);
}
.head-meta{font-size:0.6875rem;letter-spacing:0.04em;color:var(--muted-fg)}

/* ------------------------------------------------------------------ grid */
.grid{
  display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);
  gap:1.5rem;padding:1.5rem;max-width:1400px;margin:0 auto;align-items:start;
}
.ask{display:flex;flex-direction:column;gap:1rem;min-width:0}

/* ------------------------------------------------------------------- mic */
.mic{
  position:relative;display:flex;align-items:center;justify-content:center;
  gap:0.75rem;width:100%;padding:1.25rem 1.5rem;
  font-family:var(--ui);font-size:1rem;font-weight:700;
  letter-spacing:0.02em;color:var(--fg);background:var(--primary);
  border:2px solid var(--fg);border-radius:var(--r);box-shadow:var(--sh);
  cursor:pointer;transition:transform .08s steps(2),box-shadow .08s steps(2);
}
.mic:hover:not(:disabled){background:var(--primary-hi)}
.mic:active:not(:disabled){transform:translate(4px,5px);box-shadow:var(--sh-xs)}
.mic:disabled{background:var(--muted);color:var(--muted-fg);cursor:not-allowed;box-shadow:none;transform:translate(6px,8px)}
.mic.is-rec{background:var(--accent);color:#fff}
.mic-ring{
  width:0.875rem;height:0.875rem;flex:none;border-radius:50%;
  border:2px solid var(--fg);background:var(--bg);
}
.mic.is-rec .mic-ring{background:#fff;animation:blink 1s steps(2,end) infinite}
@keyframes blink{50%{opacity:.2}}

/* --------------------------------------------------------------- typebar */
.typebar{display:flex;gap:0.75rem}
.typebar input{
  flex:1;min-width:0;padding:0.75rem 0.875rem;
  font-family:var(--ui);font-size:0.875rem;color:var(--fg);
  background:var(--bg);border:2px solid var(--fg);border-radius:var(--r);
  box-shadow:var(--sh-xs);
}
.typebar input::placeholder{color:var(--muted-fg)}
.typebar input:disabled{background:var(--muted);color:var(--muted-fg);box-shadow:none}
.typebar button{
  padding:0.75rem 1.25rem;font-family:var(--ui);font-size:0.875rem;font-weight:700;
  color:var(--fg);background:var(--bg);border:2px solid var(--fg);
  border-radius:var(--r);box-shadow:var(--sh-xs);cursor:pointer;
  transition:transform .08s steps(2),box-shadow .08s steps(2);
}
.typebar button:active:not(:disabled){transform:translate(3px,4px);box-shadow:none}
.typebar button:disabled{background:var(--muted);color:var(--muted-fg);cursor:not-allowed;box-shadow:none}

/* ----------------------------------------------------------- content cards */
.eyebrow{
  display:inline-block;font-size:0.625rem;font-weight:700;letter-spacing:0.1em;
  text-transform:uppercase;color:var(--muted-fg);margin-bottom:0.375rem;
}
.heard,.answer,.refusal,.alert,.sources{
  padding:1rem;background:var(--bg);border:2px solid var(--fg);
  border-radius:var(--r);box-shadow:var(--sh);
}
.heard p,.answer p,.refusal p{
  margin:0;font-family:var(--ui);font-size:1rem;line-height:1.65;
}
.heard{background:var(--muted)}
.answer p{font-size:1.125rem}
.answer.is-unverified{border-color:var(--warning);box-shadow:rgba(245,158,11,.35) 6px 8px 0px 0px}
.unverified-note{
  margin-top:0.75rem;padding:0.5rem 0.625rem;font-size:0.75rem;
  border:2px solid var(--warning);border-radius:var(--r);background:var(--bg);
}
.refusal{border-color:var(--destructive);box-shadow:rgba(239,68,68,.35) 6px 8px 0px 0px}
.refusal .eyebrow{color:var(--destructive)}
.alert{border-color:var(--destructive);background:var(--muted);font-size:0.8125rem}

/* --------------------------------------------------------------- sources */
.sources summary{
  cursor:pointer;font-size:0.75rem;font-weight:700;letter-spacing:0.06em;
  text-transform:uppercase;color:var(--muted-fg);
}
.sources summary::marker{color:var(--fg)}
.src{
  display:flex;gap:0.75rem;padding:0.75rem 0;border-top:1px solid var(--line);
  margin-top:0.75rem;
}
.src-score{
  flex:none;font-size:0.6875rem;font-weight:700;padding:0.125rem 0.375rem;
  height:fit-content;border:2px solid var(--fg);border-radius:var(--r);
  background:var(--muted);
}
.src p{margin:0;font-family:var(--ui);font-size:0.8125rem;line-height:1.6}

/* ----------------------------------------------------------------- panel */
.panel{
  padding:1.25rem;background:var(--bg);border:2px solid var(--fg);
  border-radius:var(--r);box-shadow:var(--sh-lg);position:sticky;top:1.5rem;
}
.panel-head{margin-bottom:0.75rem}

.wf-empty{
  padding:1.5rem 0.5rem;font-size:0.8125rem;color:var(--muted-fg);
  border:2px dashed var(--line);border-radius:var(--r);text-align:center;
}
.wf{position:relative}
/* Rows wrapper is the positioning context. The overlay is inset to exactly the
   track column (label width + gap on the left, ms width + gap on the right) so
   a percentage inside it means the same thing as a percentage inside .wf-track. */
.wf-rows{position:relative;padding-top:1.25rem}
.wf-overlay{position:absolute;top:0;bottom:0;left:7.5rem;right:3.25rem;pointer-events:none;z-index:3}
.wf-budget{position:absolute;top:0;bottom:0;border-left:2px dashed var(--fg)}
.wf-budget-label{
  position:absolute;top:0;left:0.25rem;white-space:nowrap;
  font-size:0.5625rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;
  background:var(--fg);color:var(--bg);padding:0 0.3125rem;border-radius:2px;
}
.wf-row{display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3125rem}
.wf-name{
  width:7rem;flex:none;font-size:0.6875rem;letter-spacing:0.02em;
  color:var(--fg);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.wf-track{
  position:relative;flex:1;min-width:0;height:1.125rem;overflow:hidden;
  background:var(--muted);border:1px solid var(--line);border-radius:2px;
}
.wf-bar{position:absolute;top:-1px;bottom:-1px;border:1px solid var(--fg);border-radius:2px;min-width:3px}
.wf-bar.is-local{background:var(--primary)}
.wf-bar.is-remote{background:var(--accent)}
.wf-ms{
  width:2.75rem;flex:none;text-align:right;font-size:0.6875rem;
  font-variant-numeric:tabular-nums;color:var(--muted-fg);
}

.wf-sums{display:flex;flex-wrap:wrap;gap:0.75rem;margin-top:1rem}
.sum{
  flex:1;min-width:9rem;display:flex;flex-direction:column;gap:0.125rem;
  padding:0.625rem 0.75rem;border:2px solid var(--fg);border-radius:var(--r);
  background:var(--bg);box-shadow:var(--sh-xs);
}
.sum span{font-size:0.625rem;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted-fg)}
.sum strong{font-size:1.25rem;font-weight:700;font-variant-numeric:tabular-nums}
.sum.ok{background:var(--yellow)}
.sum.ok span{color:var(--fg)}
.sum.over{border-color:var(--destructive)}

.legend{display:flex;gap:1.25rem;margin-top:0.875rem;font-size:0.6875rem;color:var(--muted-fg)}
.legend span{display:flex;align-items:center;gap:0.375rem}
.sw{width:0.75rem;height:0.75rem;display:inline-block;border:1px solid var(--fg);border-radius:2px}
.sw.local{background:var(--primary)}
.sw.remote{background:var(--accent)}

.percentiles{
  margin-top:1.25rem;padding-top:1rem;border-top:2px solid var(--fg);
}
.prow{
  display:flex;align-items:baseline;justify-content:space-between;
  padding:0.3125rem 0;border-bottom:1px solid var(--line);
}
.prow span{font-size:0.6875rem;letter-spacing:0.08em;color:var(--muted-fg)}
.prow strong{font-size:1.0625rem;font-weight:700;font-variant-numeric:tabular-nums}
.prow em{font-style:normal;font-size:0.625rem;color:var(--muted-fg);margin-left:0.125rem}

/* ------------------------------------------------------------ responsive */
@media (max-width:900px){
  .grid{grid-template-columns:minmax(0,1fr)}
  .panel{position:static}
}
@media (max-width:520px){
  .head{padding:1rem}
  .grid{padding:1rem;gap:1rem}
  .mark-name{font-size:1.625rem}
  .wf-overlay{left:5.25rem;right:3.25rem}
  .wf-name{width:4.75rem;font-size:0.625rem}
  .typebar{flex-wrap:wrap}
  .typebar input{flex:1 1 100%}
  .typebar button{flex:1 1 100%}
}
@media (max-width:380px){
  .mark-sub{display:none}
  .wf-overlay{left:4.25rem;right:3.25rem}
  .wf-name{width:3.75rem}
  .sum{min-width:100%}
}

@media (prefers-reduced-motion:reduce){
  *{animation:none!important;transition:none!important}
  .mic:active:not(:disabled){transform:none}
}
`;
