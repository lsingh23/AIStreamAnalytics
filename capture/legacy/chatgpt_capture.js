// ==UserScript==
// @name         ChatGPT stream capture
// @namespace    ai-stream-analytics
// @version      1.1
// @description  Saves ChatGPT conversation streams (SSE + WebSocket) and stored conversation JSON
// @match        https://chatgpt.com/*
// @run-at       document-start
// @grant        none
// ==/UserScript==

// ChatGPT stream capture — install as a Tampermonkey userscript (the header above), or paste into the
// DevTools Console on chatgpt.com BEFORE sending a prompt. The userscript route is preferred: ChatGPT may
// open its WebSocket at page load, before a console paste. @grant none makes it run in the page's own
// context, so it patches the same fetch/WebSocket the app uses.
//
// For every conversation request it downloads two files:
//   chatgpt_<ts>.sse   raw response bytes exactly as the page received them
//   chatgpt_<ts>.meta.json   request URL/body + response status/headers (no auth headers)
// It also saves GET /backend-api/conversation/<id> responses (the stored conversation JSON
// you get after reloading the page) as chatgpt_<ts>.conversation.json.
//
// Long turns (Thinking, many searches) can move off the HTTP stream mid-answer: the stream carries
// "resume_with_websockets": true and a resume_conversation_token, and the rest of the turn arrives over
// a WebSocket or a separate resume request. So it also:
//   - captures ANY /backend-api/ response that is an event stream (catches resume endpoints by type,
//     not by guessing their path)
//   - records WebSocket traffic as chatgpt_<ts>.ws.jsonl (one {t, dir, data} line per frame), saved when
//     the socket closes, when a turn completes, or on demand with window.__captureFlush()
//   - checks each .sse for an end-of-turn marker and flags it as incomplete in meta.json if missing
// Every /backend-api/ request seen is listed in window.__captureUrls, to find new endpoints.
(() => {
  if (window.__retrievalCapture) return console.log('[capture] already installed');
  window.__retrievalCapture = true;

  const STREAM_RE = /\/backend-api\/(f\/)?conversation(\?|$)/;
  const STORED_RE = /\/backend-api\/conversation\/[0-9a-f-]{36}(\?|$)/;
  const BACKEND_RE = /\/backend-api\//;
  const END_RE = /\[DONE\]|message_stream_complete/; // loose: WebSocket frames may wrap SSE text in escaped JSON
  const origFetch = window.fetch;
  const OrigWebSocket = window.WebSocket;
  const stamp = () => new Date().toISOString().replace(/[:.]/g, '-');
  window.__captureUrls = [];

  // Chrome silently drops back-to-back programmatic downloads (and asks once to "allow multiple
  // downloads" for the site), so downloads are queued and spaced out.
  let queue = Promise.resolve();
  const download = (name, blob) => {
    queue = queue.then(() => new Promise((resolve) => {
      const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: name });
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(resolve, 1000);
    }));
  };

  // The page aborts its own fetch (AbortController) once it has seen the end-of-stream marker, which
  // also errors our tee'd branch. Keep whatever arrived instead of throwing it away.
  const readAll = async (stream) => {
    const reader = stream.getReader();
    const chunks = [];
    let aborted = null;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
      }
    } catch (e) {
      aborted = e?.name || String(e);
    }
    const blob = new Blob(chunks);
    blob.aborted = aborted;
    return blob;
  };

  window.fetch = async (...args) => {
    const req = args[0];
    const init = args[1] || {};
    const url = typeof req === 'string' ? req : req?.url || String(req);
    const method = (init.method || req?.method || 'GET').toUpperCase();
    const res = await origFetch(...args);

    if (BACKEND_RE.test(url)) window.__captureUrls.push({ t: new Date().toISOString(), method, url, status: res.status, type: res.headers.get('content-type') });

    const isEventStream = BACKEND_RE.test(url) && (res.headers.get('content-type') || '').includes('text/event-stream');
    const isStream = (method === 'POST' && STREAM_RE.test(url)) || isEventStream;
    const isStored = !isStream && method === 'GET' && STORED_RE.test(url);
    if (!(isStream || isStored) || !res.body) return res;

    const ts = stamp();
    const [forPage, forUs] = res.body.tee();

    (async () => {
      const blob = await readAll(forUs);
      if (blob.aborted) console.warn(`[capture] stream ended with ${blob.aborted} after ${blob.size} bytes — saving what arrived`);
      if (!blob.size) return;
      if (isStored) {
        download(`chatgpt_${ts}.conversation.json`, blob);
        console.log('[capture] saved stored conversation', blob.size, 'bytes');
        return;
      }
      // A clean close without the end marker means the turn continued elsewhere (WebSocket / resume
      // request) — the main way a capture ends up silently partial.
      const text = await blob.text();
      const complete = END_RE.test(text.slice(-8192));
      const handoff = /"resume_with_websockets":\s*true/.test(text);
      if (!complete) console.warn(`[capture] ${url} closed WITHOUT an end-of-turn marker — partial stream${handoff ? '; the rest is likely in the .ws.jsonl / resume capture' : ''}`);
      let body = null;
      try { body = typeof init.body === 'string' ? JSON.parse(init.body) : null; } catch { body = init.body ?? null; }
      const meta = {
        platform: 'chatgpt',
        captured_at: new Date().toISOString(),
        request: { url, method, body },
        response: {
          status: res.status, headers: Object.fromEntries(res.headers.entries()), aborted: blob.aborted, bytes: blob.size,
          complete, resume_with_websockets: handoff, is_resume_stream: !STREAM_RE.test(url),
        },
      };
      download(`chatgpt_${ts}.sse`, blob);
      download(`chatgpt_${ts}.meta.json`, new Blob([JSON.stringify(meta, null, 2)], { type: 'application/json' }));
      console.log('[capture] saved stream', blob.size, 'bytes', complete ? '(complete)' : '(INCOMPLETE)');
    })().catch((e) => console.error('[capture] failed', e));

    return new Response(forPage, { status: res.status, statusText: res.statusText, headers: res.headers });
  };

  // WebSocket capture. Frames are recorded in arrival order (binary frames are decoded as UTF-8 text);
  // a socket can stay open across turns, so the buffer is also flushed whenever a turn ends.
  const sockets = new Set();
  const flushSocket = (s, reason) => {
    if (!s.frames.length) return;
    const lines = s.frames.map((f) => JSON.stringify(f)).join('\n') + '\n';
    s.frames = [];
    download(`chatgpt_${stamp()}.ws.jsonl`, new Blob([lines], { type: 'application/x-ndjson' }));
    console.log(`[capture] saved websocket frames from ${s.url} (${reason})`);
  };
  window.__captureFlush = () => sockets.forEach((s) => flushSocket(s, 'manual flush'));

  if (OrigWebSocket) {
    const decoder = new TextDecoder();
    const toText = async (data) => {
      if (typeof data === 'string') return data;
      if (data instanceof Blob) return data.text();
      if (data instanceof ArrayBuffer || ArrayBuffer.isView(data)) return decoder.decode(data);
      return String(data);
    };

    window.WebSocket = class extends OrigWebSocket {
      constructor(...args) {
        super(...args);
        const s = { url: String(args[0]), frames: [] };
        sockets.add(s);
        // Blob frames decode asynchronously; chain them so frame order is preserved.
        let order = Promise.resolve();
        const record = (dir, data) => {
          const t = new Date().toISOString();
          order = order.then(async () => {
            const text = await toText(data);
            s.frames.push({ t, dir, data: text });
            if (dir === 'in' && END_RE.test(text)) flushSocket(s, 'turn complete');
          }).catch((e) => console.error('[capture] websocket frame failed', e));
        };
        this.__captureRecord = record;
        this.addEventListener('message', (e) => record('in', e.data));
        this.addEventListener('close', () => { order = order.then(() => { flushSocket(s, 'socket closed'); sockets.delete(s); }); });
        console.log('[capture] watching websocket', s.url);
      }

      send(data) {
        this.__captureRecord?.('out', data);
        return super.send(data);
      }
    };
  }

  console.log('[capture] installed — send your prompt now (reload the chat afterwards to also save the stored JSON; window.__captureFlush() saves pending WebSocket frames)');
})();
