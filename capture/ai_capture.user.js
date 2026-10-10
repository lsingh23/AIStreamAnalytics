// ==UserScript==
// @name         AI assistant stream capture
// @namespace    ai-stream-analytics
// @version      2.4
// @description  Saves answer streams + stored conversations from ChatGPT, Claude, Perplexity and Gemini
// @match        https://chatgpt.com/*
// @match        https://claude.ai/*
// @match        https://www.perplexity.ai/*
// @match        https://perplexity.ai/*
// @match        https://gemini.google.com/*
// @run-at       document-start
// @grant        none
// ==/UserScript==

// Multi-platform version of legacy/chatgpt_capture.js. Same approach (tee the response so the page and we
// each get a copy, keep partial data on abort, spaced downloads, WebSocket recorder), but every
// platform-specific detail lives in the PLATFORMS table below, and XMLHttpRequest is patched too
// because Gemini streams its answers over XHR, which a fetch patch never sees.
//
// Per answer it downloads:
//   <platform>_<ts>.sse                 raw response body exactly as the page received it
//                                       (SSE for ChatGPT/Claude/Perplexity, )]}'-framed arrays for Gemini)
//   <platform>_<ts>.meta.json           request URL/body + response status/headers (no auth headers)
//   <platform>_<ts>.conversation.json   the stored conversation the app fetches on reload
//   <platform>_<ts>.ws.jsonl            WebSocket frames (ChatGPT only: its resume path)
// Each answer triggers 2+ downloads, so Chrome asks once per site to allow automatic downloads; if that
// prompt is missed, only the first file (.sse) arrives and the .meta.json is silently dropped.
// window.__captureUrls lists every API request, every non-GET request, and every WebSocket / EventSource /
// Worker the page opens, to find the answer channel when a platform changes.
(() => {
  if (window.__retrievalCapture) return console.log('[capture] already installed');
  window.__retrievalCapture = true;

  // api: requests worth logging; stream: answer endpoint; stored: GET that returns the saved conversation;
  // end: end-of-answer marker (null = the platform has none, completeness unknown).
  // ignore: streams that match but are not answers. ws: record WebSockets (only ChatGPT moves answers there;
  // Perplexity's socket carries the sidebar/history, which is private and irrelevant).
  const PLATFORMS = [
    // chatgpt.com (2026-10): the stored conversation moved to GET /backend-api/conversations/<id>?num_turns=10
    // (plural, paged via page_info) and returns a flat `messages` list instead of `mapping`.
    { host: /(^|\.)chatgpt\.com$/, name: 'chatgpt', api: /\/backend-api\//,
      stream: /\/backend-api\/(f\/)?conversation(\?|$)/, stored: /\/backend-api\/conversations?\/[0-9a-f-]{36}(\?|$)/,
      end: /\[DONE\]|message_stream_complete/, ws: true },
    // claude.ai (2026-10): messages go out via Connect-RPC ConversationService/PerformAction and the answer
    // streams back from ConversationService/StreamTimeline; the older REST .../completion SSE is kept as fallback.
    { host: /(^|\.)claude\.ai$/, name: 'claude', api: /\/(api|claudeai-rpc)\//,
      stream: /\/chat_conversations\/[^/?]+\/(retry_)?completion(\?|$)|ConversationService\/(StreamTimeline|PerformAction)/,
      stored: /\/chat_conversations\/[0-9a-f-]{36}(\?|$)/, end: /message_stop/ },
    { host: /(^|\.)perplexity\.ai$/, name: 'perplexity', api: /\/rest\//,
      stream: /\/rest\/sse\//, stored: /\/rest\/thread\/(?!list)[^/?]+/,
      ignore: /\/related-queries\//, end: /"status":\s*"COMPLETED"/ },
    { host: /(^|\.)gemini\.google\.com$/, name: 'gemini', api: /\/_\/BardChatUi\//,
      stream: /StreamGenerate/, stored: /batchexecute\?.*rpcids=[^&]*hNvQHb/, // hNvQHb = read conversation
      end: null },
  ];
  const P = PLATFORMS.find((p) => p.host.test(location.hostname));
  if (!P) return console.log('[capture] no platform config for', location.hostname);
  const STREAM_TYPE_RE = /event-stream|ndjson|x-protobuf|connect\+/;

  const origFetch = window.fetch;
  const OrigWebSocket = window.WebSocket;
  const stamp = () => new Date().toISOString().replace(/[:.]/g, '-');
  window.__captureUrls = [];
  const note = (via, method, url) => {
    const entry = { t: new Date().toISOString(), via, method, url, status: 'pending' };
    window.__captureUrls.push(entry);
    return entry;
  };

  // Chrome silently drops back-to-back programmatic downloads, so downloads are queued and spaced out.
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

  // The page may abort its own request once it has the end marker, which also errors our tee'd branch.
  // Keep whatever arrived instead of throwing it away.
  // Streams still open are kept in `live` so window.__captureFlush() can save them without waiting for
  // the close (long-lived subscription streams such as claude.ai's StreamTimeline may never close).
  const live = new Set();
  const readAll = async (stream, info) => {
    const reader = stream.getReader();
    const s = { ...info, chunks: [] };
    live.add(s);
    let aborted = null;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        s.chunks.push(value);
      }
    } catch (e) {
      aborted = e?.name || String(e);
    }
    live.delete(s);
    const blob = new Blob(s.chunks);
    blob.aborted = aborted;
    return blob;
  };

  // Request bodies may be text (JSON, form-encoded) or binary (protobuf over Connect-RPC).
  const b64 = (bytes) => btoa(Array.from(bytes, (b) => String.fromCharCode(b)).join(''));
  const readBody = async (body) => {
    if (body == null) return null;
    if (body instanceof URLSearchParams) body = body.toString();
    if (body instanceof Blob) body = await body.arrayBuffer();
    if (body instanceof ArrayBuffer || ArrayBuffer.isView(body)) {
      const bytes = body instanceof ArrayBuffer ? new Uint8Array(body) : new Uint8Array(body.buffer, body.byteOffset, body.byteLength);
      try { body = new TextDecoder('utf-8', { fatal: true }).decode(bytes); } catch { return { base64: b64(bytes) }; }
    }
    if (typeof body !== 'string') return String(body);
    try { return JSON.parse(body); } catch { return body; } // Gemini sends form-encoded f.req=...
  };

  // Shared by the fetch and XHR paths. `kind` is 'stream' or 'stored'.
  const save = async (kind, { url, method, body, status, headers, blob, aborted }) => {
    if (!blob.size) return;
    if (kind === 'stream' && aborted && blob.size < 64) return; // a stream the page cancelled before any data
    const ts = stamp();
    if (kind === 'stored') {
      download(`${P.name}_${ts}.conversation.json`, blob);
      console.log('[capture] saved stored conversation', blob.size, 'bytes');
      return;
    }
    const text = await blob.text();
    const complete = P.end ? P.end.test(text.slice(-8192)) : null;
    const handoff = /"resume_with_websockets":\s*true/.test(text);
    if (complete === false) console.warn(`[capture] ${url} closed WITHOUT an end-of-answer marker — partial stream${handoff ? '; the rest is likely in the .ws.jsonl / resume capture' : ''}`);
    const meta = {
      platform: P.name,
      captured_at: new Date().toISOString(),
      page_url: location.href,
      request: { url, method, body: await readBody(body).catch(() => null) },
      response: {
        status, headers, aborted, bytes: blob.size,
        complete, resume_with_websockets: handoff, is_resume_stream: !P.stream.test(url),
      },
    };
    download(`${P.name}_${ts}.sse`, blob);
    download(`${P.name}_${ts}.meta.json`, new Blob([JSON.stringify(meta, null, 2)], { type: 'application/json' }));
    console.log('[capture] saved stream', blob.size, 'bytes', complete === null ? '' : complete ? '(complete)' : '(INCOMPLETE)');
  };

  const classify = (method, url, contentType) => {
    if (P.ignore?.test(url)) return null;
    const isStream = (method === 'POST' && P.stream.test(url)) || STREAM_TYPE_RE.test(contentType || '');
    if (isStream) return 'stream';
    if (method === 'GET' && P.stored.test(url)) return 'stored';
    if (P.name === 'gemini' && P.stored.test(url)) return 'stored'; // batchexecute is always POST
    return null;
  };

  // ---------------------------------------------------------------------------------------------- fetch
  window.fetch = async (...args) => {
    const req = args[0];
    const init = args[1] || {};
    const url = new URL(typeof req === 'string' ? req : req?.url || String(req), location.href).href;
    const method = (init.method || req?.method || 'GET').toUpperCase();
    const isApi = P.api.test(url);
    // Logged before awaiting, so a request that hangs or fails still shows up in __captureUrls.
    const entry = isApi || method !== 'GET' ? note('fetch', method, url) : null;
    if (!isApi) {
      return origFetch(...args).then((r) => { if (entry) Object.assign(entry, { status: r.status, type: r.headers.get('content-type') }); return r; });
    }

    // A Request object carries its own body, which fetch consumes: read a clone first.
    let body = init.body;
    if (body === undefined && req instanceof Request && method !== 'GET') body = await req.clone().arrayBuffer().catch(() => null);

    let res;
    try {
      res = await origFetch(...args);
    } catch (e) {
      entry.status = `failed: ${e?.name || e}`;
      throw e;
    }
    const contentType = res.headers.get('content-type');
    Object.assign(entry, { status: res.status, type: contentType });
    const kind = classify(method, url, contentType);
    if (!kind || !res.body) return res;

    const [forPage, forUs] = res.body.tee();
    (async () => {
      const blob = await readAll(forUs, { url, method, body, status: res.status, headers: Object.fromEntries(res.headers.entries()) });
      if (blob.aborted) console.warn(`[capture] stream ended with ${blob.aborted} after ${blob.size} bytes — saving what arrived`);
      await save(kind, {
        url, method, body, status: res.status, headers: Object.fromEntries(res.headers.entries()), blob, aborted: blob.aborted,
      });
    })().catch((e) => console.error('[capture] failed', e));

    const copy = new Response(forPage, { status: res.status, statusText: res.statusText, headers: res.headers });
    // new Response() resets these; some apps read them.
    Object.defineProperties(copy, { url: { value: res.url }, redirected: { value: res.redirected } });
    return copy;
  };

  // ------------------------------------------------------------------------------------------------ XHR
  const xhrOpen = XMLHttpRequest.prototype.open;
  const xhrSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    this.__capture = { method: String(method).toUpperCase(), url: new URL(String(url), location.href).href };
    return xhrOpen.call(this, method, url, ...rest);
  };
  XMLHttpRequest.prototype.send = function (body) {
    const c = this.__capture;
    if (c && !P.api.test(c.url) && c.method !== 'GET') note('xhr', c.method, c.url);
    if (c && P.api.test(c.url)) {
      const entry = note('xhr', c.method, c.url);
      this.addEventListener('loadend', () => {
        const contentType = this.getResponseHeader('content-type');
        Object.assign(entry, { status: this.status, type: contentType });
        const kind = classify(c.method, c.url, contentType);
        if (!kind) return;
        const text = this.responseType === '' || this.responseType === 'text' ? this.responseText : null;
        const blob = text !== null ? new Blob([text]) : this.response instanceof Blob ? this.response : new Blob([JSON.stringify(this.response)]);
        const headers = Object.fromEntries(this.getAllResponseHeaders().trim().split(/\r?\n/).filter(Boolean)
          .map((l) => [l.slice(0, l.indexOf(':')).trim().toLowerCase(), l.slice(l.indexOf(':') + 1).trim()]));
        save(kind, { url: c.url, method: c.method, body: typeof body === 'string' ? body : null, status: this.status, headers, blob, aborted: this.status === 0 ? 'xhr status 0' : null })
          .catch((e) => console.error('[capture] failed', e));
      });
    }
    return xhrSend.call(this, body);
  };

  // ------------------------------------------------------------------------------------------ WebSocket
  // Frames are recorded in arrival order (binary frames decoded as UTF-8); a socket can stay open across
  // turns, so the buffer is also flushed whenever an end-of-answer marker arrives.
  const sockets = new Set();
  const flushSocket = (s, reason) => {
    if (!s.frames.length) return;
    const lines = s.frames.map((f) => JSON.stringify(f)).join('\n') + '\n';
    s.frames = [];
    download(`${P.name}_${stamp()}.ws.jsonl`, new Blob([lines], { type: 'application/x-ndjson' }));
    console.log(`[capture] saved websocket frames from ${s.url} (${reason})`);
  };
  window.__captureFlush = () => {
    sockets.forEach((s) => flushSocket(s, 'manual flush'));
    live.forEach((s) => save('stream', { ...s, blob: new Blob(s.chunks), aborted: 'still open (manual flush)' }));
    return `${sockets.size} websocket(s), ${live.size} open stream(s) flushed`;
  };

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
        note('websocket', 'OPEN', String(args[0]));
        if (!P.ws) return; // logged only: on most sites the socket is not the answer channel
        const s = { url: String(args[0]), frames: [] };
        sockets.add(s);
        let order = Promise.resolve(); // Blob frames decode asynchronously; keep frame order
        const record = (dir, data) => {
          const t = new Date().toISOString();
          order = order.then(async () => {
            const text = await toText(data);
            s.frames.push({ t, dir, data: text });
            if (dir === 'in' && P.end && P.end.test(text)) flushSocket(s, 'turn complete');
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

  // ------------------------------------------------------------------------- other transports (log only)
  // If an answer never shows up as fetch/xhr, it is coming through one of these; __captureUrls says which.
  if (window.EventSource) {
    const OrigEventSource = window.EventSource;
    window.EventSource = class extends OrigEventSource {
      constructor(url, ...rest) { super(url, ...rest); note('eventsource', 'GET', new URL(String(url), location.href).href); }
    };
  }
  for (const name of ['Worker', 'SharedWorker']) {
    const Orig = window[name];
    if (!Orig) continue;
    window[name] = class extends Orig {
      constructor(url, ...rest) { super(url, ...rest); note(name.toLowerCase(), 'NEW', String(url)); }
    };
  }
  navigator.serviceWorker?.getRegistrations?.().then((regs) => regs.forEach((r) => note('serviceworker', 'ACTIVE', r.active?.scriptURL || r.scope)));

  console.log(`[capture] installed for ${P.name} — send your prompt now (reload the chat afterwards to also save the stored conversation; window.__captureUrls lists API calls)`);
})();
