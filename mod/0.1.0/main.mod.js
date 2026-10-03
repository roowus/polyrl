// PolyRL — in-game mod for the PolyTrack RL driver.
// Requires PolyModLoader (polymodloader.com). Talks to the local brain over
// ws://127.0.0.1:8766 (started by `uv run python -m brain.bridge`).
//
// v0.1.0 scope: panel + bridge + demo capture + "ghost this recording" (used
// to test whether the real game finishes a recording the headless sim won't).

// PolyTypes must come from the PML CDN (it's the loader's own API surface).
// The mod itself is served from GitHub Pages so PML can fetch it:
//   https://roowus.github.io/polyrl/mod/0.1.0/main.mod.js
// (raw.githubusercontent.com serves .js as text/plain → import() fails; Pages
// serves application/javascript)
import { PolyMod } from "https://cdn.polymodloader.com/cb/polytrackmods/PolyModLoader/0.6.3/PolyTypes.js";

const BRIDGE_URL = "ws://127.0.0.1:8766";

class PolyRLMod extends PolyMod {
  modID = "polyrl";
  modName = "PolyRL";
  modAuthor = "roowus";
  modVersion = "0.1.0";
  touchingPhysics = false; // v0.1.0: observe + replay only, no physics splice

  preInit = (pml) => {
    this.pml = pml;
  };

  init = (pml) => {
    this.pml = pml;
    this._connect();
    pml.registerKeybind("PolyRL panel", "polyrl.togglePanel", "keydown", "KeyP", null, () =>
      this._togglePanel(),
    );
  };

  postInit = () => {
    this._mountPanel();
  };

  onGameLoad = () => {
    this._setStatus("ready · bridge " + (this.ws?.readyState === 1 ? "connected" : "offline"));
  };

  // ---- bridge ---------------------------------------------------------------
  _connect() {
    try {
      this.ws = new WebSocket(BRIDGE_URL);
    } catch {
      this._setStatus("bridge unreachable");
      return;
    }
    this.ws.onopen = () => {
      this._setStatus("connected");
      this.ws.send(JSON.stringify({ type: "hello", protocol: 1, want: ["watch", "demo"] }));
    };
    this.ws.onclose = () => {
      this._setStatus("disconnected");
      this.ws = null;
      setTimeout(() => this._connect(), 3000);
    };
    this.ws.onmessage = (ev) => this._onBridge(JSON.parse(ev.data));
  }

  _send(obj) {
    if (this.ws?.readyState === 1) this.ws.send(JSON.stringify(obj));
  }

  _onBridge(msg) {
    switch (msg.type) {
      case "hello_ack":
        this._setStatus(`brain connected · ${msg.build}`);
        break;
      case "ghost_recording":
        // brain wants us to ghost a recording on the current track
        this._ghostRecording(msg.recording);
        break;
      case "export_recording":
        navigator.clipboard?.writeText(msg.recording);
        this._setStatus(`recording copied (${msg.frames}f)`);
        break;
      case "demo_saved":
        this._setStatus(`demo saved: ${msg.file}`);
        break;
      case "error":
        this._setStatus(`err: ${msg.detail}`);
        break;
    }
  }

  // ---- ghost replay (the TOTW test) ------------------------------------------
  // Arm a recording to be replayed as a ghost when a race starts. The game's
  // own ghost system re-simulates the recording against its real physics —
  // exactly the ground-truth check for "does the real game finish this lap".
  _ghostRecording(recording) {
    this._armedGhost = recording;
    this._setStatus("ghost armed — start the track to replay");
    // The game reads its ghost from localStorage per track. Writing our
    // recording into the personal-best slot makes the stock ghost system play
    // it. The key format (from the desktop app): polytrack_v5_prod_record_<slot>_<trackId>
    // We use slot 0 and the current track's id.
    try {
      const key = this._recordKey();
      if (!key) {
        this._setStatus("open a track first, then re-arm");
        return;
      }
      localStorage.setItem(
        key,
        JSON.stringify({ frames: 0, recording, nickname: "PolyRL", isSelf: false }),
      );
      this._setStatus("ghost written — restart the track");
    } catch (e) {
      this._setStatus("ghost write failed");
    }
  }

  _recordKey() {
    // discover the current track's record key from existing localStorage keys
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith("polytrack_v5_prod_record_")) return k;
    }
    return null;
  }

  // ---- demo capture ------------------------------------------------------------
  _recordDemo() {
    this._send({ type: "arm_demo_capture" });
    this._setStatus("demo armed — finish a lap");
    // The game's own recorder writes to localStorage on finish. We poll for a
    // fresh record entry and forward it to the brain.
    this._demoWatch = setInterval(() => {
      const key = this._recordKey();
      if (!key) return;
      try {
        const raw = localStorage.getItem(key);
        if (!raw) return;
        const d = JSON.parse(raw);
        if (d?.recording && d.frames > 0 && d.recording !== this._lastDemoSent) {
          this._lastDemoSent = d.recording;
          this._send({ type: "demo_recording", frames: d.frames, recording: d.recording });
          this._setStatus(`demo sent (${d.frames}f)`);
          clearInterval(this._demoWatch);
        }
      } catch {}
    }, 1000);
  }

  // ---- panel -------------------------------------------------------------------
  _mountPanel() {
    if (document.getElementById("polyrl-panel")) return;
    const el = document.createElement("div");
    el.id = "polyrl-panel";
    el.innerHTML = `
      <div class="polyrl-head">PolyRL <span id="polyrl-status">booting…</span></div>
      <div class="polyrl-row"><button id="polyrl-record">Record demo lap</button></div>
      <div class="polyrl-row"><button id="polyrl-export">Export best recording</button></div>`;
    const style = document.createElement("style");
    style.textContent = `
      #polyrl-panel{position:fixed;top:12px;right:12px;z-index:9999;background:rgba(20,22,40,.92);color:#e8ecf8;border:1px solid #3a4070;border-radius:10px;padding:10px 12px;font:12px/1.5 system-ui;min-width:210px}
      #polyrl-panel .polyrl-head{font-weight:700;margin-bottom:6px}
      #polyrl-panel #polyrl-status{font-weight:400;opacity:.8;margin-left:6px}
      #polyrl-panel .polyrl-row{display:flex;gap:6px;margin:4px 0}
      #polyrl-panel button{background:#2c3160;border:1px solid #4650a0;color:#e8ecf8;border-radius:6px;padding:3px 8px;cursor:pointer;font-size:12px;width:100%}
      #polyrl-panel button:hover{background:#3a4070}`;
    document.head.appendChild(style);
    document.body.appendChild(el);
    el.querySelector("#polyrl-record").onclick = () => this._recordDemo();
    el.querySelector("#polyrl-export").onclick = () => this._send({ type: "export_best" });
  }

  _togglePanel() {
    const el = document.getElementById("polyrl-panel");
    if (el) el.style.display = el.style.display === "none" ? "" : "none";
  }

  _setStatus(s) {
    const el = document.getElementById("polyrl-status");
    if (el) el.textContent = s;
  }
}

export let polyMod = new PolyRLMod();
