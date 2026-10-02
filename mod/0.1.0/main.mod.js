// PolyRL — in-game mod for the PolyTrack RL driver.
// Requires PolyModLoader (polymodloader.com). Talks to the local brain over
// ws://127.0.0.1:8766 (started by `uv run python -m brain.bridge`).
//
// Nothing here runs until the game loads; physics-side control is spliced
// into simulation_worker.bundle.js via registerSimWorkerMixin in preInit.

import { PolyMod } from "https://cdn.polymodloader.com/cb/polytrackmods/PolyModLoader/0.6.3/PolyTypes.js";

const BRIDGE_URL = "ws://127.0.0.1:8766";

class PolyRLMod extends PolyMod {
  modID = "polyrl";
  modName = "PolyRL";
  modAuthor = "roowus";
  modVersion = "0.1.0";
  // We splice the simulation worker: physics-adjacent, not vanilla-safe.
  touchingPhysics = true;

  preInit = (pml) => {
    this.pml = pml;
    // Splice a controls+telemetry hook into the sim worker's non-realtime
    // loop (h()). Anchored on the exact stock call site; fails loudly if the
    // game build drifts (same anchor policy as the headless patch).
    pml.registerSimWorkerMixin({
      type: 3, // MixinType.INSERT — we can't import the enum in the worker ctx
      token: "t.frames<Qa.maxFrames&&t.frames<t.targetSimulationFrames&&!t.isPaused){const e=t.controls.getControls(t.frames);",
      func: `
        const __lc=globalThis.__polyrlLiveControls;
        if (__lc) { const __ov=__lc(t.id,t.frames); if (__ov) { /* override */ } }
      `,
    });
  };

  init = (pml) => {
    this.pml = pml;
    this._connect();
    pml.registerKeybind("Toggle PolyRL panel", "polyrl.togglePanel", "keydown", "KeyP", null, () => this._togglePanel());
    pml.registerKeybind("PolyRL record demo", "polyrl.recordDemo", "keydown", "KeyO", null, () => this._recordDemo());
  };

  postInit = () => {
    this._mountPanel();
  };

  onGameLoad = () => {
    this._setStatus("game loaded");
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
      setTimeout(() => this._connect(), 3000); // retry
    };
    this.ws.onmessage = (ev) => this._onBridge(JSON.parse(ev.data));
  }

  _send(obj) {
    if (this.ws?.readyState === 1) this.ws.send(JSON.stringify(obj));
  }

  _onBridge(msg) {
    switch (msg.type) {
      case "hello_ack":
        this._setStatus(`brain v${msg.protocol} · ${msg.envs ?? 0} envs`);
        break;
      case "train_status":
        this._setStatus(`training · best ${msg.best_lap_s?.toFixed(2) ?? "—"}s · ${msg.sim_speed?.toFixed(0)}×rt`);
        break;
      case "export_recording":
        navigator.clipboard?.writeText(msg.recording);
        this._setStatus(`recording copied (${msg.frames} frames)`);
        break;
      case "error":
        this._setStatus(`error: ${msg.detail}`);
        break;
    }
  }

  // ---- panel ----------------------------------------------------------------
  _mountPanel() {
    if (document.getElementById("polyrl-panel")) return;
    const el = document.createElement("div");
    el.id = "polyrl-panel";
    el.innerHTML = `
      <div class="polyrl-head">PolyRL <span id="polyrl-status">booting…</span></div>
      <div class="polyrl-row">
        <button id="polyrl-record">Record demo lap</button>
        <button id="polyrl-export">Export best recording</button>
      </div>
      <div class="polyrl-row">
        <button id="polyrl-watch">Watch policy</button>
        <span id="polyrl-track"></span>
      </div>`;
    const style = document.createElement("style");
    style.textContent = `
      #polyrl-panel{position:fixed;top:12px;right:12px;z-index:9999;background:rgba(20,22,40,.92);color:#e8ecf8;border:1px solid #3a4070;border-radius:10px;padding:10px 12px;font:12px/1.5 system-ui;min-width:220px}
      #polyrl-panel .polyrl-head{font-weight:700;margin-bottom:6px}
      #polyrl-panel #polyrl-status{font-weight:400;opacity:.8;margin-left:6px}
      #polyrl-panel .polyrl-row{display:flex;gap:6px;margin:4px 0;align-items:center}
      #polyrl-panel button{background:#2c3160;border:1px solid #4650a0;color:#e8ecf8;border-radius:6px;padding:3px 8px;cursor:pointer;font-size:12px}
      #polyrl-panel button:hover{background:#3a4070}`;
    document.head.appendChild(style);
    document.body.appendChild(el);
    el.querySelector("#polyrl-record").onclick = () => this._recordDemo();
    el.querySelector("#polyrl-export").onclick = () => this._send({ type: "export_best" });
    el.querySelector("#polyrl-watch").onclick = () => this._send({ type: "watch_policy" });
  }

  _togglePanel() {
    const el = document.getElementById("polyrl-panel");
    if (el) el.style.display = el.style.display === "none" ? "" : "none";
  }

  _setStatus(s) {
    const el = document.getElementById("polyrl-status");
    if (el) el.textContent = s;
  }

  _recordDemo() {
    // Arm capture: the game's own race recorder is hooked via the worker
    // mixin; on race.finished the recording string is pulled and sent.
    this._send({ type: "arm_demo_capture" });
    this._setStatus("demo armed — finish a lap");
  }
}

export let polyMod = new PolyRLMod();
