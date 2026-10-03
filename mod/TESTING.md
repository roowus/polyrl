# PolyRL mod — in-game test guide

## 1. Start the brain bridge (optional for panel-load, required for demo capture)

```bash
cd ~/projects/polyrl
uv run python -m brain.bridge
```

This serves the WS endpoint the mod talks to (`ws://127.0.0.1:8766`) and a
local static mirror of the mod (`http://127.0.0.1:8767/`). The panel loads fine
without it (status will just show "bridge offline").

## 2. Install the mod in PolyModLoader / PolyLauncher

Mod base URL (the repo is public; PML fetches `manifest.json` →
`0.1.0/version.json` → `0.1.0/main.mod.js` from it). **Use GitHub Pages**, not
raw.githubusercontent.com — raw serves `.js` as `text/plain` and the browser
refuses to `import()` it ("Something went wrong importing this mod!"):

```
https://roowus.github.io/polyrl/mod/
```

Add it via PML's "import mod from URL" and enable it. Refresh / relaunch the
game after enabling.

## 3. What you should see

- A **PolyRL** panel, top-right, draggable. Status pill shows
  `connected` (bridge up) or `bridge offline` / `disconnected`.
- Press **P** to toggle the panel.

## 4. The two things to test

**A. Demo capture (unblocks the climb):** open summer1, click **Record demo
lap**, drive a lap — especially taking the 61–70% climb section *at speed*
(carry ~250 km/h in). When you finish, the mod forwards the recording to the
brain (`fixtures/demo_summer1_*.json`). That fresh line becomes the new reward
path.

**B. TOTW ghost ground-truth (settles the replay desync):** open the Track of
the Week. Tell me and I'll push the WR recording through the bridge to the
mod's ghost slot — the game's own ghost system replays it against real
physics. If the real game finishes it and my headless sim doesn't, that's the
concrete divergence to fix.

## If it doesn't load

The browser console (PolyLauncher devtools) will show the fetch or import
error. Most likely failure: a CDN hiccup on the PolyTypes import (it comes from
cdn.polymodloader.com, not GitHub). Tell me the exact console error.
