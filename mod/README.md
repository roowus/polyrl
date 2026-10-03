# polyrl-mod — PolyModLoader mod

The in-game half of PolyRL. Lives in the real game via
[PolyModLoader](https://polymodloader.com) / PolyLauncher.

## What it does

- **Panel UI** (draggable, top-right): connection status, current track,
  training controls (start/pause/stop relayed to the Python brain), live
  best-lap, and Watch mode.
- **Demo capture**: "Record my lap as demo" — hooks the game's own recorder;
  on finish, POSTs the recording string to the local bridge.
- **Watch mode**: plays the current policy's exported recording as an in-game
  ghost, or drives the live car with policy actions streamed from Python.
- **Export**: "Export best recording" → copies the policy's recording string
  (valid per the game's own verifier) to the clipboard. Submission to the
  official leaderboard is the user's call.

## Layout

```
manifest.json          PML global manifest
0.1.0/
  version.json         { targets: ["0.6.3"], main: "main.mod.js" }
  main.mod.js          the mod (plain JS, no build step)
  panel.css            panel styling
```

## Install (PML / PolyLauncher)

The repo is public: **github.com/roowus/polyrl**. Import the mod from the raw
GitHub URL (localhost doesn't work for PML's fetch in most setups):

```
https://raw.githubusercontent.com/roowus/polyrl/main/mod/
```

i.e. the mod base URL is that `mod/` directory; PML reads `manifest.json` then
`0.1.0/main.mod.js` from it. (For local dev, `uv run python -m brain.bridge`
also serves it on `http://127.0.0.1:8767/`.)

## How it talks to the brain

`main.mod.js` opens a WebSocket to `ws://127.0.0.1:8766` (the PolyRL
`bridge.py` server, local to your machine). JSON control frames both ways.

PML specifics used:
- `PolyMod` lifecycle: `preInit` (register mixins), `init` (WS connect),
  `postInit` (panel mount), `onGameLoad`.
- `pml.getFromPolyTrack(path)` / `getFromPolyTrackGlobal(path)` to reach the
  game's car/track objects.
- `pml.registerSimWorkerMixin({ type: INSERT, token, func })` to splice the
  control/telemetry hook into `simulation_worker.bundle.js`'s `h()`/`l()`
  loops.
- `pml.registerKeybind(...)` for panel toggle + record-demo hotkeys.
