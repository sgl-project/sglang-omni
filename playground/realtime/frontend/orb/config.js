// The playground server proxies /v1/realtime on the page's own origin.
window.DEMO_WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/v1/realtime`;
window.DEMO_STATUS_URL = "";
window.DEMO_MODEL_NAME = "MiniCPM-o 4.5";
// No page-level system prompt: presets carry the checkpoint's own duplex prompts, and
// without one the server uses its default. A persona prompt asking for brief answers made
// the model stop answering after its first reply in audio-only calls.
window.DEMO_INSTRUCTIONS = "";
