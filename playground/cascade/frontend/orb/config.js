// The playground server proxies /v1/realtime on the page's own origin.
window.DEMO_WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/v1/realtime`;
window.DEMO_STATUS_URL = "";
window.DEMO_MODEL_NAME = "Streaming Cascade";
