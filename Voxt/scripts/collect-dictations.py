#!/usr/bin/env python3
"""Export your own Voxt 2.0 dictation history for the formalized-ASR benchmark.

Run it on the Mac where you use Voxt (Voxt/docs/dictation-collection.md):

    python3 Voxt/scripts/collect-dictations.py

It reads Voxt's history read-only, opens a page in your browser where you
untick anything you do not want to share, and writes one zip to ~/Downloads.
Nothing leaves this Mac; you hand the zip in yourself.

Standard library only, so the Python that comes with macOS (3.9) runs it.
"""

import argparse
import hashlib
import http.server
import json
import plistlib
import re
import sqlite3
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

COLLECTOR_VERSION = "1"
SCHEMA_VERSION = 1

CONSENT = "我同意把勾选的条目（录音、识别原文、润色后文本和应用信息）用于组内 formalized ASR benchmark 研究，数据只在组内私有存放。"

DEFAULT_DATA_DIR = Path.home() / "Library" / "Application Support" / "VoxtNew"
DB_NAME = "voxtnew.db"
APP_PATHS = [Path("/Applications/Voxt.app"), Path.home() / "Applications" / "Voxt.app"]

# note (luojiaxuan): every 2.0.x database has these columns (migrations 001-005 shipped in 2.0.0).
HISTORY_COLUMNS = [
    "id",
    "kind",
    "created_at",
    "status",
    "error",
    "app_bundle",
    "app_name",
    "url_host",
    "rule_id",
    "rule_name",
    "rule_source",
    "raw_text",
    "result",
    "delivery",
    "asr_provider",
    "asr_model",
    "llm_provider",
    "llm_model",
    "audio_ms",
    "sent_ms",
    "asr_ms",
    "llm_ms",
    "total_ms",
    "dsp",
    "gain_db",
    "audio",
    "audio_until",
]
# note (luojiaxuan): the transcribe settings that shape the clean-up; hotkeys and the like stay behind.
TRANSCRIBE_SETTINGS = ("enhance", "cleanup", "structure", "prompt", "llm", "asr")
# note (luojiaxuan): what entries.jsonl carries; error, the inserted app in delivery and the file name stay behind.
EXPORTED_COLUMNS = [
    "id",
    "app_bundle",
    "app_name",
    "url_host",
    "rule_id",
    "rule_name",
    "rule_source",
    "raw_text",
    "result",
    "asr_provider",
    "asr_model",
    "llm_provider",
    "llm_model",
    "audio_ms",
    "sent_ms",
    "asr_ms",
    "llm_ms",
    "total_ms",
    "dsp",
    "gain_db",
]


class CollectError(Exception):
    """A problem the person running the script can fix; printed without a traceback."""


def data_dir():
    """The data folder in use: the default one, or where Settings → Storage moved it (location.json)."""
    location = DEFAULT_DATA_DIR / "location.json"
    if location.exists():
        moved = json.loads(location.read_text()).get("dir", "")
        if moved:
            return Path(moved)
    return DEFAULT_DATA_DIR


def voxt_version():
    """Bundle id and version of the installed Voxt.app, e.g. "dev.hehehai.voxtnew 2.0.3"; "" when none is found."""
    for app in APP_PATHS:
        plist = app / "Contents" / "Info.plist"
        if plist.exists():
            info = plistlib.loads(plist.read_bytes())
            return "{} {}".format(
                info.get("CFBundleIdentifier", ""),
                info.get("CFBundleShortVersionString", ""),
            )
    return ""


def snapshot_db(db_path):
    """Copy the database into memory through SQLite's backup API, never writing to it.

    While Voxt runs, recent commits sit in voxtnew.db-wal, which a read-only connection reads through
    the -shm index. Once Voxt quits, SQLite has folded the WAL into voxtnew.db and removed both files,
    and macOS's SQLite cannot open a WAL database read-only without them; the file alone is then the
    whole database, so it is opened immutable.
    """
    wal = db_path.with_name(db_path.name + "-wal")
    query = "?mode=ro" if wal.exists() else "?mode=ro&immutable=1"
    src = sqlite3.connect(db_path.as_uri() + query, uri=True, timeout=10)
    dst = sqlite3.connect(":memory:")
    src.backup(dst)
    src.close()
    dst.row_factory = sqlite3.Row
    return dst


def document(db, key):
    """A JSON document of Voxt's documents table; {} when it was never written (Voxt then uses its defaults)."""
    row = db.execute("SELECT doc FROM documents WHERE key = ?", [key]).fetchone()
    return json.loads(row["doc"]) if row else {}


def audio_path(name, in_use, default_dir):
    """Where a recording is: in the audio folder in use, else in the default one (Voxt's History.AudioPath)."""
    if (
        in_use / name != default_dir / name
        and not (in_use / name).exists()
        and (default_dir / name).exists()
    ):
        return default_dir / name
    return in_use / name


def statuses(e):
    """(asr_status, llm_status) of a finished transcribe row.

    asr_status: "final" when raw_text is the last recognition pass, "salvaged" when that pass failed
    and the text heard while recording went on instead (Voxt then records the ASR error, "asr ...").
    llm_status: "applied" when the language model returned the result, "failed" when it errored and
    the recognised text (or a mix of cleaned and uncleaned pieces) went out, "off" when none ran,
    "unknown" when the ASR error hides how the language model did.
    """
    error, ran = e["error"], bool(e["llm_model"])
    if error.startswith("asr "):
        return "salvaged", "unknown" if ran else "off"
    if error:
        return "final", "failed"
    return "final", "applied" if ran else "off"


def settings_snapshot(settings, groups, entries):
    """What shaped the language model's output, as configured now; Voxt keeps no copy per entry.

    The transcribe feature's switches and custom prompt, the languages, the dictionary switches,
    the model behind each provider id (rows hold only the id), and the scene prompts of the app
    enhancement groups the entries matched. Endpoints, dictionary terms and notes stay out.
    """
    general = settings.get("general", {})
    transcribe = settings.get("features", {}).get("transcribe", {})
    used = {e["rule_id"] for e in entries if e["rule_id"]}
    providers = settings.get("providers", {})
    return {
        "transcribe": {
            k: transcribe[k] for k in TRANSCRIBE_SETTINGS if k in transcribe
        },
        "general": {
            k: general[k]
            for k in ("language", "mainLanguage", "otherLanguages")
            if k in general
        },
        "dictionary": {
            k: v
            for k, v in settings.get("dictionary", {}).items()
            if isinstance(v, bool)
        },
        "app_enhancement": settings.get("appEnhancement", {}).get("enabled"),
        "providers": {
            kind: [
                {"id": p.get("id"), "model": p.get("model")}
                for p in providers.get(kind, [])
            ]
            for kind in ("llm", "asr")
        },
        "scenes": [
            {"id": g["id"], "name": g.get("name"), "prompt": g.get("prompt")}
            for g in groups.get("groups", [])
            if g["id"] in used
        ],
    }


def load_entries(root):
    """Transcribe sessions that finished with a recording kept and text recognised, oldest first.

    Returns (entries, counts, settings, groups): entries carry the history row plus the resolved
    audio path; counts says how many rows were left out and why, so the summary never hides them;
    settings and groups are Voxt's "settings:app" and "appctx:rules" documents.
    """
    db_path = root / DB_NAME
    if not db_path.exists():
        raise CollectError("没找到 Voxt 2.0 的数据库：{}".format(db_path))
    db = snapshot_db(db_path)
    have = {r["name"] for r in db.execute("PRAGMA table_info(history)")}
    missing = [c for c in HISTORY_COLUMNS if c not in have]
    if missing:
        raise CollectError(
            "Voxt 数据库的 history 表缺少 {}，这个脚本只认 Voxt 2.0.x。".format(
                ", ".join(missing)
            )
        )
    settings = document(db, "settings:app")
    groups = document(db, "appctx:rules")
    default_audio = root / "audio"
    chosen_dir = Path(settings.get("history", {}).get("audioDir", "") or default_audio)
    in_use = chosen_dir if chosen_dir.is_dir() else default_audio
    rows = db.execute(
        "SELECT {} FROM history ORDER BY created_at, id".format(
            ", ".join(HISTORY_COLUMNS)
        )
    ).fetchall()
    db.close()
    counts = {key: 0 for key, _ in SKIP_REASONS}
    counts["history"] = len(rows)
    entries = []
    for row in rows:
        e = dict(row)
        reason = skip_reason(e)
        if reason is None:
            e["audio_path"] = audio_path(e["audio"], in_use, default_audio)
            if not e["audio_path"].is_file():
                reason = "audio_missing"
        if reason:
            counts[reason] += 1
            continue
        e["asr_status"], e["llm_status"] = statuses(e)
        e["delivery"] = e["delivery"].split("|")[0]
        entries.append(e)
    counts["exported"] = len(entries)
    return entries, counts, settings, groups


# note (luojiaxuan): why a history row is not exported, in the order they are tested, with the line the summary prints.
SKIP_REASONS = [
    ("not_transcribe", "翻译或改写"),
    ("legacy", "从旧版 Voxt 导入的记录（只有最终文本，没有录音）"),
    ("not_done", "失败或取消"),
    ("no_text", "没有识别出文字"),
    ("no_audio", "没有保存录音"),
    ("audio_temporary", "只临时保留 7 天的录音（识别失败或取消）"),
    ("audio_missing", "录音文件找不到"),
]


def skip_reason(e):
    if e["kind"] != "transcribe":
        return "not_transcribe"
    if e["id"].startswith("legacy-"):
        return "legacy"
    if e["status"] != "done":
        return "not_done"
    if not e["raw_text"].strip() or not e["result"].strip():
        return "no_text"
    if not e["audio"] or "/" in e["audio"]:
        return "no_audio"
    if e["audio_until"]:
        return "audio_temporary"
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iso_utc(ms):
    return (
        datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def record(e, audio_name, audio_sha):
    """One line of entries.jsonl: the history row as the benchmark sees it."""
    out = {k: e[k] for k in EXPORTED_COLUMNS}
    out["created_at"] = iso_utc(e["created_at"])
    out["delivery"] = e["delivery"]
    out["asr_status"] = e["asr_status"]
    out["llm_status"] = e["llm_status"]
    out["audio"] = audio_name
    out["audio_sha256"] = audio_sha
    return out


def write_bundle(entries, ids, alias, consent, out_dir, voxt, counts, settings, groups):
    """Write the zip of the chosen entries and return its path."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", alias):
        raise CollectError("代号只能用字母、数字、- 和 _，最长 32 个字符。")
    chosen = [e for e in entries if e["id"] in set(ids)]
    if not chosen:
        raise CollectError("没有勾选任何条目。")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = "voxt-history-{}-{}".format(alias, stamp)
    path = Path(out_dir) / (name + ".zip")
    lines, files = [], {}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for e in chosen:
            audio_name = "audio/{}{}".format(e["id"], e["audio_path"].suffix)
            digest = sha256(e["audio_path"])
            z.write(
                e["audio_path"],
                "{}/{}".format(name, audio_name),
                compress_type=zipfile.ZIP_STORED,
            )
            files[audio_name] = digest
            lines.append(json.dumps(record(e, audio_name, digest), ensure_ascii=False))
        body = ("\n".join(lines) + "\n").encode("utf-8")
        files["entries.jsonl"] = hashlib.sha256(body).hexdigest()
        z.writestr("{}/entries.jsonl".format(name), body)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "collector_version": COLLECTOR_VERSION,
            "voxt_version": voxt,
            "contributor": alias,
            "exported_at": iso_utc(time.time() * 1000),
            "consent": consent,
            "counts": dict(
                counts, chosen=len(chosen), unticked=len(entries) - len(chosen)
            ),
            "voxt_settings": settings_snapshot(settings, groups, chosen),
            "files": files,
        }
        z.writestr(
            "{}/manifest.json".format(name),
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
    return path


class Review:
    """The local review page: serves the entries and their audio on 127.0.0.1 and packs the ticked ones."""

    def __init__(self, entries, counts, settings, groups, voxt, out_dir, page):
        self.entries = entries
        self.settings = settings
        self.groups = groups
        self.by_id = {e["id"]: e for e in entries}
        self.counts = counts
        self.voxt = voxt
        self.out_dir = out_dir
        self.page = page
        self.done = threading.Event()
        self.result = None
        self.host = ""

    def listing(self):
        keys = (
            "id",
            "app_bundle",
            "app_name",
            "url_host",
            "rule_name",
            "raw_text",
            "result",
            "audio_ms",
            "asr_status",
            "llm_status",
        )
        return {
            "consent": CONSENT,
            "settings": settings_snapshot(self.settings, self.groups, self.entries),
            "entries": [
                dict({k: e[k] for k in keys}, created_at_ms=e["created_at"])
                for e in self.entries
            ],
        }

    def handler(self):
        review = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, code, body, kind):
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def send_json(self, code, obj):
                self.send(
                    code,
                    json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )

            def from_page(self):
                """Only the page this script opened may ask: a DNS-rebound name would carry another Host."""
                if self.headers["Host"] == review.host:
                    return True
                self.send_json(403, {"error": "forbidden"})
                return False

            def do_GET(self):
                if not self.from_page():
                    return
                if self.path == "/":
                    self.send(200, review.page, "text/html; charset=utf-8")
                elif self.path == "/entries.json":
                    self.send_json(200, review.listing())
                elif self.path.startswith("/audio/"):
                    e = review.by_id.get(self.path[len("/audio/") :])
                    if e is None:
                        self.send_json(404, {"error": "not found"})
                        return
                    self.send(200, e["audio_path"].read_bytes(), "audio/wav")
                else:
                    self.send_json(404, {"error": "not found"})

            def do_POST(self):
                if not self.from_page():
                    return
                if self.path != "/pack":
                    self.send_json(404, {"error": "not found"})
                    return
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if req.get("consent") != CONSENT:
                    self.send_json(400, {"error": "需要先勾选同意。"})
                    return
                try:
                    path = write_bundle(
                        review.entries,
                        req["ids"],
                        req["alias"],
                        CONSENT,
                        review.out_dir,
                        review.voxt,
                        review.counts,
                        review.settings,
                        review.groups,
                    )
                except CollectError as err:
                    self.send_json(400, {"error": str(err)})
                    return
                review.result = path
                self.send_json(
                    200,
                    {
                        "path": str(path),
                        "count": len(set(req["ids"]) & set(review.by_id)),
                    },
                )
                review.done.set()

        return Handler


def print_counts(counts, keep_audio):
    print("Voxt 历史共 {history} 条，其中可导出的听写 {exported} 条。".format(**counts))
    for key, why in SKIP_REASONS:
        if counts[key]:
            print("  未导出 {} 条：{}".format(counts[key], why))
    if not keep_audio:
        print(
            "注意：Voxt 现在没有打开“保存录音”，之后的听写不会留下录音。"
            "打开方法：Voxt → 历史 → 历史设置 → 保存录音。"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        default=str(Path.home() / "Downloads"),
        help="where the zip goes (default ~/Downloads)",
    )
    args = parser.parse_args()

    if not DEFAULT_DATA_DIR.exists():
        raise CollectError(
            "这台 Mac 上没有 Voxt 2.0 的数据（{}）。先装好 Voxt 2.0 并打开“保存录音”，用一段时间再来。".format(
                DEFAULT_DATA_DIR
            )
        )
    root = data_dir()
    entries, counts, settings, groups = load_entries(root)
    print_counts(counts, settings.get("privacy", {}).get("keepAudio", False))
    if not entries:
        raise CollectError(
            "没有可导出的条目。先在 Voxt → 历史 → 历史设置里打开“保存录音”，用一段时间再来。"
        )

    review = Review(
        entries,
        counts,
        settings,
        groups,
        voxt_version(),
        args.out,
        REVIEW_PAGE.encode("utf-8"),
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), review.handler())
    review.host = "127.0.0.1:{}".format(server.server_address[1])
    url = "http://{}/".format(review.host)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("在浏览器里打开了挑选页面：{}".format(url))
    print("挑完点“打包”。想中途放弃就按 Ctrl-C。")
    subprocess.run(["/usr/bin/open", url], check=True)
    review.done.wait()
    server.shutdown()
    print("打包好了：{}".format(review.result))
    subprocess.run(["/usr/bin/open", "-R", str(review.result)], check=True)


REVIEW_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voxt 听写数据挑选</title>
<style>
:root {
  --bg: #fafaf9; --panel: #ffffff; --text: #1c1917; --muted: #78716c;
  --line: #e7e5e4; --accent: #2563eb; --off: #a8a29e; --warn: #b45309;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #1c1917; --panel: #292524; --text: #f5f5f4; --muted: #a8a29e;
    --line: #44403c; --accent: #60a5fa; --off: #78716c; --warn: #f59e0b;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif; }
header { background: var(--panel); border-bottom: 1px solid var(--line); padding: 12px 16px; }
header h1 { font-size: 17px; margin: 0 0 4px; }
header p { margin: 0; color: var(--muted); font-size: 13px; }
main { max-width: 920px; margin: 0 auto; padding: 16px; }
.apps { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 0; }
.apps label { border: 1px solid var(--line); border-radius: 999px; padding: 2px 10px;
  font-size: 13px; cursor: pointer; user-select: none; }
.apps label.off { color: var(--off); text-decoration: line-through; }
.apps input { display: none; }
.tools { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
button { font: inherit; border: 1px solid var(--line); background: var(--panel); color: var(--text);
  border-radius: 6px; padding: 4px 12px; cursor: pointer; }
button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
button:disabled { opacity: .5; cursor: default; }
.entry { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 10px 12px; margin: 10px 0; }
.entry.off { opacity: .45; }
.entry .meta { display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
  color: var(--muted); font-size: 13px; }
.entry .meta input { width: 18px; height: 18px; }
.entry audio { width: 100%; height: 32px; margin: 6px 0 2px; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
@media (max-width: 640px) { .pair { grid-template-columns: 1fr; } }
.pair div { white-space: pre-wrap; word-break: break-word; }
.pair b { display: block; font-size: 12px; color: var(--muted); font-weight: 500; }
.nollm { color: var(--warn); }
footer { position: sticky; bottom: 0; z-index: 2; background: var(--panel); border-top: 1px solid var(--line);
  padding: 10px 16px; }
footer .row { max-width: 920px; margin: 0 auto; display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
footer input[type=text] { font: inherit; padding: 4px 8px; border: 1px solid var(--line);
  border-radius: 6px; background: var(--bg); color: var(--text); }
#status { color: var(--muted); }
details { margin-top: 8px; font-size: 13px; color: var(--muted); }
details pre { max-height: 40vh; overflow: auto; white-space: pre-wrap; word-break: break-word;
  background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 8px; color: var(--text); }
</style>
</head>
<body>
<header>
  <h1>Voxt 听写数据挑选</h1>
  <p id="summary"></p>
  <p>请逐条看一遍：取消勾选不想分享的条目，或者点下面的应用名整组去掉。只有勾选的条目会被打包，打包前什么都不会离开这台电脑。</p>
  <div class="apps" id="apps"></div>
  <div class="tools">
    <button id="all">全选</button><button id="none">全不选</button>
  </div>
  <details>
    <summary>还会附上现在的润色设置（Voxt 不按条目保存提示词，只能附上当前的）</summary>
    <pre id="settings"></pre>
  </details>
</header>
<main id="list"></main>
<footer>
  <div class="row">
    <label>你的代号 <input type="text" id="alias" placeholder="例如 ljx" maxlength="32"></label>
    <label><input type="checkbox" id="consent"> <span id="consentText"></span></label>
    <button class="primary" id="pack" disabled>打包</button>
    <span id="status"></span>
  </div>
</footer>
<script>
const state = { entries: [], selected: new Set(), appOff: new Set() };
const $ = id => document.getElementById(id);

function fmtTime(ms) {
  const d = new Date(ms);
  const p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function appKey(e) { return e.app_name || e.app_bundle || '未知应用'; }

function render() {
  const list = $('list');
  list.textContent = '';
  for (const e of state.entries) {
    const on = state.selected.has(e.id);
    const card = document.createElement('section');
    card.className = 'entry' + (on ? '' : ' off');
    const meta = document.createElement('div');
    meta.className = 'meta';
    const box = document.createElement('input');
    box.type = 'checkbox';
    box.checked = on;
    box.onchange = () => { box.checked ? state.selected.add(e.id) : state.selected.delete(e.id); refresh(); };
    meta.append(box);
    const info = document.createElement('span');
    info.textContent = [fmtTime(e.created_at_ms), appKey(e) + (e.url_host ? ` · ${e.url_host}` : ''),
      e.rule_name ? `场景：${e.rule_name}` : '', `${(e.audio_ms / 1000).toFixed(1)} 秒`].filter(Boolean).join('  ·  ');
    meta.append(info);
    const warning = e.asr_status === 'salvaged' ? '最后一遍识别失败，识别原文是录音时听到的实时文本'
      : { failed: '润色失败，交付的是识别原文', off: '这条没有经过语言模型润色' }[e.llm_status];
    if (warning) {
      const w = document.createElement('span');
      w.className = 'nollm';
      w.textContent = warning;
      meta.append(w);
    }
    card.append(meta);
    const audio = document.createElement('audio');
    audio.controls = true;
    audio.preload = 'none';
    audio.src = `/audio/${encodeURIComponent(e.id)}`;
    card.append(audio);
    const pair = document.createElement('div');
    pair.className = 'pair';
    for (const [label, text] of [['识别原文', e.raw_text], ['润色后', e.result]]) {
      const div = document.createElement('div');
      const b = document.createElement('b');
      b.textContent = label;
      div.append(b, document.createTextNode(text));
      pair.append(div);
    }
    card.append(pair);
    list.append(card);
  }
}

function renderApps() {
  const counts = new Map();
  for (const e of state.entries) counts.set(appKey(e), (counts.get(appKey(e)) || 0) + 1);
  const apps = $('apps');
  apps.textContent = '';
  for (const [name, n] of [...counts].sort((a, b) => b[1] - a[1])) {
    const label = document.createElement('label');
    const off = state.appOff.has(name);
    label.className = off ? 'off' : '';
    label.textContent = `${name} ${n}`;
    label.onclick = () => {
      const turnOff = !state.appOff.has(name);
      turnOff ? state.appOff.add(name) : state.appOff.delete(name);
      for (const e of state.entries) if (appKey(e) === name) turnOff ? state.selected.delete(e.id) : state.selected.add(e.id);
      refresh();
    };
    apps.append(label);
  }
}

function refresh() {
  renderApps();
  render();
  $('summary').textContent = `共 ${state.entries.length} 条带录音的听写，已勾选 ${state.selected.size} 条。`;
  $('pack').textContent = `打包 ${state.selected.size} 条`;
  $('pack').disabled = !($('consent').checked && $('alias').value.trim() && state.selected.size);
}

$('all').onclick = () => { state.appOff.clear(); state.entries.forEach(e => state.selected.add(e.id)); refresh(); };
$('none').onclick = () => { state.selected.clear(); refresh(); };
$('consent').onchange = refresh;
$('alias').oninput = refresh;

$('pack').onclick = async () => {
  $('pack').disabled = true;
  $('status').textContent = '正在打包…';
  const res = await fetch('/pack', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ alias: $('alias').value.trim(), consent: $('consentText').textContent,
      ids: state.entries.filter(e => state.selected.has(e.id)).map(e => e.id) }),
  });
  const body = await res.json();
  if (!res.ok) { $('status').textContent = `打包失败：${body.error}`; $('pack').disabled = false; return; }
  document.body.innerHTML = '';
  const done = document.createElement('main');
  const h = document.createElement('h1');
  h.textContent = `打包好了：${body.count} 条`;
  const p = document.createElement('p');
  p.textContent = `文件在 ${body.path}，Finder 里已经选中它。按 playbook 把它交上去就行，这个页面可以关掉了。`;
  done.append(h, p);
  document.body.append(done);
};

(async () => {
  const data = await (await fetch('/entries.json')).json();
  state.entries = data.entries;
  state.entries.forEach(e => state.selected.add(e.id));
  $('consentText').textContent = data.consent;
  $('settings').textContent = JSON.stringify(data.settings, null, 2);
  refresh();
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    try:
        main()
    except CollectError as err:
        print(err, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n已取消，什么都没有打包。", file=sys.stderr)
        sys.exit(130)
