// Step animations for the Voxt tutorials: <div class="vx-anim" data-scene="NAME" data-lang="zh|en"></div>.
// Scenes are redrawn from Voxt's own UI strings; masked fields never hold a real value.
(function () {
  "use strict";
  var base = (document.currentScript && document.currentScript.src || "").replace(/js\/voxt_anim\.js.*$/, "");
  var ICON = base + "image/voxt-icon.png";
  var LANG = "zh";
  function L(zh, en) { return LANG === "en" ? en : zh; }

  var P = {
    home: '<path d="M3 10.5 12 3l9 7.5V21h-6v-6H9v6H3z"/>',
    history: '<path d="M3 12a9 9 0 1 0 2.6-6.4L3 8"/><path d="M3 3v5h5M12 7v5l3 2"/>',
    mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
    globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
    pen: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M14 6l4 4"/>',
    notes: '<rect x="5" y="4" width="14" height="16" rx="3"/><path d="M9 9h6M9 13h4"/>',
    plus: '<circle cx="12" cy="12" r="9"/><path d="M12 8v8M8 12h8"/>',
    wave: '<path d="M4 10v4M8 7v10M12 4v16M16 7v10M20 10v4"/>',
    cloud: '<path d="M7 18h10a4 4 0 0 0 .5-8A6 6 0 0 0 6 9a4.5 4.5 0 0 0 1 9z"/>',
    search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/>',
    chev: '<path d="M7 10l5 5 5-5"/>',
    checkc: '<circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-6"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.6v.4"/>',
    dl: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
    copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h8"/>',
    x: '<path d="M6 6l12 12M18 6 6 18"/>',
    db: '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    warn: '<path d="M12 3 2 20h20z"/><path d="M12 10v4M12 17v.4"/>',
    appA: '<path d="M12 4 5 20M12 4l7 16M8 14h8"/>',
    arrow: '<path d="M4 12h16M14 6l6 6-6 6"/>'
  };
  function ic(n, extra) { return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"' + (extra || "") + ">" + P[n] + "</svg>"; }
  function mask(w, p) { return '<span class="vx-mask" data-l="' + L("已打码", "hidden") + '" style="--mw:' + w + (p ? ";--p:0" : "") + '"' + (p ? ' data-p="' + p + '"' : "") + "></span>"; }
  function rOpt() { return L("右 ⌥", "Right ⌥"); }

  // ---------- shared chunks ----------
  function sidebar(active, note) {
    var items = [["home", "首页", "Home", "home"], ["history", "历史", "History", "history"], 0, ["transcribe", "转录", "Transcribe", "mic"], ["translate", "翻译", "Translate", "globe"], ["rewrite", "改写", "Rewrite", "pen"], ["notes", "笔记", "Notes", "notes"], ["ae", "应用增强", "App enhancement", "plus"], 1, ["asr", "语音识别模型", "Speech models", "wave"], ["llm", "语言模型", "Language models", "cloud"]];
    var h = '<div class="vx-side"><div class="lt"><i style="background:#ff5f57"></i><i style="background:#febc2e"></i><i style="background:#28c840"></i></div>' +
      '<div class="vx-search">' + ic("search", ' style="width:13px;height:13px"') + "<span>" + L("搜索", "Search") + "</span>⌘K</div>";
    items.forEach(function (it) {
      if (it === 0) { h += '<div class="vx-grp">' + L("功能", "Features") + "</div>"; return; }
      if (it === 1) { h += '<div class="vx-grp">' + L("资源", "Resources") + "</div>"; return; }
      h += '<div class="vx-nav' + (it[0] === active ? " on" : "") + '" data-nav="' + it[0] + '">' + ic(it[3]) + L(it[1], it[2]) + "</div>";
    });
    if (note) h += '<div class="vx-sidecard" id="note"><b>' + ic("warn", ' style="width:13px;height:13px;color:#c98a12"') + L("没有语言模型", "No language model") + "</b>" +
      L("翻译和改写需要它。没有它时，转录会直接输入识别出的原文。", "Translate and Rewrite need one. Without it, Transcribe types the text as recognised.") + "<br><a>" + L("打开语言模型页", "Open Language models") + "</a></div>";
    return h + "</div>";
  }
  function steps() { return LANG === "en" ? ["Welcome", "Permissions", "Model", "Transcribe", "Translate", "Rewrite", "More"] : ["欢迎", "权限", "模型", "转录", "翻译", "改写", "更多"]; }
  function crumb() {
    var h = '<div class="vx-crumb">';
    steps().forEach(function (s, i) { h += (i ? '<span class="gt">›</span>' : "") + '<b data-st="' + i + '">' + ic("checkc", ' class="ck"') + s + "</b>"; });
    return h + '<span class="x">×</span><span class="bar"></span></div>';
  }
  function crumbAt(box, cur) {
    box.querySelectorAll(".vx-crumb b").forEach(function (b) {
      var i = +b.dataset.st;
      b.classList.toggle("on", i === cur);
      b.classList.toggle("done", i < cur && i > 1);
    });
    var bar = box.querySelector(".vx-crumb .bar");
    if (bar) bar.style.setProperty("--p", (cur + 1) / 7);
  }
  function llmCard(name, remote, id) {
    return '<div class="vx-card" ' + (id ? 'id="' + id + '" ' : "") + 'style="padding:9px 12px;transition:border-color .15s,box-shadow .15s"><div style="display:flex;align-items:center;gap:6px"><span style="width:6px;height:6px;border-radius:50%;background:#c7c7cc" class="dot"></span><b style="flex:1">' + name + '</b><span class="vx-tag">' + (remote ? L("远程", "Remote") : L("本机", "Local")) + '</span></div><div style="color:#8e8e93;font-size:12px;margin-top:3px" class="m">' + L("未选择模型", "Model not chosen") + "</div></div>";
  }
  function openaiDialog(filled, left) {
    var lab = function (t) { return '<div style="margin:10px 0 5px;font-weight:500;display:flex;gap:5px;align-items:center">' + t + ic("info", ' style="width:13px;height:13px;color:#a1a1a6"') + "</div>"; };
    return '<div class="vx-dlg" id="dlg" style="left:' + (left || 20) + 'px;top:16px;width:520px;height:478px">' +
      '<div style="padding:16px 20px 0;display:flex"><div style="flex:1"><div style="font-size:17px;font-weight:700">OpenAI</div><div class="vx-sub" style="margin:0">' + L("远程语言模型 · OpenAI 兼容 API", "Remote language model · OpenAI-compatible API") + '</div></div><span style="color:#6e6e73;font-size:20px;line-height:1">×</span></div>' +
      '<div style="padding:10px 20px 0">' +
      '<div style="display:flex;justify-content:space-between;color:#6e6e73;font-size:12px"><span>' + L("识别出的文字和选中的文字会发送给 OpenAI。", "Recognised and selected text is sent to OpenAI.") + '</span><span style="color:#1f7a63;font-weight:600">' + L("获取密钥", "Get a key") + "</span></div>" +
      lab("API Key") + '<div class="vx-field" id="f-key">' + mask("92%", filled ? 0 : "key") + "</div>" +
      lab(L("地址", "Address")) + '<div class="vx-field" id="f-addr">' + mask("92%", filled ? 0 : "addr") + "</div>" +
      '<div style="margin:10px 0 5px;font-weight:500">' + L("模型", "Model") + "</div>" +
      '<div class="vx-sel" id="f-model" style="width:300px"><span id="model-text">' + (filled ? L("其他模型…", "Other model…") : mask("130px")) + "</span>" + ic("chev") + "</div>" +
      '<div id="f-idwrap" style="overflow:hidden;height:' + (filled ? 44 : 0) + 'px"><div style="height:8px"></div><div class="vx-field" id="f-id">' + mask("92%", filled ? 0 : "id") + "</div></div>" +
      '<div style="margin:12px 0 4px;color:#8e8e93;font-weight:600;font-size:12px">' + L("生成设置", "Generation settings") + "</div>" +
      '<div style="font-weight:500;margin-bottom:5px">' + L("思考", "Thinking") + '</div><div class="vx-sel" style="width:220px"><span>' + L("默认", "Default") + "</span>" + ic("chev") + "</div>" +
      "</div>" +
      '<div class="vx-menu" id="menu" style="left:20px;top:268px;width:300px"><div>' + mask("120px") + "</div><div>" + mask("150px") + "</div><div>" + mask("100px") + '</div><div id="opt-other">' + L("其他模型…", "Other model…") + "</div></div>" +
      '<div style="position:absolute;left:0;right:0;bottom:0;height:58px;border-top:1px solid #ececef;background:#fff;display:flex;align-items:center;gap:10px;padding:0 20px">' +
      '<span class="vx-btn" id="b-test">' + L("测试连接", "Test connection") + '</span><span class="vx-hide" id="testing" style="display:flex;align-items:center;gap:6px;color:#8e8e93;font-size:12px;position:absolute;left:' + L(122, 152) + 'px"><span class="vx-spin" id="spin"></span>' + L("正在测试…", "Testing…") + "</span>" +
      '<span class="vx-ok vx-hide" id="test-ok" style="position:absolute;left:' + L(122, 152) + 'px">' + ic("checkc") + L("连接正常 · 2 毫秒", "Connection OK · 2 ms") + "</span>" +
      '<span style="flex:1"></span><span class="vx-btn txt">' + L("取消", "Cancel") + '</span><span class="vx-btn pri" id="b-save">' + L("保存", "Save") + "</span></div></div>";
  }
  function editor(o) {
    return '<div class="vx-win" style="left:60px;top:18px;width:520px;height:220px"><div class="vx-tb"><i></i><i></i><i></i><span class="t">' + o.title + "</span></div>" +
      '<div class="vx-ed">' + o.body + "</div></div>" +
      '<div class="vx-pill" id="pill" style="bottom:8px"><div class="vx-pill-row">' + ic("mic") + '<span class="lbl" id="plbl"></span><div class="vx-wave" id="wave">' + new Array(17).join("<i></i>") + '</div><div class="vx-dots">' + new Array(6).join("<i></i>") + '</div></div><div class="vx-live"><span id="live"></span></div></div>';
  }
  function tickWave(box, t, on) {
    var bars = box.querySelectorAll("#wave i");
    for (var i = 0; i < bars.length; i++) {
      var v = on ? 4 + 16 * Math.abs(Math.sin(t * 7.3 + i * 0.9) * Math.sin(t * 2.1 + i * 0.37)) : 4;
      bars[i].style.height = v.toFixed(1) + "px";
    }
    var dots = box.querySelectorAll(".vx-dots i");
    for (var j = 0; j < dots.length; j++) dots[j].style.opacity = 0.3 + 0.7 * Math.max(0, Math.sin(t * 6 - j * 0.8));
  }

  // ---------- scenes: each is a function of the language ----------
  var SC = {};

  SC["install-drag"] = function () {
    return {
      w: 640, h: 330, dur: 6.2, label: L("把 Voxt 拖到 Applications", "Drag Voxt to Applications"),
      html: '<div class="vx-win" style="left:70px;top:26px;width:500px;height:270px"><div class="vx-tb"><i></i><i></i><i></i><span class="t">Voxt 2.0.3</span></div>' +
        '<div class="vx-dmg"><div class="vx-ico" id="app"><img alt="" src="' + ICON + '"><span>Voxt</span></div>' +
        '<div style="color:#9aa3b2;width:70px">' + ic("arrow", ' style="width:60px;height:40px"') + "</div>" +
        '<div class="vx-ico" id="apps"><div class="vx-folder">' + ic("appA") + "</div><span>Applications</span></div></div>" +
        '<div class="vx-hide" id="copy" style="position:absolute;left:150px;right:150px;bottom:18px;text-align:center;font-size:12px;color:#6e6e73">' + L("正在拷贝“Voxt”…", "Copying “Voxt”…") + '<div class="vx-prog" style="margin-top:6px"><i id="copyp"></i></div></div>' +
        '<div class="vx-hide" id="copied" style="position:absolute;left:0;right:0;bottom:24px;text-align:center;font-size:12.5px;color:#1f8a5b;font-weight:600">' + L("已拷贝到“应用程序”", "Copied to Applications") + "</div></div>" +
        '<div class="vx-ghost" id="ghost"><img alt="" src="' + ICON + '"></div>',
      cur: [[0, [430, 300]], [0.4, [430, 300]], [1.3, "#app img"], [1.7, "#app img"], [3.1, "#apps .vx-folder"], [3.6, "#apps .vx-folder"], [4.4, [520, 300]]],
      clk: [[1.5, "#app img"], [3.25, "#apps .vx-folder"]],
      on: [[1.5, 3.3, "#app", "on"], [1.6, 3.3, "#ghost"], [2.7, 3.4, "#apps", "on"], [3.4, 4.9, "#copy"], [4.9, 99, "#copied"]],
      p: [[3.5, 4.8, "#copyp"]],
      tick: function (t, c) { var g = c.q("#ghost"); g.style.left = (c.cur[0] - 38) + "px"; g.style.top = (c.cur[1] - 38) + "px"; }
    };
  };

  function ghRun(n, title, sha, ago, id) {
    return '<div class="vx-ghrow"' + (id ? ' id="' + id + '"' : "") + ">" + ic("checkc", ' style="width:16px;height:16px;color:#1a7f37;flex:none"') +
      '<div style="flex:1;min-width:0"><div class="tt">' + title + '</div><div class="mt">Voxt Beta #' + n + ": Commit " + sha + "</div></div>" +
      '<span class="br">main</span><span class="mt" style="width:80px;text-align:right">' + ago + "</span></div>";
  }
  SC["beta-artifact"] = function () {
    return {
      w: 640, h: 400, dur: 9.4, label: L("在 Voxt Beta 构建列表里下载最新的 Beta", "Download the newest Beta from the Voxt Beta runs"),
      html: '<div class="vx-win" style="left:14px;top:12px;width:612px;height:376px"><div class="vx-tb"><i></i><i></i><i></i><span class="t" style="font-weight:500">github.com/sgl-project/sglang-omni/actions</span></div>' +
        '<div class="vx-page" id="g-list" style="inset:30px 0 0 0;padding:16px 20px"><div class="vx-h1" style="font-size:18px">Voxt Beta</div><div class="vx-sub" style="color:#0969da">voxt-beta.yaml</div>' +
        '<div class="vx-ghbox"><div class="vx-ghhd">30 workflow runs</div>' +
        ghRun(40, "[Voxt] Add a script that exports dictations for an ASR benchmark", "5f03f7e", L("刚刚", "just now"), "run40") +
        ghRun(39, "[Voxt] The answer card shows a fast streamed answer at a steady pace", "cfa5d78", L("3 小时前", "3 hours ago")) +
        ghRun(38, "[Voxt] Merge 24 open Voxt fixes and features into one PR", "85be291", L("21 小时前", "21 hours ago")) + "</div></div>" +
        '<div class="vx-page" id="g-run" style="inset:30px 0 0 0;padding:16px 20px"><div style="color:#57606a;font-size:12px">Voxt Beta · #40</div><div class="vx-h1" style="font-size:16px;margin:2px 0 10px">[Voxt] Add a script that exports dictations for an ASR benchmark</div>' +
        '<div class="vx-ghbox" style="display:flex;gap:28px;padding:12px 14px;font-size:12px;color:#57606a"><div>Status<br><b style="color:#1a7f37">Success</b></div><div>Branch<br><b style="color:#0969da">main</b></div><div>Artifacts<br><b style="color:#1f2328">1</b></div></div>' +
        '<div class="vx-ghbox" style="margin-top:12px"><div class="vx-ghhd" style="font-weight:600;color:#1f2328">Artifacts<span style="font-weight:400;color:#57606a;margin-left:8px">Produced during runtime</span></div>' +
        '<div class="vx-ghrow" style="font-size:12px;color:#57606a"><span style="flex:1">Name</span><span style="width:80px">Size</span></div>' +
        '<div class="vx-ghrow"><span style="flex:1"><a id="art" style="color:#0969da;font-weight:600">Voxt-2.0.3-beta.20261112.40-arm64</a></span><span style="width:80px;color:#57606a">33.4 MB</span></div></div></div>' +
        '<div class="vx-dlbubble" id="dlb"><div style="font-weight:600;font-size:12px">' + ic("dl", ' style="width:14px;height:14px;display:inline-block;vertical-align:-2px;margin-right:4px"') + 'Voxt-2.0.3-beta.20261112.40-arm64.zip</div><div class="vx-prog" style="margin-top:6px"><i id="dlbp"></i></div></div>' +
        '<div class="vx-page" id="g-unzip" style="inset:30px 0 0 0;background:#fff"><div class="vx-dmg" style="top:0"><div class="vx-ico"><div class="vx-zip">ZIP</div><span style="font-size:11.5px">Voxt-…-beta.…-arm64.zip</span></div>' +
        '<div style="color:#9aa3b2;width:70px">' + ic("arrow", ' style="width:60px;height:40px"') + '</div><div class="vx-ico" id="dmg"><div class="vx-dmgico"><img alt="" src="' + ICON + '"></div><span style="font-size:11.5px">Voxt-2.0.3-beta.20261112.40-arm64.dmg</span></div></div>' +
        '<div style="position:absolute;left:0;right:0;bottom:26px;text-align:center;font-size:12.5px;color:#6e6e73">' + L("双击 ZIP 解压，再打开里面的 .dmg", "Double-click the ZIP to unzip it, then open the .dmg inside") + "</div></div></div>",
      cur: [[0, [500, 330]], [0.4, [500, 330]], [1.4, "#run40 .tt"], [2.0, "#run40 .tt"], [3.4, "#art"], [4.1, "#art"], [5.0, [520, 320]], [6.4, [520, 320]], [7.2, "#dmg .vx-dmgico"], [9.4, "#dmg .vx-dmgico"]],
      clk: [[1.6, "#run40 .tt"], [3.6, "#art"], [7.45, "#dmg .vx-dmgico"]],
      on: [[0, 1.75, "#g-list"], [1.75, 6.2, "#g-run"], [1.3, 1.75, "#run40", "hot"], [3.75, 6.2, "#dlb"], [6.2, 99, "#g-unzip"], [7.4, 99, "#dmg", "on"]],
      p: [[3.8, 5.6, "#dlbp", 1]]
    };
  };

  SC["guide-start"] = function () {
    var notAllowed = L("未允许", "Not allowed"), allowed = L("已允许", "Allowed");
    return {
      w: 640, h: 440, dur: 10.6, label: L("设置引导：开始设置、允许麦克风和辅助功能", "Setup guide: get started and allow the microphone and Accessibility"),
      html: '<div class="vx-dlg" style="left:30px;top:16px;width:580px;height:408px">' + crumb() +
        '<div class="vx-gbody">' +
        '<div class="vx-gpane" id="pw"><div class="vx-h2" style="font-size:22px">' + L("欢迎使用 Voxt", "Welcome to Voxt") + '</div><div class="vx-p" style="font-size:13px">' + L("在任何 App 里说话，文字就出现在光标处，可以转录、翻译、改写。接下来几步都很短，每一步都可以跳过。", "Speak in any app and get text where your cursor is: transcribe, translate and rewrite. There are a few short steps, and you can skip any of them.") + "</div>" +
        '<div class="vx-prow">' + ic("wave") + "<div><b>" + L("你的音频只留在这台 Mac 上", "Your audio stays on this Mac") + '</b><span class="vx-p">' + L("语音在这台 Mac 上识别，除非你选用远程语音识别服务。文字只发给你设置的语言模型。Voxt 没有自己的服务器，没有账号，也不做统计。", "Speech is recognised on this Mac unless you choose a remote speech service. Text goes only to the language model you set up. Voxt has no server of its own, no account and no analytics.") + "</span></div></div>" +
        '<div class="vx-prow">' + ic("db") + "<div><b>" + L("历史只保存在这台 Mac 上", "History stays on this Mac") + '</b><span class="vx-p">' + L("结果（以及你选择保留的录音）只保存在本机。可以在“设置 → 隐私”里关闭。", "Results, and their recordings if you choose, are kept only on this Mac. You can turn this off in Settings → Privacy.") + "</span></div></div></div>" +
        '<div class="vx-gpane" id="pp"><div class="vx-h2">' + L("权限", "Permissions") + '</div><div class="vx-p">' + L("Voxt 需要麦克风来听你说话，需要辅助功能把结果输入到光标处。在这里允许，或以后在“设置 → 权限”里允许。", "Voxt needs the microphone to hear you, and Accessibility to type the result where your cursor is. Allow them here, or later in Settings → Permissions.") + "</div>" +
        '<div class="vx-p" style="margin:8px 0">' + L("macOS 会依次询问：先是麦克风，然后是辅助功能。", "macOS asks for each in turn: the microphone, then Accessibility.") + '</div><div class="vx-card" style="padding:2px 14px">' +
        '<div class="vx-permrow"><div class="nm"><b>' + L("麦克风", "Microphone") + '</b><span class="vx-tag">' + L("必需", "Required") + '</span></div><span class="vx-status" id="st-mic">' + notAllowed + '</span><span class="vx-btn" id="b-mic">' + L("请求授权", "Ask") + "</span></div>" +
        '<div class="vx-permrow"><div class="nm"><b>' + L("辅助功能", "Accessibility") + '</b><span class="vx-tag">' + L("必需", "Required") + '</span><span class="d">' + L("让 Voxt 能输入结果、读取选中的文字，并用右 ⌥ 这样的单键作快捷键。", "Lets Voxt type results, read the selected text, and use single keys like Right ⌥ as shortcuts.") + '</span></div><span class="vx-status" id="st-ax">' + notAllowed + '</span><span class="vx-btn" id="b-ax">' + L("打开系统设置", "Open System Settings") + "</span></div></div></div>" +
        '<div class="vx-gpane" id="pm"><div class="vx-h2">' + L("模型", "Model") + '</div><div class="vx-p">' + L("Voxt 用这台 Mac 上的模型识别语音，用你设置的语言模型整理、翻译和改写。", "Voxt recognises speech with a model on this Mac and cleans up, translates and rewrites with a language model you set up.") + "</div></div>" +
        "</div>" +
        '<div class="vx-gfoot"><span class="vx-btn txt vx-hide" id="back">‹ ' + L("上一步", "Back") + '</span><span class="sp"></span>' +
        '<span class="vx-gone" id="fw"><span class="vx-btn txt">' + L("稍后设置", "Set up later") + '</span> <span class="vx-btn pri" id="b-start">' + L("开始设置", "Get started") + "</span></span>" +
        '<span class="vx-gone" id="fp"><span class="vx-btn txt">' + L("跳过这一步", "Skip this step") + '</span> <span class="vx-btn pri" id="b-cont">' + L("继续", "Continue") + "</span></span>" +
        '<span class="vx-gone" id="fm"><span class="vx-btn txt">' + L("跳过这一步", "Skip this step") + '</span> <span class="vx-btn pri">' + L("下载并继续（2.5 GB）", "Download and continue (2.5 GB)") + "</span></span></div></div>" +
        '<div class="vx-alert" id="alert" style="left:195px;top:120px"><img alt="" src="' + ICON + '"><b>' + L("“Voxt”想要访问麦克风。", "“Voxt” would like to access the microphone.") + "</b><p>" + L("Voxt 需要麦克风来听你说话。", "Voxt needs the microphone to hear you.") + '</p><div class="bs"><span>' + L("不允许", "Don’t Allow") + '</span><span class="pri" id="a-ok">' + L("允许", "Allow") + "</span></div></div>" +
        '<div class="vx-alert" id="sys" style="left:170px;top:130px;width:300px;text-align:left;padding:14px 16px"><b style="margin-bottom:10px">' + L("隐私与安全性 › 辅助功能", "Privacy & Security › Accessibility") + '</b><div style="display:flex;align-items:center;gap:10px;background:#fff;border-radius:8px;padding:8px 10px"><img alt="" src="' + ICON + '" style="width:26px;height:26px;margin:0"><span style="flex:1;font-weight:600">Voxt</span><span class="vx-tog" id="tog"></span></div></div>',
      cur: [[0, [470, 300]], [0.4, [470, 300]], [1.3, "#b-start"], [1.9, "#b-start"], [2.7, "#b-mic"], [3.2, "#b-mic"], [3.9, "#a-ok"], [4.6, "#a-ok"], [5.3, "#b-ax"], [5.9, "#b-ax"], [6.6, "#tog"], [7.6, "#tog"], [8.6, "#b-cont"], [9.4, "#b-cont"], [10.2, [480, 330]]],
      clk: [[1.45, "#b-start"], [2.85, "#b-mic"], [4.05, "#a-ok"], [5.45, "#b-ax"], [6.8, "#tog"], [8.75, "#b-cont"]],
      on: [[0, 1.55, "#pw"], [0, 1.55, "#fw"], [1.55, 8.85, "#pp"], [1.55, 8.85, "#fp"], [8.85, 99, "#pm"], [8.85, 99, "#fm"], [1.55, 99, "#back"],
        [3.0, 4.2, "#alert"], [5.65, 7.6, "#sys"], [6.9, 99, "#tog"], [4.25, 99, "#st-mic", "ok"], [7.7, 99, "#st-ax", "ok"]],
      tick: function (t, c) {
        crumbAt(c.box, t < 1.55 ? 0 : t < 8.85 ? 1 : 2);
        c.q("#st-mic").textContent = t >= 4.25 ? allowed : notAllowed;
        c.q("#st-ax").textContent = t >= 7.7 ? allowed : notAllowed;
      }
    };
  };

  SC["guide-model"] = function () {
    return {
      w: 640, h: 440, dur: 6.4, label: L("设置引导：下载并继续", "Setup guide: download and continue"),
      html: '<div class="vx-dlg" style="left:30px;top:16px;width:580px;height:408px">' + crumb() +
        '<div class="vx-gbody">' +
        '<div class="vx-gpane" id="pm"><div class="vx-h2">' + L("模型", "Model") + '</div><div class="vx-p">' + L("Voxt 用这台 Mac 上的模型识别语音，用你设置的语言模型整理、翻译和改写。下面这个模型最适合起步，“语音识别模型”页还有其他模型。", "Voxt recognises speech with a model on this Mac and cleans up, translates and rewrites with a language model you set up. The model below is the best start, and the Speech models page has others.") + "</div>" +
        '<div class="vx-card" style="margin-top:12px"><div style="font-size:12px;color:#8e8e93;font-weight:600">' + L("语音识别", "Speech recognition") + '</div><div style="display:flex;align-items:center;margin-top:4px"><b style="flex:1;font-size:14px">Qwen3-ASR 1.7B (8bit)</b><span style="color:#8e8e93;font-size:12px">' + L("还需下载 2.5 GB", "2.5 GB still to download") + '</span></div><div style="color:#1f7a63;font-size:12px;font-weight:600;margin-top:6px">' + L("选择其他模型…", "Choose another model…") + "</div></div>" +
        '<div class="vx-card" style="margin-top:10px"><div style="font-size:12px;color:#8e8e93;font-weight:600">' + L("语言模型", "Language model") + '</div><div class="vx-p" style="margin-top:3px">' + L("用来整理、翻译和改写文字。可以用 Ollama、LM Studio 或 oMLX 在这台 Mac 上运行一个，也可以使用服务商的 API Key。", "Cleans up, translates and rewrites the text. Run one on this Mac with Ollama, LM Studio or oMLX, or use a service's API key.") + '</div><span class="vx-btn" style="margin-top:8px">' + L("设置语言模型", "Set up a language model") + "</span></div></div>" +
        '<div class="vx-gpane" id="pt"><div class="vx-h2">' + L("试一试转录", "Try Transcribe") + '</div><div class="vx-p">' + L("转录把你说的话写成文字。按下快捷键（或点“开始”），读下面这句话，再按一次结束。", "Transcribe writes down what you say. Press the shortcut (or Start), read the sentence below, then press it again to stop.") + "</div>" +
        '<div class="vx-banner" style="margin-top:12px">' + ic("dl") + '<div style="flex:1">' + L("模型会在后台下载，可以先做下一步，进度条走满就能用了。", "The model downloads in the background, so you can continue. It is ready when the bar fills.") + '<div style="display:flex;align-items:center;gap:8px;margin-top:6px"><div class="vx-prog" style="flex:1"><i id="dlp"></i></div><span id="dlt" style="font-size:11.5px;color:#6e6e73;min-width:96px;text-align:right"></span></div></div></div>' +
        '<div class="vx-card" style="margin-top:10px;background:#f6f6f8;border-color:#f6f6f8"><div style="font-size:12px;color:#8e8e93;font-weight:600">' + L("读这句话，或者随便说点什么", "Read this, or say anything") + '</div><div style="margin-top:3px">' + L("嗯，那个设计评审改到周四下午，不对，周五上午，议程我今晚发出来。", "Um, so the design review moves to Thursday afternoon, no, Friday morning, and I'll send the agenda tonight.") + "</div></div></div>" +
        "</div>" +
        '<div class="vx-gfoot"><span class="vx-btn txt">‹ ' + L("上一步", "Back") + '</span><span class="sp"></span><span class="vx-btn txt">' + L("跳过这一步", "Skip this step") + '</span><span class="vx-gone" id="fm"><span class="vx-btn pri" id="b-dl">' + L("下载并继续（2.5 GB）", "Download and continue (2.5 GB)") + '</span></span><span class="vx-gone" id="ft"><span class="vx-btn dis">' + L("继续", "Continue") + "</span></span></div></div>",
      cur: [[0, [300, 330]], [0.4, [300, 330]], [1.4, "#b-dl"], [2.0, "#b-dl"], [2.9, [430, 330]]],
      clk: [[1.55, "#b-dl"]],
      on: [[0, 1.65, "#pm"], [0, 1.65, "#fm"], [1.65, 99, "#pt"], [1.65, 99, "#ft"]],
      p: [[1.9, 6.4, "#dlp", 1]],
      tick: function (t, c) {
        crumbAt(c.box, t < 1.65 ? 2 : 3);
        var f = Math.max(0, Math.min(1, (t - 1.9) / 4.5)) * 0.42;
        c.q("#dlt").textContent = (f * 2.5).toFixed(2) + " / 2.5 GB";
      }
    };
  };

  SC["asr-download"] = function () {
    return {
      w: 640, h: 400, dur: 7.6, label: L("语音识别模型页：下载 Qwen3-ASR 1.7B (8bit)", "Speech models page: download Qwen3-ASR 1.7B (8bit)"),
      html: '<div class="vx-win" style="left:14px;top:12px;width:612px;height:376px">' + sidebar("asr") +
        '<div class="vx-main"><div style="display:flex;justify-content:space-between"><span class="vx-h1">' + L("语音识别模型", "Speech models") + '</span><span style="font-weight:600">' + L("网络…", "Network…") + "</span></div>" +
        '<div class="vx-sec" style="margin-top:10px">' + L("本机", "On this Mac") + '</div><div class="vx-note">' + L("免费，音频不离开这台 Mac。", "Free, and your audio stays on this Mac.") + "</div>" +
        '<div class="vx-card" style="padding:10px 14px"><div style="display:flex;align-items:center;gap:8px">' + ic("wave", ' style="width:16px;height:16px;color:#6e6e73"') + '<b>Qwen3-ASR</b><span style="color:#8e8e93" id="cnt"></span></div>' +
        '<div style="border-top:1px solid #efeff2;margin-top:8px;padding-top:9px;display:flex;gap:10px"><div style="flex:1;min-width:0"><div style="display:flex;align-items:center;gap:6px"><b style="font-size:13.5px;white-space:nowrap">Qwen3-ASR 1.7B (8bit)</b><span class="vx-tag">' + L("默认", "Default") + '</span><span class="vx-ok vx-hide" id="inst">' + ic("checkc") + L("已安装", "Installed") + "</span></div>" +
        '<div style="color:#6e6e73;font-size:12px" id="st"></div><div style="margin-top:4px;font-size:12px">' + L("最准确。能识别口音、方言、人名，以及一句话里夹杂的多种语言，边说边出文字。", "The most accurate. Good with accents, dialects, names and several languages in one sentence. Shows the text as you speak.") + "</div>" +
        '<div style="display:flex;gap:5px;margin-top:6px;flex-wrap:wrap"><span class="vx-tag">' + L("本地", "Local") + '</span><span class="vx-tag">' + L("准确", "Accurate") + '</span><span class="vx-tag">' + L("实时", "Realtime") + '</span><span class="vx-tag">' + L("支持主要语言", "Supports your main language") + "</span></div>" +
        '<div class="vx-prog vx-hide" id="bar" style="margin-top:9px"><i id="barp"></i></div></div>' +
        '<div style="display:flex;align-items:flex-start;gap:6px"><span class="vx-btn txt" id="hide1">' + L("隐藏", "Hide") + '</span><span class="vx-gone" id="g-dl"><span class="vx-btn" id="b-dl">' + ic("dl") + L("下载", "Download") + '</span></span><span class="vx-gone" id="g-cancel"><span class="vx-btn">' + L("取消", "Cancel") + '</span></span><span class="vx-gone" id="g-del"><span class="vx-btn txt">' + L("删除", "Delete") + "</span></span></div></div>" +
        '<div style="border-top:1px solid #efeff2;margin-top:9px;padding-top:9px;display:flex"><div style="flex:1"><b style="font-size:13.5px">Qwen3-ASR 0.6B (4bit)</b><div style="color:#6e6e73;font-size:12px">' + L("需下载 712.8 MB", "712.8 MB download") + '</div></div><span class="vx-btn txt">' + L("隐藏", "Hide") + '</span><span class="vx-btn">' + ic("dl") + L("下载", "Download") + "</span></div></div></div>" +
        '<div class="vx-toast" id="toast" style="top:14px;left:58%">' + L("Qwen3-ASR 1.7B (8bit) 已安装。", "Qwen3-ASR 1.7B (8bit) is installed.") + "</div></div>",
      cur: [[0, [420, 330]], [0.4, [420, 330]], [1.3, "#b-dl"], [1.9, "#b-dl"], [2.8, [470, 360]]],
      clk: [[1.45, "#b-dl"]],
      on: [[0, 1.55, "#g-dl"], [1.55, 5.6, "#g-cancel"], [1.55, 5.6, "#bar"], [5.6, 99, "#g-del"], [5.6, 99, "#inst"], [5.7, 7.3, "#toast"]],
      p: [[1.6, 5.5, "#barp", 1]],
      tick: function (t, c) {
        var f = Math.max(0, Math.min(1, (t - 1.6) / 3.9)), done = t >= 5.6;
        var got = f < 0.4 ? Math.round(f * 2500) + " MB" : (f * 2.5).toFixed(1) + " GB", pct = Math.round(f * 100) + "%";
        c.q("#st").textContent = t < 1.55 ? L("需下载 2.5 GB", "2.5 GB download") :
          !done ? L("正在下载：" + got + " / 2.5 GB（" + pct + "）", "Downloading: " + got + " of 2.5 GB (" + pct + ")") :
            L("占用 2.5 GB · 用于转录 · 翻译 · 改写", "2.5 GB on disk · for Transcribe · Translate · Rewrite");
        c.q("#hide1").style.visibility = done ? "hidden" : "";
        c.q("#cnt").textContent = done ? L("3 个模型 · 已安装 1 个", "3 models · 1 installed") : L("3 个模型", "3 models");
      }
    };
  };

  SC["llm-open"] = function () {
    return {
      w: 640, h: 400, dur: 4.6, label: L("打开语言模型页，点 OpenAI", "Open Language models and choose OpenAI"),
      html: '<div class="vx-win" style="left:14px;top:12px;width:612px;height:376px">' + sidebar("home", true) +
        '<div class="vx-main"><div class="vx-page" id="pg-home"><div class="vx-h1">' + L("首页", "Home") + '</div><div style="display:flex;gap:10px;margin-top:14px"><div class="vx-card" style="flex:1;height:80px;background:#fafafb"></div><div class="vx-card" style="flex:1;height:80px;background:#fafafb"></div></div><div class="vx-sec">' + L("最近结果", "Recent results") + '</div><div class="vx-card" style="height:120px;background:#fafafb"></div></div>' +
        '<div class="vx-page" id="pg-llm"><div class="vx-h1">' + L("语言模型", "Language models") + '</div><div class="vx-sub">' + L("Voxt 只把识别出的文字或选中的文字发给语言模型，不发送音频。", "Voxt sends the language model the recognised or selected text, never your audio.") + "</div>" +
        '<div class="vx-sec">' + L("本机", "On this Mac") + '</div><div class="vx-note">' + L("免费，文字不离开这台 Mac。", "Free, and the text stays on this Mac.") + '</div><div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">' + llmCard("Ollama") + llmCard("oMLX") + "</div>" +
        '<div class="vx-sec">' + L("在线服务", "Online services") + '</div><div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:6px">' + llmCard("Anthropic", 1) + llmCard("Google Gemini", 1) + llmCard("OpenAI", 1, "c-openai") + llmCard("DeepSeek", 1) + "</div></div></div></div>",
      cur: [[0, [420, 250]], [0.4, [420, 250]], [1.2, '[data-nav="llm"]'], [1.8, '[data-nav="llm"]'], [2.9, "#c-openai b"], [4.0, "#c-openai b"]],
      clk: [[1.35, '[data-nav="llm"]'], [3.2, "#c-openai b"]],
      on: [[0, 1.45, "#pg-home"], [1.45, 99, "#pg-llm"], [1.45, 99, '[data-nav="llm"]'], [0, 1.45, '[data-nav="home"]']],
      tick: function (t, c) {
        var o = c.q("#c-openai"), hot = t >= 2.8;
        o.style.borderColor = hot ? "#2b7a67" : ""; o.style.boxShadow = hot ? "0 0 0 3px rgba(43,122,103,.14)" : "";
        c.q("#note").style.opacity = t < 1.45 ? 1 : 0;
      }
    };
  };

  SC["llm-fill"] = function () {
    var other = L("其他模型…", "Other model…");
    return {
      w: 640, h: 510, dur: 8.6, label: L("填写 API Key、地址和模型 ID（已打码）", "Fill in the API key, address and model ID (hidden)"),
      kcPos: "left:auto;right:12px;bottom:auto;top:96px;flex-direction:column",
      html: openaiDialog(false),
      keys: [{ a: 1.55, b: 2.4, k: ["⌘", "V"] }, { a: 3.35, b: 4.2, k: ["⌘", "V"] }, { a: 6.85, b: 7.7, k: ["⌘", "V"] }],
      cur: [[0, [380, 420]], [0.4, [380, 420]], [1.1, "#f-key"], [2.4, "#f-key"], [2.9, "#f-addr"], [4.3, "#f-addr"], [4.8, "#f-model"], [5.1, "#f-model"], [5.6, "#opt-other"], [6.1, "#opt-other"], [6.45, "#f-id"], [7.9, "#f-id"], [8.4, [400, 420]]],
      clk: [[1.25, "#f-key"], [3.05, "#f-addr"], [4.95, "#f-model"], [5.85, "#opt-other"], [6.55, "#f-id"]],
      on: [[1.25, 3.05, "#f-key", "focus"], [3.05, 4.95, "#f-addr", "focus"], [6.55, 99, "#f-id", "focus"], [5.0, 5.95, "#menu"], [5.55, 5.95, "#opt-other", "hl"]],
      p: [[1.75, 2.25, '[data-p="key"]'], [3.55, 4.05, '[data-p="addr"]'], [7.05, 7.55, '[data-p="id"]']],
      tick: function (t, c) {
        var o = t >= 5.95, mt = c.q("#model-text");
        if (o !== (mt.dataset.o === "1")) { mt.dataset.o = o ? "1" : "0"; mt.innerHTML = o ? other : mask("130px"); }
        c.q("#f-idwrap").style.height = (Math.max(0, Math.min(1, (t - 5.95) / 0.3)) * 44) + "px";
      }
    };
  };

  SC["llm-test"] = function () {
    var none = L("未选择模型", "Model not chosen");
    return {
      w: 640, h: 510, dur: 5.6, label: L("测试连接，然后保存", "Test the connection, then save"),
      html: '<div class="vx-page" id="after" style="inset:40px 50px"><div class="vx-h1">' + L("语言模型", "Language models") + '</div><div class="vx-sec">' + L("在线服务", "Online services") + '</div><div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:6px">' +
        llmCard("Anthropic", 1) + llmCard("Google Gemini", 1) + llmCard("OpenAI", 1, "c-openai") + llmCard("DeepSeek", 1) + "</div></div>" + openaiDialog(true, 60),
      cur: [[0, [300, 330]], [0.4, [300, 330]], [1.0, "#b-test"], [2.6, "#b-test"], [3.3, "#b-save"], [3.8, "#b-save"], [4.4, [420, 400]]],
      clk: [[1.15, "#b-test"], [3.45, "#b-save"]],
      on: [[1.25, 2.3, "#testing"], [2.3, 99, "#test-ok"], [3.6, 99, "#after"]],
      tick: function (t, c) {
        c.q("#spin").style.transform = "rotate(" + (t * 540) + "deg)";
        var d = c.q("#dlg"), k = Math.max(0, Math.min(1, (t - 3.55) / 0.25));
        d.style.opacity = 1 - k; d.style.transform = "scale(" + (1 - 0.04 * k) + ")";
        var o = c.q("#c-openai"), saved = t >= 3.8;
        o.style.borderColor = saved ? "#2b7a67" : "";
        o.querySelector(".dot").style.background = saved ? "#34c759" : "";
        var m = o.querySelector(".m");
        if (saved !== (m.dataset.s === "1")) { m.dataset.s = saved ? "1" : "0"; m.innerHTML = saved ? mask("110px") : none; }
      }
    };
  };

  SC["llm-use"] = function () {
    var notSet = L("未设置", "Not set up");
    return {
      w: 640, h: 360, dur: 4.6, label: L("在翻译页选用刚保存的语言模型", "Choose the saved language model on the Translate page"),
      html: '<div class="vx-win" style="left:14px;top:12px;width:612px;height:336px">' + sidebar("translate") +
        '<div class="vx-main"><div class="vx-h1">' + L("翻译", "Translate") + '</div><div class="vx-sub">' + L("把选中的文字或你说的话译成你选的语言。", "Turns the selected text, or what you say, into the language you choose.") + "</div>" +
        '<div class="vx-card" style="margin-top:14px;padding:4px 14px">' +
        [["<span>" + L("快捷键", "Shortcut") + '</span><span class="vx-tag" style="font-size:12px;font-weight:600">⇧ ' + rOpt() + "</span>"],
          ["<span>" + L("目标语言", "Target language") + '</span><div class="vx-sel" style="width:190px"><span>' + L("英语", "Chinese") + "</span>" + ic("chev") + "</div>"],
          ["<span>" + L("语音识别模型", "Speech model") + '</span><div class="vx-sel" style="width:190px"><span>Qwen3-ASR 1.7B (8bit)</span>' + ic("chev") + "</div>"],
          ["<span>" + L("语言模型", "Language model") + '</span><div class="vx-sel" id="s-llm" style="width:190px"><span id="llm-t">' + notSet + "</span>" + ic("chev") + "</div>"]].map(function (r) {
            return '<div style="display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f0f0f2">' + r[0] + "</div>";
          }).join("") + "</div>" +
        '<div class="vx-menu" id="menu" style="right:24px;top:236px;width:190px"><div id="opt">OpenAI · ' + mask("70px") + "</div></div></div></div>",
      cur: [[0, [330, 300]], [0.4, [330, 300]], [1.2, "#s-llm"], [1.6, "#s-llm"], [2.2, "#opt"], [2.6, "#opt"], [3.3, [380, 320]]],
      clk: [[1.35, "#s-llm"], [2.45, "#opt"]],
      on: [[1.45, 2.55, "#menu"], [2.1, 2.55, "#opt", "hl"]],
      tick: function (t, c) {
        var e = c.q("#llm-t"), s = t >= 2.55;
        if (s !== (e.dataset.s === "1")) { e.dataset.s = s ? "1" : "0"; e.innerHTML = s ? "OpenAI · " + mask("54px") : notSet; }
      }
    };
  };

  // Shortcut scenes. o.keys, o.icon, o.live (spoken), o.work (overlay label), o.result, o.sel (selected text the result replaces).
  function shortcutScene(o) {
    var sel = o.sel, tl = o.tl;
    var recordingLabel = { mic: L("转录", "Transcribe"), globe: L("翻译", "Translate"), pen: L("改写", "Rewrite") }[o.icon];
    var body = (o.head ? '<div class="muted">' + o.head + "</div>" : "") +
      (sel ? '<span class="vx-selt" id="selt">' + sel + "</span>" : "") + '<span id="out"></span><span class="vx-caret" id="caret"></span>';
    var sc = {
      w: 640, h: 330, dur: tl.end, label: o.label,
      html: editor({ title: o.title, body: body }) + (o.card ? '<div class="vx-acard" id="card" style="top:150px"><div class="hd">' + ic("pen", ' style="width:13px;height:13px"') + L("改写", "Rewrite") + '<span class="sp"></span><span class="cap">' + L("继续", "Continue") + '</span><span class="ib" id="b-ins">' + ic("pen") + L("插入", "Insert") + '</span><span class="ib">' + ic("copy") + L("复制", "Copy") + '</span><span class="ib">' + ic("x") + '</span></div><div class="bd" id="ans"></div></div>' : ""),
      keys: tl.keys.map(function (k) { return { a: k[0], b: k[1], k: o.keys }; }),
      on: [[tl.rec[0], tl.work[1], "#pill"], [tl.work[0], tl.work[1], "#pill", "work"]].concat(o.live ? [[tl.live[0], tl.work[1], "#pill", "live"]] : []).concat(sel ? [[0, tl.put, "#selt"]] : []).concat(o.card ? [[tl.card[0], tl.card[1], "#card"], [tl.ins - 0.4, tl.card[1], "#b-ins", "hot"]] : []),
      type: [].concat(o.live ? [[tl.live[0], tl.live[1], "#live", o.live]] : []).concat(o.card ? [[tl.card[0], tl.card[0] + 1.1, "#ans", o.result]] : []).concat([[tl.put, tl.put + (o.instant ? 0.01 : 0.6), "#out", o.result]]),
      tick: function (t, c) {
        tickWave(c.box, t, t >= tl.rec[0] && t < tl.work[0] && !!o.live);
        c.q("#plbl").textContent = t >= tl.work[0] ? o.work : recordingLabel;
        if (sel) c.q("#selt").style.display = t >= tl.put ? "none" : "";
        c.q("#caret").style.display = sel && t < tl.put ? "none" : "";
      }
    };
    if (o.card) { sc.curFrom = tl.card[0] + 0.3; sc.cur = [[0, [600, 320]], [tl.card[0] + 0.6, [600, 320]], [tl.ins - 0.3, "#b-ins"], [tl.ins + 0.4, "#b-ins"], [tl.end, [560, 320]]]; sc.clk = [[tl.ins, "#b-ins"]]; }
    return sc;
  }

  SC["dictate"] = function () {
    return shortcutScene({
      label: L("轻按右 ⌥ 说话，再按一次，文字输入到光标处", "Tap Right ⌥, speak, tap it again: the text goes in at the cursor"),
      keys: [rOpt()], icon: "mic", title: L("备忘录", "Notes"), head: L("周会记录", "Weekly sync"),
      live: L("嗯，明天下午三点开会，记得带上周报", "um, the meeting is tomorrow at 3 pm, bring the weekly report"),
      work: L("识别中…", "Recognising…"), result: L("明天下午三点开会，记得带上周报。", "The meeting is tomorrow at 3 pm. Bring the weekly report."),
      tl: { keys: [[0.7, 1.5], [4.6, 5.4]], rec: [0.85], live: [1.5, 4.1], work: [4.75, 5.8], put: 5.9, end: 7.6 }
    });
  };
  SC["translate-voice"] = function () {
    return shortcutScene({
      label: L("⇧ + 右 ⌥ 说中文，输出英文", "⇧ + Right ⌥: speak English, get Chinese"),
      keys: ["⇧", rOpt()], icon: "globe", title: L("邮件", "Mail"), head: L("Hi team,", "大家好："),
      live: L("请在周五前确认时间", "please confirm the time by Friday"), work: L("正在翻译…", "Translating…"),
      result: L("Please confirm the time by Friday.", "请在周五前确认时间。"),
      tl: { keys: [[0.7, 1.5], [3.9, 4.7]], rec: [0.85], live: [1.5, 3.4], work: [4.05, 5.3], put: 5.4, end: 7.2 }
    });
  };
  SC["translate-select"] = function () {
    return shortcutScene({
      label: L("选中文字，按 ⇧ + 右 ⌥，译文替换选区", "Select text and press ⇧ + Right ⌥: the translation replaces it"),
      keys: ["⇧", rOpt()], icon: "globe", title: L("邮件", "Mail"), head: L("Hi team,", "大家好："),
      sel: L("这个周末我们去山里露营吧，记得带上厚外套和手电筒。", "Let's go camping in the mountains this weekend; remember to bring a warm jacket and a torch."),
      work: L("正在翻译…", "Translating…"), instant: true,
      result: L("Let's go camping in the mountains this weekend. Remember to bring a warm jacket and a flashlight.", "这个周末我们去山里露营吧，记得带上厚外套和手电筒。"),
      tl: { keys: [[1.0, 1.8]], rec: [1.15], work: [1.15, 2.9], put: 3.0, end: 5.4 }
    });
  };
  SC["rewrite-card"] = function () {
    return shortcutScene({
      label: L("选中文字，⌃ + 右 ⌥ 说要求，在卡片里插入", "Select text, press ⌃ + Right ⌥ and say what you want, then insert from the card"),
      keys: ["⌃", rOpt()], icon: "pen", title: L("消息", "Messages"), head: L("发给：小组群", "To: Team chat"),
      sel: L("我明天有事，咱们换个时间吧", "can't make it tomorrow, let's pick another time"),
      live: L("改成礼貌简洁的改期消息", "make it a short, polite message to reschedule"), work: L("正在生成…", "Writing…"),
      result: L("抱歉，我明天临时有事，能否把会议改到其他时间？谢谢理解！", "Sorry, something came up tomorrow. Could we move the meeting to another time? Thanks for understanding!"),
      instant: true, card: true,
      tl: { keys: [[0.7, 1.5], [3.9, 4.7]], rec: [0.85], live: [1.5, 3.4], work: [4.05, 5.0], card: [5.05, 8.0], ins: 7.6, put: 8.0, end: 9.6 }
    });
  };

  // ---------- player ----------
  var clamp = function (x) { return Math.max(0, Math.min(1, x)); };
  var ease = function (x) { return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var HOLD = 1.3;

  function Player(el, sc, lang) {
    this.sc = sc; this.t = 0; this.paused = false; this.visible = false; this.s = 1;
    var en = lang === "en";
    var f = this.frame = document.createElement("div");
    f.className = "vx-frame"; f.tabIndex = 0; f.setAttribute("role", "img");
    f.setAttribute("aria-label", sc.label + (en ? " (click to pause or play)" : "（点击暂停或播放）"));
    f.style.paddingTop = (sc.h / sc.w * 100) + "%";
    var b = this.box = document.createElement("div");
    b.className = "vx-box"; b.lang = en ? "en" : "zh-CN"; b.style.width = sc.w + "px"; b.style.height = sc.h + "px";
    b.innerHTML = sc.html +
      '<div class="vx-kc" id="kc"' + (sc.kcPos ? ' style="' + sc.kcPos + '"' : "") + '></div><div class="vx-rip"></div>' +
      '<svg class="vx-cur" viewBox="0 0 19 27"><path d="M1.5 1.5v20.5l5.2-5 3.9 8.6 3.4-1.6-3.8-8.4h7.3z" fill="#111" stroke="#fff" stroke-width="1.5" stroke-linejoin="round"/></svg>';
    f.appendChild(b);
    var pp = document.createElement("div"); pp.className = "vx-pp"; pp.textContent = en ? "Paused · click to play" : "已暂停 · 点击播放"; f.appendChild(pp);
    el.innerHTML = ""; el.appendChild(f);
    this.cursor = b.querySelector(".vx-cur"); this.rip = b.querySelector(".vx-rip"); this.kc = b.querySelector("#kc");
    if (reduce) this.cursor.style.display = "none";
    var self = this;
    var toggle = function () { self.paused = !self.paused; f.classList.toggle("paused", self.paused); self.kick(); };
    f.addEventListener("click", toggle);
    f.addEventListener("keydown", function (e) { if (e.key === " " || e.key === "Enter") { e.preventDefault(); toggle(); } });
    var fit = function () { self.s = f.clientWidth / sc.w || 1; b.style.transform = "scale(" + self.s + ")"; };
    fit();
    if (window.ResizeObserver) new ResizeObserver(function () { fit(); self.render(self.t); }).observe(f);
    if (window.IntersectionObserver) new IntersectionObserver(function (es) { self.visible = es[0].isIntersecting; self.kick(); }, { threshold: 0.25 }).observe(f);
    else this.visible = true;
    this.render(reduce ? sc.dur : 0);
    this.kick();
  }
  Player.prototype.q = function (sel) { return this.box.querySelector(sel); };
  Player.prototype.pt = function (target) {
    if (Array.isArray(target)) return target;
    var e = this.q(target); if (!e) return [0, 0];
    var r = e.getBoundingClientRect(), o = this.box.getBoundingClientRect();
    return [(r.left - o.left + r.width * 0.5) / this.s, (r.top - o.top + r.height * 0.55) / this.s];
  };
  Player.prototype.cursorAt = function (t) {
    var w = this.sc.cur, i = 0;
    while (i < w.length - 1 && w[i + 1][0] <= t) i++;
    var a = w[i], b = w[Math.min(i + 1, w.length - 1)];
    var pa = this.pt(a[1]), pb = this.pt(b[1]);
    var k = t <= a[0] ? 0 : b[0] === a[0] ? 1 : ease(clamp((t - a[0]) / (b[0] - a[0])));
    return [pa[0] + (pb[0] - pa[0]) * k, pa[1] + (pb[1] - pa[1]) * k];
  };
  Player.prototype.render = function (t) {
    var sc = this.sc, box = this.box, self = this, i;
    this.t = t;
    (sc.on || []).forEach(function (r) {
      var on = t >= r[0] && t < r[1];
      box.querySelectorAll(r[2]).forEach(function (e) { e.classList.toggle(r[3] || "on", on); });
    });
    (sc.p || []).forEach(function (r) {
      var k = clamp((t - r[0]) / (r[1] - r[0])); if (!r[3]) k = ease(k);
      box.querySelectorAll(r[2]).forEach(function (e) { e.style.setProperty("--p", k); });
    });
    (sc.type || []).forEach(function (r) {
      var e = self.q(r[2]), n = Math.round(r[3].length * clamp((t - r[0]) / (r[1] - r[0])));
      if (e && e.textContent.length !== n) e.textContent = r[3].slice(0, n);
    });
    this.cur = sc.cur ? this.cursorAt(t) : [0, 0];
    if (sc.tick) sc.tick(t, this);
    if (sc.cur) {
      this.cursor.style.opacity = sc.curFrom ? clamp((t - sc.curFrom) / 0.25) : 1;
      this.cursor.style.transform = "translate(" + (this.cur[0] - 2) + "px," + (this.cur[1] - 2) + "px)";
    } else this.cursor.style.opacity = 0;
    var rv = 0;
    (sc.clk || []).forEach(function (c) {
      var dt = t - c[0], tgt = self.q(c[1]);
      if (tgt) tgt.classList.toggle("down", dt >= 0 && dt < 0.16);
      if (dt >= 0 && dt < 0.5) {
        var p = self.pt(c[1]); self.rip.style.left = p[0] + "px"; self.rip.style.top = p[1] + "px";
        self.rip.style.transform = "scale(" + (0.5 + dt * 2) + ")"; rv = 1 - dt / 0.5;
      }
    });
    this.rip.style.opacity = rv;
    var kOn = null;
    for (i = 0; i < (sc.keys || []).length; i++) { var k = sc.keys[i]; if (t >= k.a && t < k.b) kOn = k; }
    this.kc.classList.toggle("on", !!kOn);
    if (kOn) {
      var html = kOn.k.map(function (x) { return "<span>" + x + "</span>"; }).join("<em>+</em>");
      if (this.kc.innerHTML !== html) this.kc.innerHTML = html;
      this.kc.classList.toggle("dn", t - kOn.a < 0.3);
    }
  };
  Player.prototype.kick = function () {
    var self = this;
    var run = this.visible && !this.paused && !reduce;
    if (run && !this.raf) {
      var start = performance.now() - this.t * 1000;
      var loop = function (now) {
        var total = self.sc.dur + HOLD, t = ((now - start) / 1000) % total;
        self.render(Math.min(t, self.sc.dur));
        self.box.style.opacity = t > total - 0.35 ? (total - t) / 0.35 : t < 0.25 ? t / 0.25 : 1;
        self.raf = requestAnimationFrame(loop);
      };
      this.raf = requestAnimationFrame(loop);
    } else if (!run && this.raf) {
      cancelAnimationFrame(this.raf); this.raf = 0; this.box.style.opacity = 1;
    }
  };

  function init() {
    document.querySelectorAll(".vx-anim[data-scene]").forEach(function (el) {
      var make = SC[el.dataset.scene];
      if (!make || el.vxPlayer) return;
      LANG = el.dataset.lang === "en" ? "en" : "zh";
      el.vxPlayer = new Player(el, make(), LANG);
    });
  }
  window.VoxtAnim = { scenes: SC, init: init };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
