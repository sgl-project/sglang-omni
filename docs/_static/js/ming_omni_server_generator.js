// Server command generator for Ming-flash-omni-2.0, mounting into
// #sgl-ming-server-gen-mount. Sibling of qwen3_omni_server_generator.js.
(function () {
  'use strict';

  var MODEL_PATH = 'inclusionAI/Ming-flash-omni-2.0';

  var MODES = {
    'text-only': { label: 'Text only',  subtitle: 'thinker only', audio: 'Text output only' },
    'speech':    { label: 'Text + Audio', subtitle: '+ talker',   audio: 'Text + Audio output' },
  };

  var THINKER_TP = {
    'tp1': { label: 'TP=1', subtitle: '1 GPU',  n: 1 },
    'tp2': { label: 'TP=2', subtitle: '2 GPUs', n: 2 },
    'tp4': { label: 'TP=4', subtitle: '4 GPUs', n: 4 },
  };

  var VISION_TP = {
    'off': { label: 'Off',  subtitle: 'TP=1', n: 1 },
    'tp2': { label: 'TP=2', subtitle: '2 ranks', n: 2 },
    'tp4': { label: 'TP=4', subtitle: '4 ranks', n: 4 },
  };

  // Vision encoder defaults to GPU 0 with the thinker; 'colo' reuses thinker
  // GPUs and is valid only when vision TP <= thinker TP.
  var VISION_PLACE = {
    'colo':      { label: 'With thinker', subtitle: 'shares thinker GPUs' },
    'dedicated': { label: 'Dedicated',    subtitle: 'separate GPUs' },
  };

  // Larger-VRAM tiers fit the weights and still need headroom for the KV pool,
  // so they take a higher static fraction than H100 80GB.
  var HARDWARE = {
    'h100': { label: 'H100', subtitle: '80 GB',  mem: '0.80' },
    'h200': { label: 'H200', subtitle: '141 GB', mem: '0.90' },
  };

  function range(start, count) {
    var out = [];
    for (var i = 0; i < count; i++) out.push(start + i);
    return out;
  }

  function coloValid(ctx) {
    return VISION_TP[ctx.vtp].n <= THINKER_TP[ctx.tp].n;
  }

  function plan(ctx) {
    var isSpeech = ctx.mode === 'speech';
    var tT = THINKER_TP[ctx.tp].n;
    var vT = VISION_TP[ctx.vtp].n;
    var visionOn = vT > 1;
    var colo = visionOn && ctx.vplace === 'colo' && coloValid(ctx);

    var thinkerGpus = range(0, tT);
    var visionGpus = null;
    if (colo) visionGpus = range(0, vT);

    var next = tT;
    var talkerGpu = null;
    if (isSpeech) { talkerGpu = next; next += 1; }

    if (visionOn && !colo) { visionGpus = range(next, vT); next += vT; }

    return {
      isSpeech: isSpeech, tT: tT, vT: vT, visionOn: visionOn, colo: colo,
      thinkerGpus: thinkerGpus, talkerGpu: talkerGpu, visionGpus: visionGpus,
      totalGpus: next,
    };
  }

  function buildCommand(ctx) {
    var p = plan(ctx);
    var parts = [
      '--model-path ' + MODEL_PATH,
      '--model-name ming-omni',
    ];
    if (!p.isSpeech) parts.push('--text-only');
    parts.push('--thinker-tp-size ' + p.tT);
    parts.push('--thinker-gpus ' + p.thinkerGpus.join(','));
    if (p.talkerGpu !== null) parts.push('--talker-gpu ' + p.talkerGpu);
    if (p.visionGpus) {
      parts.push('--image-encoder-tp-size ' + p.vT);
      parts.push('--image-encoder-gpus ' + p.visionGpus.join(','));
    }
    parts.push('--cpu-offload-gb 0');
    parts.push('--mem-fraction-static ' + HARDWARE[ctx.hw].mem);
    parts.push('--port 8000');

    var prefix = 'CUDA_VISIBLE_DEVICES=' + range(0, p.totalGpus).join(',');
    var cmd = prefix + ' sgl-omni serve';
    for (var i = 0; i < parts.length; i++) cmd += ' \\\n  ' + parts[i];
    return cmd;
  }

  function getExplanations(ctx) {
    var p = plan(ctx);
    var items = [];
    if (!p.isSpeech) {
      items.push({ flag: '--text-only', desc: 'Thinker-only pipeline — no talker, no audio output' });
    }
    items.push({ flag: '--thinker-tp-size ' + p.tT, desc: 'Tensor-parallel the thinker across ' + p.tT + ' GPU' + (p.tT > 1 ? 's' : '') });
    items.push({ flag: '--thinker-gpus ' + p.thinkerGpus.join(','), desc: 'Thinker TP ranks on GPU' + (p.tT > 1 ? 's ' : ' ') + p.thinkerGpus.join(', ') });
    if (p.talkerGpu !== null) {
      items.push({ flag: '--talker-gpu ' + p.talkerGpu, desc: 'Dedicated talker GPU for speech output (must not overlap the thinker)' });
    }
    if (p.visionGpus) {
      items.push({ flag: '--image-encoder-tp-size ' + p.vT, desc: 'Tensor-parallel the image (vision) encoder across ' + p.vT + ' ranks' });
      items.push({ flag: '--image-encoder-gpus ' + p.visionGpus.join(','),
                   desc: p.colo
                     ? 'Vision-encoder ranks colocated on thinker GPUs ' + p.visionGpus.join(', ')
                     : 'Vision-encoder ranks on dedicated GPUs ' + p.visionGpus.join(', ') });
    }
    var hwDef = HARDWARE[ctx.hw];
    items.push({ flag: '--mem-fraction-static ' + hwDef.mem,
                 desc: 'Static memory fraction for ' + hwDef.label + ' (' + hwDef.subtitle + '); higher on larger-VRAM tiers so the KV pool fits after weights load' });
    return items;
  }

  function fallbackCopy(text, done) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0;';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try { document.execCommand('copy'); done(); } catch (e) {}
    document.body.removeChild(ta);
  }

  function detectDark() {
    var html = document.documentElement;
    return html.classList.contains('dark') ||
           html.getAttribute('data-theme') === 'dark' ||
           html.getAttribute('data-color-mode') === 'dark';
  }

  function render(mountEl) {
    var state = { mode: 'text-only', tp: 'tp4', vtp: 'off', vplace: 'colo', hw: 'h100' };
    var dark  = detectDark();

    function update() {
      dark = detectDark();

      var ACCENT     = '#D45D44';
      var cardBg     = dark ? '#1f2937' : '#fff';
      var cardBorder = dark ? '#374151' : '#e5e7eb';
      var titleColor = dark ? '#e5e7eb' : '#374151';
      var btnBg      = dark ? '#374151' : '#fff';
      var btnBorder  = dark ? '#9ca3af' : '#d1d5db';
      var btnColor   = dark ? '#e5e7eb' : '#374151';
      var cmdBg      = dark ? '#111827' : '#f5f5f5';
      var cmdColor   = dark ? '#e5e7eb' : '#374151';

      var containerS = 'max-width:900px;display:flex;flex-direction:column;gap:4px;'
                     + 'font-family:system-ui,-apple-system,BlinkMacSystemFont,sans-serif;';
      var cardS = 'padding:8px 12px;'
                + 'border:1px solid ' + cardBorder + ';'
                + 'border-left:3px solid ' + ACCENT + ';'
                + 'border-radius:4px;'
                + 'display:flex;align-items:center;gap:12px;'
                + 'background:' + cardBg + ';';
      var titleS = 'font-size:13px;font-weight:600;min-width:140px;flex-shrink:0;color:' + titleColor + ';';
      var itemsS = 'display:flex;row-gap:4px;column-gap:6px;flex-wrap:wrap;align-items:center;';
      var subS   = 'display:block;font-size:9px;margin-top:1px;line-height:1.1;opacity:0.75;';

      function btnS(active) {
        return 'padding:4px 12px;'
             + 'border:1px solid ' + (active ? ACCENT : btnBorder) + ';'
             + 'border-radius:3px;cursor:pointer;'
             + 'display:inline-flex;flex-direction:column;align-items:center;justify-content:center;'
             + 'font-weight:500;font-size:13px;user-select:none;'
             + 'min-width:60px;text-align:center;'
             + 'background:' + (active ? ACCENT : btnBg) + ';'
             + 'color:' + (active ? '#fff' : btnColor) + ';';
      }

      function makeRow(title, items, activeId) {
        var btns = items.map(function(it) {
          var isDisabled = !!it.disabled;
          var isActive = it.id === activeId && !isDisabled;
          var sub = it.subtitle ? '<small style="' + subS + '">' + it.subtitle + '</small>' : '';
          var extra = isDisabled ? 'opacity:0.4;cursor:not-allowed;' : '';
          return '<button data-dim="' + it.dim + '" data-val="' + it.id + '" '
               + (isDisabled ? 'data-disabled="1" ' : '')
               + 'style="' + btnS(isActive) + extra + '">' + it.label + sub + '</button>';
        }).join('');
        return '<div style="' + cardS + '">'
             + '<div style="' + titleS + '">' + title + '</div>'
             + '<div style="' + itemsS + '">' + btns + '</div>'
             + '</div>';
      }

      if (state.vtp !== 'off' && state.vplace === 'colo' && !coloValid(state)) {
        state.vplace = 'dedicated';
      }

      var ctx = state;
      var modeDef = MODES[ctx.mode];
      var p = plan(ctx);

      var modeItems = Object.keys(MODES).map(function(k) {
        return { id: k, dim: 'mode', label: MODES[k].label, subtitle: MODES[k].subtitle };
      });
      var html = makeRow('Mode', modeItems, ctx.mode);

      var tpItems = Object.keys(THINKER_TP).map(function(k) {
        return { id: k, dim: 'tp', label: THINKER_TP[k].label, subtitle: THINKER_TP[k].subtitle };
      });
      html += makeRow('Thinker TP', tpItems, ctx.tp);

      var vtpItems = Object.keys(VISION_TP).map(function(k) {
        return { id: k, dim: 'vtp', label: VISION_TP[k].label, subtitle: VISION_TP[k].subtitle };
      });
      html += makeRow('Vision TP', vtpItems, ctx.vtp);

      if (ctx.vtp !== 'off') {
        var coloOk = coloValid(ctx);
        var vplaceItems = Object.keys(VISION_PLACE).map(function(k) {
          var sub = VISION_PLACE[k].subtitle;
          if (k === 'colo' && !coloOk) sub = 'needs vision TP ≤ thinker TP';
          return { id: k, dim: 'vplace', label: VISION_PLACE[k].label, subtitle: sub,
                   disabled: k === 'colo' && !coloOk };
        });
        html += makeRow('Vision GPUs', vplaceItems, ctx.vplace);
      }

      var hwItems = Object.keys(HARDWARE).map(function(k) {
        return { id: k, dim: 'hw', label: HARDWARE[k].label, subtitle: HARDWARE[k].subtitle };
      });
      html += makeRow('Hardware', hwItems, ctx.hw);

      function badge(bg, fg, text) {
        return '<span style="padding:2px 10px;border-radius:10px;font-size:0.8em;'
             + 'font-weight:500;background:' + bg + ';color:' + fg + ';white-space:nowrap;">'
             + text + '</span>';
      }
      html += '<div style="' + cardS + 'gap:8px;flex-wrap:wrap;">'
            + badge('#dbeafe', '#1e40af', '&#128421;&#xFE0E; ' + p.totalGpus + ' GPU' + (p.totalGpus > 1 ? 's' : ''))
            + badge('#dcfce7', '#166534', '&#128266;&#xFE0E; ' + modeDef.audio)
            + badge('#f3e8ff', '#6b21a8', '&#128190;&#xFE0E; ' + HARDWARE[ctx.hw].label + ' ' + HARDWARE[ctx.hw].subtitle)
            + '</div>';

      var cmdPreS  = 'flex:1;padding:12px 16px;background:' + cmdBg + ';border-radius:6px;'
                   + 'font-family:Menlo,Monaco,"Courier New",monospace;'
                   + 'font-size:12px;line-height:1.6;color:' + cmdColor + ';'
                   + 'white-space:pre;overflow-x:auto;margin:0;border:1px solid ' + cardBorder + ';';
      var copyBtnS = 'padding:4px 10px;font-size:11px;width:64px;text-align:center;'
                   + 'background:' + btnBg + ';color:' + btnColor + ';'
                   + 'border:1px solid ' + btnBorder + ';border-radius:3px;cursor:pointer;flex-shrink:0;';
      html += '<div style="' + cardS + '">'
            + '<div style="' + titleS + '">Run this Command:</div>'
            + '<pre id="sgl-ming-cmd" style="' + cmdPreS + '">' + buildCommand(ctx) + '</pre>'
            + '<button id="sgl-ming-cmd-copy" style="' + copyBtnS + '">Copy</button>'
            + '</div>';

      var explanations = getExplanations(ctx);
      var flagRowS = 'display:flex;flex-direction:column;gap:4px;flex:1;';
      var flagLineS = 'display:flex;gap:12px;align-items:baseline;font-size:12px;';
      var flagNameS = 'font-family:Menlo,Monaco,"Courier New",monospace;'
                    + 'color:' + (dark ? '#a5b4fc' : '#4f46e5') + ';'
                    + 'white-space:nowrap;flex-shrink:0;min-width:220px;';
      var flagDescS = 'color:' + titleColor + ';opacity:0.8;';
      var lines = explanations.map(function(e) {
        return '<div style="' + flagLineS + '">'
             + '<span style="' + flagNameS + '">' + e.flag + '</span>'
             + '<span style="' + flagDescS + '">' + e.desc + '</span>'
             + '</div>';
      }).join('');
      html += '<div style="' + cardS + 'align-items:flex-start;">'
            + '<div style="' + titleS + '">Key flags:</div>'
            + '<div style="' + flagRowS + '">' + lines + '</div>'
            + '</div>';

      mountEl.innerHTML = '<div style="' + containerS + '">' + html + '</div>';

      mountEl.querySelectorAll('button[data-dim]').forEach(function(btn) {
        btn.addEventListener('click', function() {
          if (btn.getAttribute('data-disabled')) return;
          state[btn.getAttribute('data-dim')] = btn.getAttribute('data-val');
          update();
        });
      });

      var copyBtn = document.getElementById('sgl-ming-cmd-copy');
      if (copyBtn) {
        copyBtn.addEventListener('click', function() {
          var text = document.getElementById('sgl-ming-cmd').textContent;
          var flash = function() {
            copyBtn.textContent = 'Copied!';
            setTimeout(function() { copyBtn.textContent = 'Copy'; }, 1500);
          };
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(flash).catch(function() { fallbackCopy(text, flash); });
          } else {
            fallbackCopy(text, flash);
          }
        });
      }
    }

    new MutationObserver(function() {
      var d = detectDark();
      if (d !== dark) update();
    }).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class', 'data-theme', 'data-color-mode', 'style'],
    });

    update();
  }

  document.addEventListener('DOMContentLoaded', function () {
    var mount = document.getElementById('sgl-ming-server-gen-mount');
    if (mount) render(mount);
  });
}());
