// Voice orb: a matte, two-tone disc whose waterline and outline follow the voice.
// app.js calls frame() once per animation frame with the audio level (0..1) and a
// mood; without WebGL the CSS fallback in index.html is shown instead.

const VERTEX = `
attribute vec2 aPosition;
void main() { gl_Position = vec4(aPosition, 0.0, 1.0); }
`;

const FRAGMENT = `
precision highp float;
uniform vec2 uResolution;
uniform vec3 uDrift;      // integrated phases for the waterline
uniform float uSpin;      // integrated phase for the outline lobes
uniform float uBreath;    // integrated breathing phase
uniform float uEnergy;    // slow envelope: phrases
uniform float uPulse;     // fast envelope minus slow: syllable accents
uniform float uActivity;  // how much the orb is allowed to move, slew-limited
uniform float uThinking;  // 0..1, a light sweeping across while waiting
uniform vec3 uPaper;
uniform vec3 uTint;
uniform vec3 uDeep;

float hash(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.zyx + 31.32);
  return fract((p.x + p.y) * p.z);
}
float noise(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x), mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x), mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
float fbm(vec3 p) {
  float value = 0.0;
  float amplitude = 0.55;
  for (int i = 0; i < 3; i++) {
    value += amplitude * noise(p);
    p = p * 2.05 + vec3(3.1, 7.3, 1.7);
    amplitude *= 0.45;
  }
  return value;
}

void main() {
  vec2 uv = (gl_FragCoord.xy / uResolution - 0.5) * 2.0;
  float r = length(uv);
  float angle = atan(uv.y, uv.x);
  float pixel = 2.0 / uResolution.x;

  // A true circle at rest; activity lets a few broad lobes in, energy swells it.
  float lobes = 0.6 * sin(2.0 * angle + uSpin) + 0.4 * sin(3.0 * angle - uSpin + 2.0);
  float edge = 0.94 + uActivity * (-0.03 + 0.012 * sin(uBreath) + 0.022 * lobes) + 0.045 * uEnergy;
  float alpha = 1.0 - smoothstep(edge - 1.5 * pixel, edge + 0.5 * pixel, r);
  if (alpha <= 0.0) { gl_FragColor = vec4(0.0); return; }

  // One waterline across the disc: tint below, paper above. It rises with the
  // phrase envelope and its broad waves grow with activity; noise only feathers it.
  // The noise is sampled along closed loops, so the wrapped phases never jump.
  vec3 path = vec3(0.8 * cos(uDrift.x), 0.5 * sin(uDrift.y), 0.6 * sin(uDrift.z));
  float feather = fbm(vec3(uv * 2.2, 0.0) + path);
  float waves = (0.6 + 0.9 * uActivity) * (0.09 * sin(1.8 * uv.x + uDrift.x) + 0.045 * sin(3.1 * uv.x - uDrift.y));
  float line = 0.02 + waves + (feather - 0.5) * 0.30 + 0.30 * uEnergy + 0.05 * uPulse;
  float depth = line - uv.y;
  float pigment = smoothstep(-0.16, 0.20, depth);

  vec3 water = mix(uTint, uDeep, smoothstep(0.1, 1.3, depth));
  vec3 color = mix(uPaper, water, pigment);
  // A pale crest along the waterline, brighter on syllables.
  float crest = exp(-pow((depth - 0.02) / 0.10, 2.0));
  color = mix(color, uPaper, crest * (0.18 + 0.35 * uPulse));
  // Daylight from the upper left keeps large areas soft instead of flat.
  color = mix(color, vec3(1.0), 0.10 * (1.0 - smoothstep(-1.0, 0.2, uv.x - uv.y)));
  // Waiting: a soft light passes slowly across.
  vec2 lamp = vec2(0.6 * cos(uDrift.z * 2.0), 0.3 * sin(uDrift.z * 2.0) + 0.1);
  color = mix(color, uPaper, uThinking * 0.45 * exp(-dot(uv - lamp, uv - lamp) * 3.0));

  gl_FragColor = vec4(color, alpha);
}
`;

// [paper, tint, deep] per mood. One hue family, so a mood change reads as a shift, not a swap.
export const PALETTES = {
  idle: [[0.93, 0.95, 0.94], [0.58, 0.73, 0.71], [0.42, 0.57, 0.57]],
  listening: [[0.92, 0.96, 0.94], [0.45, 0.70, 0.67], [0.27, 0.50, 0.50]],
  user: [[0.93, 0.96, 0.98], [0.44, 0.66, 0.82], [0.24, 0.43, 0.64]],
  thinking: [[0.93, 0.95, 0.96], [0.52, 0.66, 0.72], [0.33, 0.47, 0.55]],
  speaking: [[0.92, 0.96, 0.95], [0.36, 0.64, 0.66], [0.18, 0.40, 0.48]],
  error: [[0.98, 0.94, 0.93], [0.86, 0.58, 0.56], [0.66, 0.36, 0.38]],
};

// How much each mood may move and how fast its material drifts.
const ACTIVITY = { idle: 0, listening: 0.45, user: 0.8, thinking: 0.7, speaking: 0.95, error: 0 };
const SPEED = { idle: 0.7, listening: 1.0, user: 1.2, thinking: 1.2, speaking: 1.35, error: 0.5 };

const TAU = Math.PI * 2;
function wrap(phase) {
  return phase - Math.floor(phase / TAU) * TAU;
}
// Exponential approach with separate rise and fall time constants (seconds).
function follow(value, target, dt, rise, fall) {
  return value + (target - value) * (1 - Math.exp(-dt / (target > value ? rise : fall)));
}

export class Orb {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl", { alpha: true, antialias: false, premultipliedAlpha: false, powerPreference: "low-power" });
    this.ready = false;
    this.last = 0;
    this.pulse = 0;
    this.energy = 0;
    this.activity = 0;
    this.thinking = 0;
    this.speed = SPEED.idle;
    this.drift = [0, 0, 0];
    this.spin = 0;
    this.breath = 0;
    this.colors = PALETTES.idle.map((color) => color.slice());
    this.reduced = matchMedia("(prefers-reduced-motion: reduce)");
    if (!this.gl) return;
    canvas.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      this.ready = false;
      canvas.parentElement.classList.remove("has-gl");
    });
    canvas.addEventListener("webglcontextrestored", () => this.init());
    this.init();
  }

  init() {
    const gl = this.gl;
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)) return shader;
      console.warn("orb shader:", gl.getShaderInfoLog(shader));
      return null;
    };
    const vertex = compile(gl.VERTEX_SHADER, VERTEX);
    const fragment = compile(gl.FRAGMENT_SHADER, FRAGMENT);
    if (!vertex || !fragment) return;
    const program = gl.createProgram();
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return;
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const position = gl.getAttribLocation(program, "aPosition");
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    const names = ["uResolution", "uDrift", "uSpin", "uBreath", "uEnergy", "uPulse", "uActivity", "uThinking", "uPaper", "uTint", "uDeep"];
    this.uniforms = Object.fromEntries(names.map((name) => [name, gl.getUniformLocation(program, name)]));
    this.ready = true;
    this.canvas.parentElement.classList.add("has-gl");
  }

  // level: 0..1 audio level; mood: a key of PALETTES.
  frame(now, level, mood) {
    const dt = this.last ? Math.min((now - this.last) / 1000, 0.1) : 0;
    this.last = now;
    const target = Math.max(0, Math.min(1, level));
    // Two envelopes from one level: the fast one accents syllables inside the
    // disc, the slow one is the phrase and is all the outline ever follows.
    this.pulse = follow(this.pulse, target, dt, 0.06, 0.18);
    this.energy = follow(this.energy, this.pulse, dt, 0.2, 0.42);
    this.thinking = follow(this.thinking, mood === "thinking" ? 1 : 0, dt, 0.6, 0.6);
    // Activity is also slew-limited, so a mood change never snaps the shape.
    const base = ACTIVITY[mood] ?? 0;
    const wanted = mood === "listening" || mood === "user" ? Math.min(0.9, base + this.energy) : base;
    const eased = follow(this.activity, wanted, dt, 0.7, 0.7);
    this.activity += Math.max(-1.1 * dt, Math.min(1.1 * dt, eased - this.activity));
    this.speed = follow(this.speed, SPEED[mood] ?? 1, dt, 0.6, 0.6);
    if (!this.reduced.matches) {
      // Phases are integrated rather than time * speed, so a speed change never jumps them.
      const step = dt * this.speed * 0.2 * (1 + 1.6 * this.activity);
      this.drift = [wrap(this.drift[0] + step * 0.72), wrap(this.drift[1] + step * 0.41), wrap(this.drift[2] + step * 0.9)];
      this.spin = wrap(this.spin + step * 0.8);
      this.breath = wrap(this.breath + dt * this.speed * 0.8);
    }
    const palette = PALETTES[mood] || PALETTES.idle;
    for (let i = 0; i < 3; i += 1) for (let c = 0; c < 3; c += 1) this.colors[i][c] = follow(this.colors[i][c], palette[i][c], dt, 0.5, 0.5);
    if (!this.ready || document.hidden) return;
    this.draw();
  }

  draw() {
    const gl = this.gl;
    const canvas = this.canvas;
    const size = Math.round(canvas.clientWidth * Math.min(window.devicePixelRatio || 1, 2));
    if (size && canvas.width !== size) {
      canvas.width = size;
      canvas.height = size;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const u = this.uniforms;
    const still = this.reduced.matches;
    gl.uniform2f(u.uResolution, canvas.width, canvas.height);
    gl.uniform3fv(u.uDrift, this.drift);
    gl.uniform1f(u.uSpin, this.spin);
    gl.uniform1f(u.uBreath, this.breath);
    gl.uniform1f(u.uEnergy, still ? 0 : this.energy);
    gl.uniform1f(u.uPulse, still ? 0 : Math.max(0, this.pulse - this.energy));
    gl.uniform1f(u.uActivity, still ? 0 : this.activity);
    gl.uniform1f(u.uThinking, this.thinking);
    gl.uniform3fv(u.uPaper, this.colors[0]);
    gl.uniform3fv(u.uTint, this.colors[1]);
    gl.uniform3fv(u.uDeep, this.colors[2]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}
