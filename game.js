'use strict';

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------
const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl', { antialias: true });
if (!gl) {
  document.getElementById('msg').textContent = 'Sorry, your browser does not support WebGL.';
  throw new Error('WebGL not supported');
}

const FOG_COLOR = [0.62, 0.72, 0.66];
const WORLD_HALF = 80;   // ground extends this far from the center
const PLAY_HALF = 60;    // the player is kept inside this square
const NUM_TREES = 420;
const NUM_COINS = 15;
const EYE_HEIGHT = 1.7;
const PLAYER_RADIUS = 0.4;

// ---------------------------------------------------------------------------
// Math helpers (column-major 4x4 matrices)
// ---------------------------------------------------------------------------
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function lookAt(e, t, up) {
  let zx = e[0] - t[0], zy = e[1] - t[1], zz = e[2] - t[2];
  let l = Math.hypot(zx, zy, zz); zx /= l; zy /= l; zz /= l;
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
  l = Math.hypot(xx, xy, xz); xx /= l; xy /= l; xz /= l;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  return new Float32Array([
    xx, yx, zx, 0,
    xy, yy, zy, 0,
    xz, yz, zz, 0,
    -(xx * e[0] + xy * e[1] + xz * e[2]),
    -(yx * e[0] + yy * e[1] + yz * e[2]),
    -(zx * e[0] + zy * e[1] + zz * e[2]),
    1,
  ]);
}

// Translation followed by a rotation around the Y axis.
function translateRotateY(x, y, z, angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, x, y, z, 1]);
}

const IDENTITY = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

// Seeded PRNG so the forest layout is the same every time.
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(1337);
const range = (a, b) => a + rand() * (b - a);

function heightAt(x, z) {
  return Math.sin(x * 0.05) * 1.2 + Math.cos(z * 0.04) * 1.5 +
         Math.sin((x + z) * 0.11) * 0.4 + Math.cos((x - z) * 0.07) * 0.5;
}

// ---------------------------------------------------------------------------
// Mesh building (flat-shaded, low-poly)
// ---------------------------------------------------------------------------
class MeshBuilder {
  constructor() { this.pos = []; this.nrm = []; this.col = []; }

  // `center` is a point inside the shape, used to make the normal face outward.
  tri(a, b, c, color, center) {
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz) || 1; nx /= l; ny /= l; nz /= l;
    const mx = (a[0] + b[0] + c[0]) / 3 - center[0];
    const my = (a[1] + b[1] + c[1]) / 3 - center[1];
    const mz = (a[2] + b[2] + c[2]) / 3 - center[2];
    if (nx * mx + ny * my + nz * mz < 0) { nx = -nx; ny = -ny; nz = -nz; }
    for (const p of [a, b, c]) {
      this.pos.push(p[0], p[1], p[2]);
      this.nrm.push(nx, ny, nz);
      this.col.push(color[0], color[1], color[2]);
    }
  }

  quad(a, b, c, d, color, center) {
    this.tri(a, b, c, color, center);
    this.tri(a, c, d, color, center);
  }

  upload() {
    const make = (arr) => {
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr), gl.STATIC_DRAW);
      return buf;
    };
    return { pos: make(this.pos), nrm: make(this.nrm), col: make(this.col), count: this.pos.length / 3 };
  }
}

function jitter(c, amt = 0.06) {
  const d = (rand() - 0.5) * amt * 2;
  return [c[0] + d, c[1] + d, c[2] + d * 0.5];
}

function cone(mb, cx, cy, cz, r, h, seg, color) {
  const apex = [cx, cy + h, cz], base = [cx, cy, cz], ctr = [cx, cy + h * 0.25, cz];
  const rot = rand() * Math.PI;
  for (let i = 0; i < seg; i++) {
    const a0 = rot + (i / seg) * Math.PI * 2, a1 = rot + ((i + 1) / seg) * Math.PI * 2;
    const p0 = [cx + Math.cos(a0) * r, cy, cz + Math.sin(a0) * r];
    const p1 = [cx + Math.cos(a1) * r, cy, cz + Math.sin(a1) * r];
    mb.tri(p0, p1, apex, jitter(color), ctr);
    mb.tri(p1, p0, base, color.map((v) => v * 0.6), ctr);
  }
}

function cylinder(mb, cx, cy, cz, r, h, seg, color) {
  const ctr = [cx, cy + h / 2, cz];
  for (let i = 0; i < seg; i++) {
    const a0 = (i / seg) * Math.PI * 2, a1 = ((i + 1) / seg) * Math.PI * 2;
    const x0 = cx + Math.cos(a0) * r, z0 = cz + Math.sin(a0) * r;
    const x1 = cx + Math.cos(a1) * r, z1 = cz + Math.sin(a1) * r;
    mb.quad([x0, cy, z0], [x1, cy, z1], [x1, cy + h, z1], [x0, cy + h, z0], jitter(color, 0.03), ctr);
    mb.tri([x0, cy + h, z0], [x1, cy + h, z1], [cx, cy + h, cz], color, ctr);
  }
}

// Lumpy low-poly sphere, used for broadleaf canopies, bushes and rocks.
function blob(mb, cx, cy, cz, rx, ry, rz, color, lump = 0.2) {
  const rings = 4, seg = 7, verts = [];
  for (let i = 0; i <= rings; i++) {
    const row = [], phi = (i / rings) * Math.PI;
    for (let j = 0; j < seg; j++) {
      const th = (j / seg) * Math.PI * 2;
      const k = (i === 0 || i === rings) ? 1 : 1 + (rand() - 0.5) * lump * 2;
      row.push([
        cx + Math.sin(phi) * Math.cos(th) * rx * k,
        cy + Math.cos(phi) * ry * k,
        cz + Math.sin(phi) * Math.sin(th) * rz * k,
      ]);
    }
    verts.push(row);
  }
  const ctr = [cx, cy, cz];
  for (let i = 0; i < rings; i++) {
    for (let j = 0; j < seg; j++) {
      const j1 = (j + 1) % seg;
      const a = verts[i][j], b = verts[i][j1], c = verts[i + 1][j1], d = verts[i + 1][j];
      // The top and bottom rows collapse to a pole, so skip the degenerate half.
      if (i !== 0) mb.tri(a, b, c, jitter(color), ctr);
      if (i !== rings - 1) mb.tri(a, c, d, jitter(color), ctr);
    }
  }
}

// ---------------------------------------------------------------------------
// World generation
// ---------------------------------------------------------------------------
const trees = []; // { x, z, r } collision circles

function buildWorld() {
  const mb = new MeshBuilder();

  // Ground
  const N = 100, cell = (WORLD_HALF * 2) / N, below = [0, -1000, 0];
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const x0 = -WORLD_HALF + i * cell, z0 = -WORLD_HALF + j * cell;
      const x1 = x0 + cell, z1 = z0 + cell;
      const p00 = [x0, heightAt(x0, z0), z0], p10 = [x1, heightAt(x1, z0), z0];
      const p11 = [x1, heightAt(x1, z1), z1], p01 = [x0, heightAt(x0, z1), z1];
      const patch = Math.sin(x0 * 0.13) * Math.cos(z0 * 0.17);
      const base = patch > 0.55 ? [0.42, 0.36, 0.22] : [0.25, 0.45, 0.2];
      mb.tri(p00, p10, p11, jitter(base, 0.04), below);
      mb.tri(p00, p11, p01, jitter(base, 0.04), below);
    }
  }

  // Trees
  let attempts = 0;
  while (trees.length < NUM_TREES && attempts++ < 20000) {
    const x = range(-WORLD_HALF + 2, WORLD_HALF - 2), z = range(-WORLD_HALF + 2, WORLD_HALF - 2);
    if (Math.hypot(x, z) < 5) continue; // keep the spawn point clear
    if (trees.some((t) => Math.hypot(t.x - x, t.z - z) < 3)) continue;
    const y = heightAt(x, z) - 0.2;
    const s = range(0.8, 1.4);
    if (rand() < 0.65) {
      // Pine: trunk + stacked cones
      cylinder(mb, x, y, z, 0.25 * s, 2 * s, 6, [0.35, 0.24, 0.14]);
      const green = [range(0.1, 0.16), range(0.3, 0.42), range(0.15, 0.22)];
      cone(mb, x, y + 1.4 * s, z, 1.9 * s, 2.6 * s, 8, green);
      cone(mb, x, y + 2.8 * s, z, 1.5 * s, 2.3 * s, 8, green.map((v) => v * 1.08));
      cone(mb, x, y + 4.0 * s, z, 1.0 * s, 2.0 * s, 8, green.map((v) => v * 1.16));
      trees.push({ x, z, r: 0.3 * s });
    } else {
      // Broadleaf: trunk + lumpy canopy
      cylinder(mb, x, y, z, 0.3 * s, 3.2 * s, 6, [0.38, 0.27, 0.16]);
      const green = [range(0.2, 0.32), range(0.42, 0.55), range(0.12, 0.2)];
      blob(mb, x, y + 4.3 * s, z, 2.1 * s, 1.8 * s, 2.1 * s, green, 0.25);
      trees.push({ x, z, r: 0.35 * s });
    }
  }

  // Bushes and rocks for decoration (no collision)
  for (let i = 0; i < 260; i++) {
    const x = range(-WORLD_HALF, WORLD_HALF), z = range(-WORLD_HALF, WORLD_HALF);
    const y = heightAt(x, z);
    if (rand() < 0.7) {
      const r = range(0.5, 0.9);
      blob(mb, x, y + r * 0.4, z, r, r * 0.7, r, [0.2, range(0.38, 0.5), 0.15], 0.3);
    } else {
      const r = range(0.3, 0.7);
      const g = range(0.4, 0.55);
      blob(mb, x, y + r * 0.15, z, r, r * 0.55, r * 0.8, [g, g, g * 0.95], 0.3);
    }
  }

  return mb.upload();
}

function buildCoinMesh() {
  const mb = new MeshBuilder();
  const seg = 20, r = 0.45, t = 0.06, ctr = [0, 0, 0];
  const face = [1.0, 0.8, 0.15], rim = [0.85, 0.6, 0.1], inner = [0.95, 0.7, 0.1];
  for (let i = 0; i < seg; i++) {
    const a0 = (i / seg) * Math.PI * 2, a1 = ((i + 1) / seg) * Math.PI * 2;
    const x0 = Math.cos(a0) * r, y0 = Math.sin(a0) * r;
    const x1 = Math.cos(a1) * r, y1 = Math.sin(a1) * r;
    mb.quad([x0, y0, -t], [x1, y1, -t], [x1, y1, t], [x0, y0, t], rim, ctr);
    for (const z of [t, -t]) {
      // Outer ring and a slightly inset center for a stamped look
      const k = 0.7;
      mb.quad([x0, y0, z], [x1, y1, z], [x1 * k, y1 * k, z], [x0 * k, y0 * k, z], face, ctr);
      mb.tri([x0 * k, y0 * k, z], [x1 * k, y1 * k, z], [0, 0, z], inner, ctr);
    }
  }
  return mb.upload();
}

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------
const VS = `
attribute vec3 aPos;
attribute vec3 aNormal;
attribute vec3 aColor;
uniform mat4 uProj, uView, uModel;
varying vec3 vColor;
varying vec3 vNormal;
varying float vDist;
void main() {
  vec4 world = uModel * vec4(aPos, 1.0);
  vec4 view = uView * world;
  vColor = aColor;
  vNormal = mat3(uModel) * aNormal;
  vDist = length(view.xyz);
  gl_Position = uProj * view;
}`;

const FS = `
precision mediump float;
uniform vec3 uLightDir;
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform vec3 uEmissive;
varying vec3 vColor;
varying vec3 vNormal;
varying float vDist;
void main() {
  vec3 n = normalize(vNormal);
  float diffuse = max(dot(n, uLightDir), 0.0);
  vec3 ambient = mix(vec3(0.25, 0.22, 0.18), vec3(0.45, 0.5, 0.55), n.y * 0.5 + 0.5);
  vec3 col = vColor * (ambient + vec3(1.0, 0.95, 0.8) * diffuse * 0.8) + uEmissive;
  float f = vDist * uFogDensity;
  float fog = clamp(1.0 - exp(-f * f), 0.0, 1.0);
  gl_FragColor = vec4(mix(col, uFogColor, fog), 1.0);
}`;

function compile(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
  return sh;
}

const prog = gl.createProgram();
gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
gl.useProgram(prog);

const loc = {
  aPos: gl.getAttribLocation(prog, 'aPos'),
  aNormal: gl.getAttribLocation(prog, 'aNormal'),
  aColor: gl.getAttribLocation(prog, 'aColor'),
};
for (const name of ['uProj', 'uView', 'uModel', 'uLightDir', 'uFogColor', 'uFogDensity', 'uEmissive']) {
  loc[name] = gl.getUniformLocation(prog, name);
}

const light = [0.4, 0.8, 0.3], ll = Math.hypot(...light);
gl.uniform3f(loc.uLightDir, light[0] / ll, light[1] / ll, light[2] / ll);
gl.uniform3fv(loc.uFogColor, FOG_COLOR);
gl.uniform1f(loc.uFogDensity, 0.028);
gl.enable(gl.DEPTH_TEST);
gl.clearColor(FOG_COLOR[0], FOG_COLOR[1], FOG_COLOR[2], 1);

function drawMesh(mesh, model, emissive) {
  const bind = (buf, attr) => {
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(attr);
    gl.vertexAttribPointer(attr, 3, gl.FLOAT, false, 0, 0);
  };
  bind(mesh.pos, loc.aPos);
  bind(mesh.nrm, loc.aNormal);
  bind(mesh.col, loc.aColor);
  gl.uniformMatrix4fv(loc.uModel, false, model);
  gl.uniform3fv(loc.uEmissive, emissive);
  gl.drawArrays(gl.TRIANGLES, 0, mesh.count);
}

const worldMesh = buildWorld();
const coinMesh = buildCoinMesh();

// ---------------------------------------------------------------------------
// Game state
// ---------------------------------------------------------------------------
const player = { x: 0, y: 0, z: 0, vy: 0, yaw: 0, pitch: 0, onGround: true };
let coins = [];
let collected = 0;
let elapsed = 0;
let playing = false;
let finished = false;

const $ = (id) => document.getElementById(id);
const overlay = $('overlay');

function placeCoins() {
  coins = [];
  let attempts = 0;
  while (coins.length < NUM_COINS && attempts++ < 10000) {
    const x = (Math.random() * 2 - 1) * (PLAY_HALF - 3);
    const z = (Math.random() * 2 - 1) * (PLAY_HALF - 3);
    if (Math.hypot(x, z) < 12) continue;
    if (trees.some((t) => Math.hypot(t.x - x, t.z - z) < t.r + 1.2)) continue;
    if (coins.some((c) => Math.hypot(c.x - x, c.z - z) < 10)) continue;
    coins.push({ x, z, y: heightAt(x, z) + 1.0, phase: Math.random() * Math.PI * 2, taken: false });
  }
}

function resetGame() {
  placeCoins();
  collected = 0;
  elapsed = 0;
  finished = false;
  Object.assign(player, { x: 0, z: 0, vy: 0, yaw: 0, pitch: 0, onGround: true });
  player.y = heightAt(0, 0);
  $('total').textContent = coins.length;
  updateHud();
}

function updateHud() {
  $('count').textContent = collected;
  $('time').textContent = elapsed.toFixed(1);
  let nearest = Infinity;
  for (const c of coins) if (!c.taken) nearest = Math.min(nearest, Math.hypot(c.x - player.x, c.z - player.z));
  let sense = '';
  if (nearest < 6) sense = '\u{1F525} Very hot!';
  else if (nearest < 15) sense = '☀️ Warm';
  else if (nearest < 30) sense = '\u{1F343} Cool';
  else if (nearest < Infinity) sense = '❄️ Cold';
  $('sense').textContent = sense;
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------
let audio = null;
function chime(final) {
  if (!audio) return;
  const notes = final ? [660, 880, 1320] : [988, 1319];
  notes.forEach((freq, i) => {
    const t = audio.currentTime + i * 0.08;
    const osc = audio.createOscillator(), g = audio.createGain();
    osc.type = 'triangle';
    osc.frequency.value = freq;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.25, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.35);
    osc.connect(g).connect(audio.destination);
    osc.start(t);
    osc.stop(t + 0.4);
  });
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------
const keys = new Set();
addEventListener('keydown', (e) => {
  keys.add(e.code);
  if (e.code === 'Space' || e.code.startsWith('Arrow')) e.preventDefault();
});
addEventListener('keyup', (e) => keys.delete(e.code));
addEventListener('blur', () => keys.clear());

const SENSITIVITY = 0.0025;
function look(dx, dy) {
  // Some browsers report a huge jump on the first event after pointer lock.
  if (Math.abs(dx) > 250 || Math.abs(dy) > 250) return;
  player.yaw -= dx * SENSITIVITY;
  player.pitch = Math.max(-1.4, Math.min(1.4, player.pitch - dy * SENSITIVITY));
}

// Pointer lock when available; click-and-drag look as a fallback.
let dragging = false;
canvas.addEventListener('mousedown', () => {
  if (!playing) return;
  dragging = true;
  if (document.pointerLockElement !== canvas && canvas.requestPointerLock) {
    try { canvas.requestPointerLock(); } catch (_) { /* ignore */ }
  }
});
addEventListener('mouseup', () => { dragging = false; });
addEventListener('mousemove', (e) => {
  if (!playing) return;
  if (document.pointerLockElement === canvas || dragging) look(e.movementX, e.movementY);
});

let hadLock = false;
document.addEventListener('pointerlockchange', () => {
  if (document.pointerLockElement === canvas) hadLock = true;
  else if (hadLock && playing) pause();
});

function start() {
  if (!audio) {
    try { audio = new (window.AudioContext || window.webkitAudioContext)(); } catch (_) { audio = null; }
  }
  if (audio && audio.state === 'suspended') audio.resume();
  if (finished || coins.length === 0) resetGame();
  playing = true;
  overlay.classList.add('hidden');
  if (canvas.requestPointerLock) {
    try { canvas.requestPointerLock(); } catch (_) { /* drag-to-look still works */ }
  }
}

function pause() {
  playing = false;
  hadLock = false;
  $('title').textContent = 'Paused';
  $('msg').textContent = `${collected} of ${coins.length} coins found so far.`;
  $('start').textContent = 'Resume';
  overlay.classList.remove('hidden');
}

function win() {
  playing = false;
  finished = true;
  hadLock = false;
  if (document.exitPointerLock) document.exitPointerLock();
  $('title').textContent = 'You found them all!';
  $('msg').textContent = `All ${coins.length} coins collected in ${elapsed.toFixed(1)} seconds.`;
  $('start').textContent = 'Play again';
  overlay.classList.remove('hidden');
}

$('start').addEventListener('click', start);
addEventListener('keydown', (e) => {
  if (e.code === 'Escape' && playing) pause();
  else if (e.code === 'Enter' && !playing) start();
});

// ---------------------------------------------------------------------------
// Update & render
// ---------------------------------------------------------------------------
function update(dt) {
  const turn = (keys.has('ArrowLeft') ? 1 : 0) - (keys.has('ArrowRight') ? 1 : 0);
  player.yaw += turn * 2.2 * dt;

  const fwd = (keys.has('KeyW') || keys.has('ArrowUp') ? 1 : 0) - (keys.has('KeyS') || keys.has('ArrowDown') ? 1 : 0);
  const strafe = (keys.has('KeyD') ? 1 : 0) - (keys.has('KeyA') ? 1 : 0);
  const speed = keys.has('ShiftLeft') || keys.has('ShiftRight') ? 11 : 6;
  const sin = Math.sin(player.yaw), cos = Math.cos(player.yaw);
  let mx = -sin * fwd + cos * strafe, mz = -cos * fwd - sin * strafe;
  const ml = Math.hypot(mx, mz);
  if (ml > 0) { mx /= ml; mz /= ml; }
  player.x += mx * speed * dt;
  player.z += mz * speed * dt;

  // Push the player out of tree trunks
  for (const t of trees) {
    const dx = player.x - t.x, dz = player.z - t.z, min = t.r + PLAYER_RADIUS;
    const d2 = dx * dx + dz * dz;
    if (d2 < min * min && d2 > 1e-8) {
      const d = Math.sqrt(d2);
      player.x = t.x + (dx / d) * min;
      player.z = t.z + (dz / d) * min;
    }
  }
  player.x = Math.max(-PLAY_HALF, Math.min(PLAY_HALF, player.x));
  player.z = Math.max(-PLAY_HALF, Math.min(PLAY_HALF, player.z));

  // Jumping & gravity
  const ground = heightAt(player.x, player.z);
  if (keys.has('Space') && player.onGround) { player.vy = 6; player.onGround = false; }
  player.vy -= 18 * dt;
  player.y += player.vy * dt;
  if (player.y <= ground) { player.y = ground; player.vy = 0; player.onGround = true; }

  // Coin pickup
  const eyeY = player.y + EYE_HEIGHT;
  for (const c of coins) {
    if (c.taken) continue;
    if (Math.hypot(c.x - player.x, c.z - player.z) < 1.3 && Math.abs(c.y - (eyeY - 0.8)) < 1.8) {
      c.taken = true;
      collected++;
      chime(collected === coins.length);
    }
  }

  elapsed += dt;
  updateHud();
  if (collected === coins.length) win();
}

function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const w = Math.floor(innerWidth * dpr), h = Math.floor(innerHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.viewport(0, 0, w, h);
}

function render(time) {
  resize();
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const eye = [player.x, player.y + EYE_HEIGHT, player.z];
  const cp = Math.cos(player.pitch);
  const target = [
    eye[0] - Math.sin(player.yaw) * cp,
    eye[1] + Math.sin(player.pitch),
    eye[2] - Math.cos(player.yaw) * cp,
  ];
  gl.uniformMatrix4fv(loc.uProj, false, perspective(Math.PI / 3, canvas.width / canvas.height, 0.1, 200));
  gl.uniformMatrix4fv(loc.uView, false, lookAt(eye, target, [0, 1, 0]));

  drawMesh(worldMesh, IDENTITY, [0, 0, 0]);

  const glow = 0.25 + 0.15 * Math.sin(time * 4);
  for (const c of coins) {
    if (c.taken) continue;
    const bob = Math.sin(time * 2 + c.phase) * 0.15;
    drawMesh(coinMesh, translateRotateY(c.x, c.y + bob, c.z, time * 2.5 + c.phase), [glow, glow * 0.75, 0]);
  }
}

let last = performance.now();
function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;
  if (playing) update(dt);
  render(now / 1000);
  requestAnimationFrame(frame);
}

resetGame();
requestAnimationFrame(frame);
