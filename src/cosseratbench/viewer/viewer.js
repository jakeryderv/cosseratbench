import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const SVG = "http://www.w3.org/2000/svg";

// Names and descriptions come from result files, so they only ever enter the page
// as text nodes (append) or attributes, never as markup.
function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  node.append(...children);
  return node;
}

function svg(tag, attributes = {}, ...children) {
  const node = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  node.append(...children);
  return node;
}

const cssColor = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// ---------------------------------------------------------------- formatting

function formatNumber(value) {
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e-3 && magnitude < 1e4) return String(Number(value.toPrecision(3)));
  return value.toExponential(2).replace("e+", "e");
}

const PREFIXES = [[1e9, "G"], [1e6, "M"], [1e3, "k"], [1, ""], [1e-3, "m"], [1e-6, "µ"]];

function si(value, unit) {
  if (value === 0) return `0 ${unit}`;
  const [scale, prefix] = PREFIXES.find(([s]) => Math.abs(value) >= s) ?? PREFIXES.at(-1);
  return `${Number((value / scale).toPrecision(4))} ${prefix}${unit}`;
}

function direction(vector) {
  const length = Math.hypot(...vector);
  return `(${vector.map((v) => Number((v / length).toFixed(2))).join(", ")})`;
}

function niceStep(span, targetTicks) {
  const raw = span / targetTicks;
  const power = 10 ** Math.floor(Math.log10(raw));
  return power * [1, 2, 5, 10].find((m) => m * power >= raw);
}

// ---------------------------------------------------------------- rod geometry

const RADIAL_SEGMENTS = 12;

/** A tube along a polyline whose vertices are rewritten in place every frame. */
class RodMesh {
  constructor(nNodes, radius, material) {
    this.nNodes = nNodes;
    this.radius = radius;
    this.nodes = new Float32Array(nNodes * 3); // the polyline currently shown

    const vertices = nNodes * RADIAL_SEGMENTS;
    this.positions = new THREE.BufferAttribute(new Float32Array(vertices * 3), 3);
    this.normals = new THREE.BufferAttribute(new Float32Array(vertices * 3), 3);
    this.positions.setUsage(THREE.DynamicDrawUsage);
    this.normals.setUsage(THREE.DynamicDrawUsage);

    const index = [];
    for (let i = 0; i < nNodes - 1; i++) {
      for (let j = 0; j < RADIAL_SEGMENTS; j++) {
        const a = i * RADIAL_SEGMENTS + j;
        const b = i * RADIAL_SEGMENTS + ((j + 1) % RADIAL_SEGMENTS);
        index.push(a, b, a + RADIAL_SEGMENTS, b, b + RADIAL_SEGMENTS, a + RADIAL_SEGMENTS);
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setIndex(index);
    geometry.setAttribute("position", this.positions);
    geometry.setAttribute("normal", this.normals);

    const tube = new THREE.Mesh(geometry, material);
    tube.frustumCulled = false; // its bounds change every frame
    const cap = new THREE.SphereGeometry(radius, RADIAL_SEGMENTS, 8);
    this.caps = [new THREE.Mesh(cap, material), new THREE.Mesh(cap, material)];
    this.object = new THREE.Group().add(tube, ...this.caps);
  }

  /** Show the blend of two frames, each a Float32Array of nNodes * 3. */
  show(from, to, blend) {
    const p = this.nodes;
    for (let k = 0; k < p.length; k++) p[k] = from[k] + (to[k] - from[k]) * blend;

    const n = this.nNodes;
    const position = this.positions.array;
    const normal = this.normals.array;
    let ux = 0, uy = 0, uz = 0; // ring reference direction, carried along the rod
    for (let i = 0; i < n; i++) {
      const before = 3 * Math.max(i - 1, 0);
      const after = 3 * Math.min(i + 1, n - 1);
      let tx = p[after] - p[before], ty = p[after + 1] - p[before + 1], tz = p[after + 2] - p[before + 2];
      const tl = Math.hypot(tx, ty, tz) || 1;
      tx /= tl; ty /= tl; tz /= tl;

      if (i === 0) {
        // Start from whichever world axis is least aligned with the rod.
        const ax = Math.abs(tx), ay = Math.abs(ty), az = Math.abs(tz);
        ux = ax <= ay && ax <= az ? 1 : 0;
        uy = ux === 0 && ay <= az ? 1 : 0;
        uz = ux === 0 && uy === 0 ? 1 : 0;
      }
      // Carry the reference forward by removing its component along the new tangent.
      const along = ux * tx + uy * ty + uz * tz;
      ux -= along * tx; uy -= along * ty; uz -= along * tz;
      const ul = Math.hypot(ux, uy, uz) || 1;
      ux /= ul; uy /= ul; uz /= ul;
      const vx = ty * uz - tz * uy, vy = tz * ux - tx * uz, vz = tx * uy - ty * ux;

      for (let j = 0; j < RADIAL_SEGMENTS; j++) {
        const angle = (2 * Math.PI * j) / RADIAL_SEGMENTS;
        const c = Math.cos(angle), s = Math.sin(angle);
        const nx = c * ux + s * vx, ny = c * uy + s * vy, nz = c * uz + s * vz;
        const o = 3 * (i * RADIAL_SEGMENTS + j);
        normal[o] = nx; normal[o + 1] = ny; normal[o + 2] = nz;
        position[o] = p[3 * i] + this.radius * nx;
        position[o + 1] = p[3 * i + 1] + this.radius * ny;
        position[o + 2] = p[3 * i + 2] + this.radius * nz;
      }
    }
    this.positions.needsUpdate = true;
    this.normals.needsUpdate = true;
    this.caps[0].position.set(p[0], p[1], p[2]);
    this.caps[1].position.set(p[3 * n - 3], p[3 * n - 2], p[3 * n - 1]);
  }
}

// ---------------------------------------------------------------- state

const manifest = await fetch("data/manifest.json").then((response) => response.json());
const trajectories = new Map(); // file -> Promise<Float32Array>

const state = {
  experiment: null,
  variant: null, // the swept variant of the experiment on show
  axis: null, // what the variant controls currently vary
  entries: [], // one per run that has a trajectory: { run, colorVar, rods, object, shown, speeds }
  reference: null, // { object, shown }
  mode: "overlay",
  cameraMoved: false, // by the user, since the experiment was framed
  time: 0,
  duration: 1,
  playing: false,
  speed: 1,
  dirty: true,
};

const seriesVar = (solver) => {
  const slot = manifest.solvers.indexOf(solver) + 1;
  return slot >= 1 && slot <= 8 ? `--series-${slot}` : "--muted"; // past eight, never a recycled hue
};

// ---------------------------------------------------------------- 3D scene

const canvas = $("canvas");
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
} catch {
  notify("This browser cannot create a WebGL context, so the 3D view is unavailable.");
}
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
camera.up.set(0, 0, 1); // simulations are z-up
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.addEventListener("change", invalidate);
controls.addEventListener("start", () => (state.cameraMoved = true));

scene.add(new THREE.HemisphereLight(0xffffff, 0x777777, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(0.4, -1, 1.2);
scene.add(sun);

const content = new THREE.Group(); // everything that belongs to the current experiment
scene.add(content);

function invalidate() {
  state.dirty = true;
}

function notify(message) {
  $("notice").textContent = message ?? "";
  $("notice").hidden = !message;
}

function bounds() {
  const box = new THREE.Box3();
  const point = new THREE.Vector3();
  const include = (array) => {
    for (let k = 0; k < array.length; k += 3) box.expandByPoint(point.set(array[k], array[k + 1], array[k + 2]));
  };
  for (const entry of state.entries) for (const rod of entry.rods) include(rod.frames);
  if (state.variant.reference) include(state.variant.reference.flat());
  return box;
}

function panes() {
  const shown = state.entries.filter((entry) => entry.shown);
  return state.mode === "split" && shown.length > 1 ? shown : null;
}

/** Point the camera at everything the experiment ever does, seen from the front. */
function frameCamera() {
  const box = bounds();
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.y, size.z);

  const aspect = canvas.clientWidth / (panes()?.length ?? 1) / canvas.clientHeight;
  const halfAngle = THREE.MathUtils.degToRad(camera.fov / 2);
  const distance = ((extent / 2) * 1.35) / Math.tan(halfAngle) / Math.min(aspect, 1);
  camera.position.copy(center).addScaledVector(new THREE.Vector3(0.12, -1, 0.1).normalize(), distance);
  camera.near = distance / 100;
  camera.far = distance * 100;
  controls.target.copy(center);
  controls.update();
  state.cameraMoved = false;
  buildGrid();
}

/** A floor grid just below the motion, for scale; it is not part of the physics. */
function buildGrid() {
  const box = bounds();
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const extent = Math.max(size.x, size.y, size.z);

  const previous = content.getObjectByName("grid");
  previous?.geometry.dispose();
  if (previous) content.remove(previous);
  const cell = niceStep(extent, 8);
  const cells = 2 * Math.ceil(extent / cell);
  const grid = new THREE.GridHelper(cell * cells, cells, cssColor("--axis"), cssColor("--grid"));
  grid.name = "grid";
  grid.rotation.x = Math.PI / 2;
  grid.position.set(Math.round(center.x / cell) * cell, Math.round(center.y / cell) * cell, box.min.z - 0.05 * extent);
  content.add(grid);
  $("scale").textContent = `grid ${si(cell, "m")}`;
  invalidate();
}

function render() {
  if (!renderer) return;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (canvas.width !== Math.floor(width * devicePixelRatio) || canvas.height !== Math.floor(height * devicePixelRatio)) {
    renderer.setPixelRatio(devicePixelRatio);
    renderer.setSize(width, height, false);
  }
  if (state.reference) state.reference.object.visible = state.reference.shown;

  const split = panes();
  renderer.setScissorTest(true);
  const views = split ? split.map((entry) => [entry]) : [state.entries.filter((entry) => entry.shown)];
  views.forEach((visible, i) => {
    for (const entry of state.entries) entry.object.visible = visible.includes(entry);
    const paneWidth = width / views.length;
    renderer.setViewport(i * paneWidth, 0, paneWidth, height);
    renderer.setScissor(i * paneWidth, 0, paneWidth, height);
    camera.aspect = paneWidth / height;
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
  });
}

function recolor() {
  for (const entry of state.entries) entry.material.color.set(cssColor(entry.colorVar));
  state.reference?.object.material.color.set(cssColor("--reference"));
  buildGrid();
  invalidate();
}

// ---------------------------------------------------------------- loading an experiment

function loadTrajectory(file) {
  if (!trajectories.has(file)) {
    trajectories.set(file, fetch(file).then((r) => r.arrayBuffer()).then((b) => new Float32Array(b)));
  }
  return trajectories.get(file);
}

/** Fastest node's speed at each frame after the first, in rod lengths per second. */
function nodeSpeeds(rods, times, lengths) {
  const speeds = new Float64Array(times.length).fill(NaN);
  for (let f = 1; f < times.length; f++) {
    let fastest = 0;
    rods.forEach((rod, r) => {
      const stride = rod.nNodes * 3;
      for (let k = f * stride; k < (f + 1) * stride; k += 3) {
        const d = Math.hypot(
          rod.frames[k] - rod.frames[k - stride],
          rod.frames[k + 1] - rod.frames[k + 1 - stride],
          rod.frames[k + 2] - rod.frames[k + 2 - stride],
        );
        fastest = Math.max(fastest, d / lengths[r]);
      }
    });
    speeds[f] = fastest / (times[f] - times[f - 1]);
  }
  return speeds;
}

function select(experiment, key) {
  if (state.experiment !== experiment) {
    state.experiment = experiment;
    for (const button of $("experiments").children) {
      button.setAttribute("aria-current", String(button.dataset.name === experiment.name));
    }
    $("title").textContent = experiment.name;
    $("description").textContent = experiment.description;
    $("notes").textContent = experiment.notes;
    $("notes").hidden = !experiment.notes;
  }
  const variant = experiment.variants.find((v) => v.key === key) ?? experiment.variants[0];
  buildVariantControls(variant);
  if (state.variant !== variant) showVariant(variant);
}

/** Every name the experiment was swept over, with its values and the variant for each. */
function axes(experiment) {
  const ordinary = experiment.variants.find((v) => !Object.keys(v.varied).length) ?? experiment.variants[0];
  const found = new Map();
  for (const variant of experiment.variants) {
    for (const [name, value] of Object.entries(variant.varied)) {
      if (!found.has(name)) found.set(name, new Map([[experiment.defaults[name], ordinary]]));
      found.get(name).set(value, variant);
    }
  }
  for (const [name, values] of found) found.set(name, new Map([...values].sort((a, b) => a[0] - b[0])));
  return found;
}

function axisLabel(experiment, name) {
  const unit = experiment.parameters.find((p) => p.name === name)?.unit;
  return name.replaceAll("_", " ") + (unit ? ` (${unit})` : "");
}

function buildVariantControls(variant) {
  const experiment = state.experiment;
  const found = axes(experiment);
  $("variants").hidden = found.size === 0;
  if (!found.size) return;
  const [current] = Object.keys(variant.varied);
  const axis = current ?? state.axis ?? [...found.keys()][0];
  state.axis = found.has(axis) ? axis : [...found.keys()][0];

  const picker = el("select", { id: "vary", "aria-label": "What to vary" },
    ...[...found.keys()].map((name) => {
      const option = el("option", { value: name }, axisLabel(experiment, name));
      option.selected = name === state.axis;
      return option;
    }));
  picker.addEventListener("change", () => {
    state.axis = picker.value;
    location.hash = experiment.name; // back to the ordinary case, which every axis contains
  });

  const values = found.get(state.axis);
  const buttons = [...values].map(([value, target]) => {
    const isDefault = target === values.get(experiment.defaults[state.axis]);
    const button = el("button", { type: "button", "aria-pressed": String(target === variant) },
      formatNumber(value) + (isDefault ? " (usual)" : ""));
    button.addEventListener("click", () => (location.hash = `${experiment.name}/${target.key}`));
    return button;
  });
  $("variants").replaceChildren(
    el("label", { class: "vary" }, "Vary ", picker),
    el("div", { class: "segmented", role: "group", "aria-label": "Value" }, ...buttons),
  );
}

async function showVariant(variant) {
  state.variant = variant;
  state.playing = false;

  // Replace the previous variant's objects.
  content.traverse((object) => {
    object.geometry?.dispose();
    object.material?.dispose();
  });
  content.clear();
  state.entries = [];

  const radii = variant.scenario.rods.map((rod) => rod.radius);
  const runs = variant.runs.filter((run) => run.trajectory);
  const loaded = await Promise.all(runs.map((run) => loadTrajectory(run.trajectory.file)));
  if (state.variant !== variant) return; // another variant was picked while this one loaded

  runs.forEach((run, order) => {
    const { times, n_nodes: nNodes } = run.trajectory;
    const colorVar = seriesVar(run.solver);
    // Coincident rods would flicker where they overlap; the offset gives a stable winner.
    const material = new THREE.MeshStandardMaterial({
      color: cssColor(colorVar), roughness: 0.55, metalness: 0,
      polygonOffset: true, polygonOffsetFactor: order, polygonOffsetUnits: order,
    });
    let offset = 0;
    const rods = nNodes.map((n, r) => {
      const frames = loaded[order].subarray(offset, offset + times.length * n * 3);
      offset += frames.length;
      return { nNodes: n, frames, mesh: new RodMesh(n, radii[r], material) };
    });
    const object = new THREE.Group().add(...rods.map((rod) => rod.mesh.object));
    content.add(object);
    state.entries.push({
      run, colorVar, material, rods, object, times, shown: true,
      speeds: nodeSpeeds(rods, times, variant.rod_lengths),
    });
  });

  state.reference = null;
  if (variant.reference) {
    const geometry = new THREE.BufferGeometry().setFromPoints(variant.reference.map((p) => new THREE.Vector3(...p)));
    // Drawn over the rods: the answer is a centreline, and would otherwise sit inside their tubes.
    const material = new THREE.LineBasicMaterial({ color: cssColor("--reference"), depthTest: false });
    const object = new THREE.Line(geometry, material);
    object.renderOrder = 1;
    content.add(object);
    state.reference = { object, shown: true };
  }

  state.duration = Math.max(variant.scenario.duration, ...state.entries.map((entry) => entry.times.at(-1)));
  $("scrub").max = state.duration;
  notify(state.entries.length ? null : "No solver produced a trajectory here; see why under Metrics.");

  buildLegend();
  buildMetrics();
  buildScenario();
  layoutChanged();
  frameCamera();

  // Play once and come to rest on the final state, which is what the metrics describe.
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  seek(still ? state.duration : 0);
  setPlaying(!still && state.entries.length > 0);
}

// ---------------------------------------------------------------- playback

function seek(time) {
  state.time = Math.min(Math.max(time, 0), state.duration);
  for (const entry of state.entries) {
    const last = entry.times.length - 1;
    const position = Math.min((state.time / entry.times[last]) * last, last); // frames are evenly spaced
    const frame = Math.min(Math.floor(position), last - 1);
    for (const rod of entry.rods) {
      const stride = rod.nNodes * 3;
      rod.mesh.show(
        rod.frames.subarray(frame * stride, (frame + 1) * stride),
        rod.frames.subarray((frame + 1) * stride, (frame + 2) * stride),
        position - frame,
      );
    }
  }
  $("scrub").value = state.time;
  $("clock").textContent = `${state.time.toFixed(2)} / ${state.duration.toFixed(2)} s`;
  chart.movePlayhead();
  invalidate();
}

function setPlaying(playing) {
  if (playing && state.time >= state.duration) seek(0);
  state.playing = playing;
  $("play").textContent = playing ? "❚❚" : "▶";
  $("play").setAttribute("aria-label", playing ? "Pause" : "Play");
}

let previous = performance.now();
function tick(now) {
  const elapsed = (now - previous) / 1000;
  previous = now;
  if (state.playing) {
    seek(state.time + elapsed * state.speed);
    if (state.time >= state.duration) setPlaying(false);
  }
  controls.update();
  if (state.dirty) {
    state.dirty = false;
    render();
  }
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- legend and panes

function swatch(colorVar, kind = "") {
  const node = el("span", { class: `swatch ${kind}` });
  node.style.background = `var(${colorVar})`;
  return node;
}

function buildLegend() {
  const toggle = (target, label, mark) => {
    const button = el("button", { type: "button", "aria-pressed": "true" }, mark, label);
    button.addEventListener("click", () => {
      target.shown = !target.shown;
      button.setAttribute("aria-pressed", String(target.shown));
      layoutChanged();
    });
    return button;
  };
  $("legend").replaceChildren(
    ...state.entries.map((entry) => toggle(entry, entry.run.solver, swatch(entry.colorVar))),
    ...(state.reference ? [toggle(state.reference, "analytical reference", swatch("--reference", "line"))] : []),
  );
}

/** Anything that changes which panes exist: mode, visibility, or the experiment. */
function layoutChanged() {
  const split = panes();
  $("panes").style.gridTemplateColumns = `repeat(${split?.length ?? 1}, 1fr)`;
  $("panes").replaceChildren(...(split ?? []).map((entry) => el("div", {}, swatch(entry.colorVar), entry.run.solver)));
  chart.draw();
  invalidate();
}

// ---------------------------------------------------------------- metrics table

function buildMetrics() {
  const runs = state.variant.runs;
  const names = [...new Set(runs.flatMap((run) => Object.keys(run.metrics)))];
  const observed = [...new Set(runs.flatMap((run) => Object.keys(run.observations ?? {})))];
  const rows = [
    ...names.map((name) => ({ label: name.replaceAll("_", " "), value: (run) => run.metrics[name], bar: true })),
    ...(observed.length ? [{ section: "Observed on every run" }] : []),
    ...observed.map((name) => ({ label: name.replaceAll("_", " "), value: (run) => run.observations?.[name], bar: true })),
    { label: "wall time (s)", value: (run) => run.wall_time, bar: true },
    { label: "elements per rod", value: (run) => run.n_elements, bar: false },
  ];

  const head = el("tr", {}, el("th", { scope: "col" }, "Metric"),
    ...runs.map((run) => el("th", { scope: "col" }, swatch(seriesVar(run.solver)), run.solver)));
  const body = rows.map((row) => {
    if (row.section) return el("tr", { class: "section" }, el("th", { scope: "rowgroup", colspan: runs.length + 1 }, row.section));
    const values = runs.map((run) => (run.trajectory || !row.bar ? row.value(run) : undefined));
    const largest = Math.max(...values.filter((v) => v != null).map(Math.abs));
    return el("tr", {}, el("th", { scope: "row" }, row.label), ...runs.map((run, i) => {
      if (values[i] == null) return el("td", { class: "na" }, "n/a");
      const cell = el("td", {}, formatNumber(values[i]));
      if (row.bar && largest > 0) {
        const bar = el("span", { class: "bar" });
        bar.style.width = `${(100 * Math.abs(values[i])) / largest}%`;
        bar.style.background = `var(${seriesVar(run.solver)})`;
        cell.append(bar);
      }
      return cell;
    }));
  });
  $("metrics").replaceChildren(el("thead", {}, head), el("tbody", {}, ...body));

  $("problems").replaceChildren(...runs.flatMap((run) => {
    if (run.outcome === "unsupported") {
      return [el("li", {}, `${run.solver} was not run: its model cannot represent ${run.missing.join(", ")}.`)];
    }
    if (run.outcome === "diverged") {
      const when = run.diverged_at == null ? "" : ` at t = ${formatNumber(run.diverged_at)} s`;
      return [el("li", {}, `${run.solver} diverged${when}: ${run.failure}.`)];
    }
    return [];
  }));
}

// ---------------------------------------------------------------- scenario

function buildScenario() {
  const { scenario, rod_lengths: lengths } = state.variant;
  const items = [];
  // Rods that differ only in placement and motion are described once.
  const describe = (rod, i) => JSON.stringify([rod.radius, rod.material, rod.start, rod.end, rod.loads, lengths[i]]);
  const alike = scenario.rods.every((rod, i) => describe(rod, i) === describe(scenario.rods[0], 0));
  const rods = alike ? scenario.rods.slice(0, 1) : scenario.rods;
  rods.forEach((rod, i) => {
    const of = alike ? (scenario.rods.length > 1 ? ` (each of ${scenario.rods.length} rods)` : "") : ` (rod ${i + 1})`;
    items.push(
      [`Length${of}`, si(lengths[i], "m")],
      [`Radius${of}`, si(rod.radius, "m")],
      [`Young's modulus${of}`, si(rod.material.youngs_modulus, "Pa")],
      [`Shear modulus${of}`, si(rod.material.shear_modulus, "Pa")],
      [`Density${of}`, `${formatNumber(rod.material.density)} kg/m³`],
      [`Ends${of}`, `${rod.start_motion ? "driven" : rod.start} start, ${rod.end_motion ? "driven" : rod.end} end`],
      ...rod.loads.map((load) => [`Load${of}`, `${si(Math.hypot(...load.force), "N")} at ${load.at} along ${direction(load.force)}`]),
    );
  });
  const g = Math.hypot(...scenario.gravity);
  items.push(
    ["Gravity", g ? `${formatNumber(g)} m/s² along ${direction(scenario.gravity)}` : "none"],
    ["Duration", `${formatNumber(scenario.duration)} s`],
    ["Judged on", scenario.quasi_static ? "final equilibrium; solvers may add damping to reach it" : "the motion itself; no added damping"],
  );
  $("scenario").replaceChildren(...items.map(([term, value]) => el("div", {}, el("dt", {}, term), el("dd", {}, value))));
}

// ---------------------------------------------------------------- speed chart

const chart = (() => {
  const container = $("chart");
  const tooltip = el("div", { class: "tooltip", hidden: "" });
  const HEIGHT = 200, MARGIN = { left: 46, right: 14, top: 8, bottom: 26 };
  let x, y, playhead, hover, series = [], frameTimes = [];

  function draw() {
    series = state.entries.filter((entry) => entry.shown);
    const width = container.clientWidth;
    if (!series.length || !width) return container.replaceChildren();

    const plotWidth = width - MARGIN.left - MARGIN.right;
    const plotHeight = HEIGHT - MARGIN.top - MARGIN.bottom;
    const FLOOR = 1e-12; // a node at rest has no logarithm
    const logs = series.flatMap((entry) => [...entry.speeds].filter(Number.isFinite).map((v) => Math.log10(Math.max(v, FLOOR))));
    const low = Math.floor(Math.min(...logs)), high = Math.max(Math.ceil(Math.max(...logs)), low + 1);
    x = (time) => MARGIN.left + (time / state.duration) * plotWidth;
    y = (speed) => MARGIN.top + ((high - Math.log10(Math.max(speed, FLOOR))) / (high - low)) * plotHeight;
    frameTimes = series[0].times;

    const root = svg("svg", { width, height: HEIGHT, viewBox: `0 0 ${width} ${HEIGHT}`, tabindex: 0, role: "img",
      "aria-label": "Speed of the fastest node over time, one line per solver. The same values are in the table below." });

    const decadeStep = Math.ceil((high - low) / 6);
    for (let decade = low; decade <= high; decade += decadeStep) {
      const py = y(10 ** decade);
      root.append(
        svg("line", { class: decade === low ? "baseline" : "gridline", x1: MARGIN.left, x2: width - MARGIN.right, y1: py, y2: py }),
        svg("text", { x: MARGIN.left - 8, y: py + 4, "text-anchor": "end" }, `1e${decade}`),
      );
    }
    const step = niceStep(state.duration, 6);
    for (let t = 0; t <= state.duration + 1e-9; t += step) {
      root.append(svg("text", { x: x(t), y: HEIGHT - 8, "text-anchor": "middle" }, `${Number(t.toFixed(6))} s`));
    }

    for (const entry of series) {
      const points = [...entry.speeds].map((v, f) => (Number.isFinite(v) ? `${x(entry.times[f]).toFixed(1)},${y(v).toFixed(1)}` : null));
      const path = svg("path", { class: "series", d: `M${points.filter(Boolean).join("L")}` });
      path.style.stroke = `var(${entry.colorVar})`;
      root.append(path);
    }

    playhead = svg("line", { class: "playhead", y1: MARGIN.top, y2: HEIGHT - MARGIN.bottom });
    hover = svg("g", { visibility: "hidden" }, svg("line", { class: "crosshair", y1: MARGIN.top, y2: HEIGHT - MARGIN.bottom }),
      ...series.map((entry) => {
        const dot = svg("circle", { class: "dot", r: 5 });
        dot.style.fill = `var(${entry.colorVar})`;
        return dot;
      }));
    root.append(playhead, hover);

    let focused = 1;
    const nearest = (event) => {
      const time = ((event.clientX - root.getBoundingClientRect().left - MARGIN.left) / plotWidth) * state.duration;
      return Math.min(Math.max(Math.round((time / frameTimes.at(-1)) * (frameTimes.length - 1)), 1), frameTimes.length - 1);
    };
    root.addEventListener("pointermove", (event) => inspect((focused = nearest(event))));
    root.addEventListener("pointerleave", () => inspect(null));
    root.addEventListener("blur", () => inspect(null));
    root.addEventListener("focus", () => inspect(focused));
    root.addEventListener("click", (event) => {
      setPlaying(false);
      seek(frameTimes[nearest(event)]);
    });
    root.addEventListener("keydown", (event) => {
      const move = { ArrowLeft: -1, ArrowRight: 1 }[event.key];
      if (!move) return;
      event.preventDefault();
      focused = Math.min(Math.max(focused + move, 1), frameTimes.length - 1);
      inspect(focused);
    });

    tooltip.hidden = true; // its position belonged to the previous layout
    container.replaceChildren(root, tooltip, table());
    movePlayhead();
  }

  /** The same numbers as the lines, for readers who cannot use the plot. */
  function table() {
    const head = el("tr", {}, el("th", { scope: "col" }, "Time (s)"),
      ...series.map((entry) => el("th", { scope: "col" }, swatch(entry.colorVar), entry.run.solver)));
    const rows = frameTimes.slice(1).map((time, i) => el("tr", {}, el("th", { scope: "row" }, time.toFixed(2)),
      ...series.map((entry) => el("td", {}, formatNumber(entry.speeds[i + 1])))));
    const open = container.querySelector("details")?.open ?? false;
    const details = el("details", {}, el("summary", {}, "Show as table"),
      el("div", { class: "table-scroll tall" }, el("table", {}, el("thead", {}, head), el("tbody", {}, ...rows))));
    details.open = open;
    return details;
  }

  /** Crosshair and readout for one frame: every solver's value, not just the line under the pointer. */
  function inspect(frame) {
    tooltip.hidden = frame == null;
    hover?.setAttribute("visibility", frame == null ? "hidden" : "visible");
    if (frame == null) return;

    const px = x(frameTimes[frame]);
    const [line, ...dots] = hover.children;
    line.setAttribute("x1", px);
    line.setAttribute("x2", px);
    dots.forEach((dot, i) => {
      dot.setAttribute("cx", px);
      dot.setAttribute("cy", y(series[i].speeds[frame]));
    });
    tooltip.replaceChildren(
      el("div", { class: "when" }, `t = ${frameTimes[frame].toFixed(2)} s`),
      ...series.map((entry) => el("div", { class: "row" }, swatch(entry.colorVar, "line"),
        el("strong", {}, formatNumber(entry.speeds[frame])), el("span", {}, entry.run.solver))),
    );
    const flip = px + tooltip.offsetWidth + 16 > container.clientWidth;
    const left = flip ? px - tooltip.offsetWidth - 12 : px + 12;
    const furthest = container.clientWidth - tooltip.offsetWidth;
    tooltip.style.transform = `translate(${Math.max(0, Math.min(left, furthest))}px, ${MARGIN.top}px)`;
  }

  function movePlayhead() {
    if (!playhead || !x) return;
    playhead.setAttribute("x1", x(state.time));
    playhead.setAttribute("x2", x(state.time));
  }

  return { draw, movePlayhead };
})();

// ---------------------------------------------------------------- wiring

$("experiments").replaceChildren(...manifest.experiments.map((experiment) => {
  const button = el("button", { type: "button", "data-name": experiment.name },
    el("strong", {}, experiment.name), el("span", {}, experiment.description));
  button.addEventListener("click", () => (location.hash = experiment.name));
  return button;
}));

for (const button of document.querySelectorAll("[data-mode]")) {
  button.addEventListener("click", () => {
    state.mode = button.dataset.mode;
    for (const other of document.querySelectorAll("[data-mode]")) {
      other.setAttribute("aria-pressed", String(other === button));
    }
    layoutChanged();
    frameCamera(); // panes are narrower than the whole viewport
  });
}

$("play").addEventListener("click", () => setPlaying(!state.playing));
$("scrub").addEventListener("input", (event) => {
  setPlaying(false);
  seek(Number(event.target.value));
});
$("speed").addEventListener("change", (event) => (state.speed = Number(event.target.value)));
document.addEventListener("keydown", (event) => {
  if (event.code !== "Space" || event.target.closest("button, select, input, svg")) return;
  event.preventDefault();
  setPlaying(!state.playing);
});

new ResizeObserver(() => {
  // Keep the experiment in frame as the viewport changes shape, unless the user chose their own view.
  if (state.experiment && !state.cameraMoved) frameCamera();
  invalidate();
}).observe($("viewport"));
new ResizeObserver(() => chart.draw()).observe($("chart"));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", recolor);

function route() {
  const [name, key] = decodeURIComponent(location.hash.slice(1)).split("/");
  select(manifest.experiments.find((experiment) => experiment.name === name) ?? manifest.experiments[0], key);
}
addEventListener("hashchange", route);
route();
requestAnimationFrame(tick);
