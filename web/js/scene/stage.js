// Renderer, camera, controls, glow and the frame loop. Knows nothing about Somni.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { SCENE } from "./palette.js";

const FLY_SECONDS = 1.1;
export const FLY_MS = FLY_SECONDS * 1000;
const tmpProjected = new THREE.Vector3();
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

export class Stage {
  #tickers = new Set();
  #clock = new THREE.Clock();
  #flight = null;
  #lastTouched = 0;             // when the user last moved the camera (performance.now())
  #bloomOn = true;
  #insets = { left: 0, right: 0 };

  constructor(host) {
    this.host = host;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(SCENE.background);
    this.scene.fog = new THREE.FogExp2(SCENE.background, SCENE.fogDensity);

    this.camera = new THREE.PerspectiveCamera(52, 1, 0.5, 6000);
    this.camera.position.set(-30, 60, 420);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(this.renderer.domElement);

    this.labels = new CSS2DRenderer();
    this.labels.domElement.className = "label-layer";
    Object.assign(this.labels.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none" });
    host.appendChild(this.labels.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.6;
    this.controls.zoomSpeed = 0.9;
    this.controls.minDistance = 8;
    this.controls.maxDistance = 1600;
    this.controls.autoRotateSpeed = 0.35;
    this.controls.addEventListener("start", () => {
      this.#flight = null;
      this.#lastTouched = performance.now();
    });

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    const { strength, radius, threshold } = SCENE.bloom;
    this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), strength, radius, threshold);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.#addStarfield();
    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    this.renderer.setAnimationLoop(() => this.#frame());
  }

  /** Panels covering the canvas edges (px). The projection is shifted so the scene's centre sits in
   * the middle of the uncovered area; picking stays exact because the shift is in the projection. */
  setInsets(left = 0, right = 0) {
    this.#insets = { left, right };
    this.resize();
  }

  get visibleAspect() {
    const w = Math.max(1, this.host.clientWidth - this.#insets.left - this.#insets.right);
    return w / Math.max(1, this.host.clientHeight);
  }

  resize() {
    const w = Math.max(1, this.host.clientWidth);
    const h = Math.max(1, this.host.clientHeight);
    this.camera.aspect = w / h;
    const shift = (this.#insets.left - this.#insets.right) / 2;
    if (shift) this.camera.setViewOffset(w, h, -shift, 0, w, h);
    else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
    this.labels.setSize(w, h);
  }

  /** Register a per-frame callback (dt seconds, elapsed seconds). Returns an unregister function. */
  onTick(fn) {
    this.#tickers.add(fn);
    return () => this.#tickers.delete(fn);
  }

  set bloomEnabled(on) {
    this.#bloomOn = on;
  }

  set autoRotate(on) {
    this.controls.autoRotate = on;
  }

  /** Animate the camera to look at `target` from `distance` away (keeps the current direction). */
  flyTo(target, distance) {
    const to = new THREE.Vector3().copy(target);
    const dir = new THREE.Vector3().subVectors(this.camera.position, this.controls.target).normalize();
    const camTo = to.clone().addScaledVector(dir, distance ?? this.camera.position.distanceTo(this.controls.target));
    this.#flight = {
      t: 0,
      fromTarget: this.controls.target.clone(),
      fromCam: this.camera.position.clone(),
      toTarget: to,
      toCam: camTo,
    };
  }

  /** Seconds since the user last moved the camera (Infinity if never). */
  get idleSeconds() {
    return this.#lastTouched ? (performance.now() - this.#lastTouched) / 1000 : Infinity;
  }

  /** Is this point on screen (inside the visible part of the view)? */
  inView(point, margin = 0.92) {
    tmpProjected.copy(point).project(this.camera);
    return tmpProjected.z < 1 && Math.abs(tmpProjected.x) < margin && Math.abs(tmpProjected.y) < margin;
  }

  /** Frame a bounding box in view. */
  frameBox(box, margin = 1.08) {
    const centre = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const fitH = (size.y / 2) / Math.tan(fov / 2);
    const fitW = (size.x / 2) / Math.tan(fov / 2) / this.visibleAspect;
    const distance = Math.max(fitH, fitW) * margin + size.z / 2;
    this.#flight = {
      t: 0,
      fromTarget: this.controls.target.clone(),
      fromCam: this.camera.position.clone(),
      toTarget: centre,
      toCam: centre.clone().add(new THREE.Vector3(0, distance * 0.18, distance)),
    };
  }

  #frame() {
    const dt = Math.min(this.#clock.getDelta(), 0.1);
    const elapsed = this.#clock.elapsedTime;
    if (this.#flight) {
      const f = this.#flight;
      f.t = Math.min(1, f.t + dt / FLY_SECONDS);
      const k = easeInOut(f.t);
      this.controls.target.lerpVectors(f.fromTarget, f.toTarget, k);
      this.camera.position.lerpVectors(f.fromCam, f.toCam, k);
      if (f.t >= 1) this.#flight = null;
    }
    this.controls.update();
    for (const tick of this.#tickers) tick(dt, elapsed);
    if (this.#bloomOn) this.composer.render(dt);
    else this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
  }

  #addStarfield() {
    const count = 2200;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const r = 1400 + Math.random() * 1600;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      positions.set([r * Math.sin(phi) * Math.cos(theta), r * Math.cos(phi), r * Math.sin(phi) * Math.sin(theta)], i * 3);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({ color: 0x8090c0, size: 1.6, sizeAttenuation: false, fog: false,
      transparent: true, opacity: 0.55, depthWrite: false });
    this.scene.add(new THREE.Points(geo, mat));
  }
}
