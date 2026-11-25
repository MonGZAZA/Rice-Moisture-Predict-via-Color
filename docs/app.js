/* Interactive Rice Paddy Moisture Prediction Pipeline Demo (IEEE iSAI-NLP 2025) */

let DEMO_DATA = null;
let activeSample = null;
let activeCategory = "ALL";
let searchQuery = "";
let fig6HitPoints = [];

async function initDemo() {
  const res = await fetch("data/demo_data.json");
  DEMO_DATA = await res.json();

  renderFeaturedGrid();
  populateAllSamplesDropdown();
  renderAnovaTable();
  bindEvents();

  // Default to featured sample 0027.jpg (Actual 15.2%, Predicted 15.16%)
  const defaultSample =
    DEMO_DATA.test_samples.find((s) => s.id === 27) ||
    DEMO_DATA.test_samples.find((s) => s.featured) ||
    DEMO_DATA.test_samples[0];
  selectSample(defaultSample);
}

function renderFeaturedGrid() {
  const grid = document.getElementById("featuredGrid");
  grid.innerHTML = "";
  const featuredSamples = DEMO_DATA.test_samples.filter((s) => s.featured);

  featuredSamples.forEach((s) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "sample-card";
    btn.dataset.id = String(s.id);
    btn.innerHTML = `
      <img class="sample-thumb" src="${s.images.cropped}" alt="Sample ${s.filename}" loading="lazy" />
      <div class="sample-info">
        <div class="sample-top">
          <span>${s.filename}</span>
          <span>#${s.test_rank + 1}/45</span>
        </div>
        <div class="sample-badges">
          <span class="tag-actual">GT: ${s.actual.toFixed(1)}%</span>
          <span class="tag-pred">Pred: ${s.predicted.toFixed(1)}%</span>
        </div>
      </div>
    `;
    btn.addEventListener("click", () => selectSample(s));
    grid.appendChild(btn);
  });
}

function populateAllSamplesDropdown() {
  const sel = document.getElementById("allSamplesSelect");
  sel.innerHTML = "";
  DEMO_DATA.test_samples.forEach((s) => {
    const opt = document.createElement("option");
    opt.value = String(s.id);
    const star = s.featured ? "★ " : "";
    opt.textContent = `${star}${s.filename} — Actual: ${s.actual.toFixed(1)}% | Pred: ${s.predicted.toFixed(2)}%`;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", () => {
    const id = Number(sel.value);
    const found = DEMO_DATA.test_samples.find((s) => s.id === id);
    if (found) selectSample(found);
  });
}

function selectSample(sample) {
  activeSample = sample;

  // Highlight featured card if applicable
  document.querySelectorAll(".sample-card").forEach((el) => {
    el.classList.toggle("active", Number(el.dataset.id) === sample.id);
  });

  const sel = document.getElementById("allSamplesSelect");
  if (sel && sample.id > 0) {
    sel.value = String(sample.id);
  }

  // Update active banner
  document.getElementById("activeSampleBadge").textContent =
    sample.id > 0 ? `Test Sample #${sample.test_rank + 1} / 45` : "Custom Image";
  document.getElementById("activeSampleTitle").textContent =
    sample.id > 0
      ? `${sample.filename} — Held-Out Test Sample`
      : `${sample.filename} — Custom Uploaded Image`;
  document.getElementById("activeSampleMeta").textContent =
    sample.id > 0
      ? "FUJIFILM X100T · 4896×3264 px · ISO 200 · f/11 · 20×20 cm Lightbox"
      : "Processed live in browser via HTML5 Canvas HSV & CIELab pipeline";

  document.getElementById("kpiActual").textContent =
    sample.actual != null ? `${sample.actual.toFixed(2)}%` : "N/A (Custom)";
  document.getElementById("kpiPred").textContent = `${sample.predicted.toFixed(2)}%`;
  document.getElementById("kpiError").textContent =
    sample.actual != null
      ? `${sample.abs_error.toFixed(2)}% (${sample.pct_error.toFixed(2)}% APE)`
      : "—";

  // Update visual stage images
  const v1 = document.getElementById("stage1Visuals");
  const v2 = document.getElementById("stage2Visuals");
  const v3 = document.getElementById("stage3Visuals");
  const ratioBox = document.getElementById("segmentRatioContainer");
  const notice = document.getElementById("noVisualNotice1");

  if (sample.images) {
    v1.hidden = false;
    v2.hidden = false;
    v3.hidden = false;
    ratioBox.hidden = false;
    notice.hidden = true;

    document.getElementById("imgOriginal").src = sample.images.original;
    document.getElementById("imgCropped").src = sample.images.cropped;
    document.getElementById("imgHueMap").src = sample.images.hue_map;
    document.getElementById("imgDenoised").src = sample.images.denoised;
    document.getElementById("imgGreen").src = sample.images.green;
    document.getElementById("imgBrown").src = sample.images.brown;

    document.getElementById("artifactRemovedText").innerHTML =
      `Pixels in <code>H ∈ [100, 140]</code> removed (${sample.artifact_removed_pct.toFixed(3)}% of ROI)`;
    document.getElementById("greenPctLabel").textContent = `${sample.green_segment_pct.toFixed(2)}%`;
    document.getElementById("brownPctLabel").textContent = `${sample.brown_segment_pct.toFixed(2)}%`;
    document.getElementById("greenRatioBar").style.width = `${Math.min(100, Math.max(0, sample.green_segment_pct))}%`;
  } else {
    v1.hidden = true;
    v2.hidden = true;
    v3.hidden = true;
    ratioBox.hidden = true;
    notice.hidden = false;
    notice.innerHTML =
      `Pre-rendered stage images are bundled for the 12 starred (★) featured samples to keep the site lightweight. ` +
      `All <strong>38 extracted color features</strong>, <strong>z-scores</strong>, and <strong>OLS predictions</strong> for <code>${sample.filename}</code> are loaded below in Stages 4 and 5.`;
  }

  renderFeatureTable();
  renderPredictionPanel();
  drawFig6Chart();
}

function renderFeatureTable() {
  const tbody = document.getElementById("featureTableBody");
  tbody.innerHTML = "";
  if (!activeSample || !DEMO_DATA) return;

  const feats = DEMO_DATA.model.features;
  const q = searchQuery.trim().toLowerCase();

  feats.forEach((f, i) => {
    if (activeCategory !== "ALL" && f.group !== activeCategory) return;
    if (
      q &&
      !f.name.toLowerCase().includes(q) &&
      !f.description.toLowerCase().includes(q) &&
      !f.group.toLowerCase().includes(q)
    ) {
      return;
    }

    const rawVal = activeSample.raw_features[i];
    const zVal = activeSample.z_scores[i];
    const termVal = activeSample.z_contributions[i];
    const rClass = f.pearson_r >= 0 ? "r-pos" : "r-neg";
    const segClass = f.segment === "Green" ? "seg-green" : "seg-all";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${f.index}</td>
      <td title="${f.description}"><strong><code>${f.name}</code></strong></td>
      <td><span class="seg-badge ${segClass}">${f.segment}</span></td>
      <td>${f.group}</td>
      <td class="num-col">${formatNum(rawVal)}</td>
      <td class="num-col">${formatNum(f.train_mean)} ± ${formatNum(f.train_std)}</td>
      <td class="num-col">${zVal >= 0 ? "+" : ""}${zVal.toFixed(3)}</td>
      <td class="num-col ${rClass}">${f.pearson_r >= 0 ? "+" : ""}${f.pearson_r.toFixed(4)}</td>
      <td class="num-col">${f.beta_std >= 0 ? "+" : ""}${f.beta_std.toFixed(3)}</td>
      <td class="num-col"><strong>${termVal >= 0 ? "+" : ""}${termVal.toFixed(3)}</strong></td>
    `;
    tbody.appendChild(tr);
  });
}

function renderPredictionPanel() {
  if (!activeSample || !DEMO_DATA) return;
  const minM = 8.0;
  const maxM = 32.0;

  if (activeSample.actual != null) {
    const pctA = ((activeSample.actual - minM) / (maxM - minM)) * 100;
    document.getElementById("barActual").style.width = `${Math.min(100, Math.max(4, pctA))}%`;
    document.getElementById("gaugeActualVal").textContent = `${activeSample.actual.toFixed(2)}%`;
  } else {
    document.getElementById("barActual").style.width = "0%";
    document.getElementById("gaugeActualVal").textContent = "N/A";
  }

  const pctP = ((activeSample.predicted - minM) / (maxM - minM)) * 100;
  document.getElementById("barPred").style.width = `${Math.min(100, Math.max(4, pctP))}%`;
  document.getElementById("gaugePredVal").textContent = `${activeSample.predicted.toFixed(2)}%`;

  const b0 = DEMO_DATA.model.intercept_std;
  const sumTerms = activeSample.z_contributions.reduce((a, b) => a + b, 0);
  document.getElementById("eqBreakdownText").innerHTML =
    `ŷ = β₀<sup>std</sup> (${b0.toFixed(3)}%) + ∑<sub>j=1..38</sub> β<sub>j</sub><sup>std</sup>·z<sub>j</sub> (${sumTerms >= 0 ? "+" : ""}${sumTerms.toFixed(3)}%) = <strong>${activeSample.predicted.toFixed(3)}%</strong>`;
}

function renderAnovaTable() {
  const tbody = document.getElementById("anovaTableBody");
  tbody.innerHTML = "";
  DEMO_DATA.model.anova.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${row.Source}</strong></td>
      <td class="num-col">${row.SS != null ? Number(row.SS).toFixed(3) : "—"}</td>
      <td class="num-col">${row.df != null ? row.df : "—"}</td>
      <td class="num-col">${row.MS != null ? Number(row.MS).toFixed(4) : "—"}</td>
      <td class="num-col">${row.F != null ? Number(row.F).toFixed(3) : "—"}</td>
      <td class="num-col">${row["p-value"] != null ? Number(row["p-value"]).toExponential(2) : "—"}</td>
    `;
    tbody.appendChild(tr);
  });
}

function drawFig6Chart() {
  const canvas = document.getElementById("fig6Canvas");
  if (!canvas || !DEMO_DATA) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const padL = 52;
  const padR = 22;
  const padT = 22;
  const padB = 42;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const samples = DEMO_DATA.test_samples;
  const n = samples.length;
  const yMin = 8.0;
  const yMax = 32.0;

  const xOf = (idx) => padL + (idx / Math.max(1, n - 1)) * plotW;
  const yOf = (val) => padT + plotH - ((val - yMin) / (yMax - yMin)) * plotH;

  // Gridlines
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#64748b";
  ctx.font = "12px -apple-system, BlinkMacSystemFont, sans-serif";

  for (let yv = 10; yv <= 30; yv += 5) {
    const py = yOf(yv);
    ctx.beginPath();
    ctx.moveTo(padL, py);
    ctx.lineTo(W - padR, py);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(`${yv}%`, padL - 8, py + 4);
  }

  for (let xv = 0; xv < n; xv += 10) {
    const px = xOf(xv);
    ctx.beginPath();
    ctx.moveTo(px, padT);
    ctx.lineTo(px, H - padB);
    ctx.stroke();
    ctx.textAlign = "center";
    ctx.fillText(String(xv), px, H - padB + 18);
  }

  ctx.textAlign = "center";
  ctx.fillStyle = "#334155";
  ctx.font = "600 12px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.fillText("Held-Out Test Sample Index (sorted by actual moisture)", padL + plotW / 2, H - 8);

  fig6HitPoints = [];

  // Draw vertical error connectors and dots
  samples.forEach((s, idx) => {
    const px = xOf(idx);
    const pyAct = yOf(s.actual);
    const pyPred = yOf(s.predicted);
    const isSel = activeSample && activeSample.id === s.id;

    // Residual connector line
    ctx.strokeStyle = isSel ? "rgba(22, 163, 74, 0.55)" : "rgba(148, 163, 184, 0.35)";
    ctx.lineWidth = isSel ? 2.5 : 1;
    ctx.beginPath();
    ctx.moveTo(px, pyAct);
    ctx.lineTo(px, pyPred);
    ctx.stroke();

    // Actual dot (blue)
    ctx.fillStyle = "#2563eb";
    ctx.beginPath();
    ctx.arc(px, pyAct, isSel ? 6 : 4, 0, Math.PI * 2);
    ctx.fill();

    // Predicted dot (orange)
    ctx.fillStyle = "#ea580c";
    ctx.beginPath();
    ctx.arc(px, pyPred, isSel ? 6 : 4, 0, Math.PI * 2);
    ctx.fill();

    if (isSel) {
      ctx.strokeStyle = "#16a34a";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(px, pyAct, 10, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(px, pyPred, 10, 0, Math.PI * 2);
      ctx.stroke();
    }

    fig6HitPoints.push({ sample: s, x: px, yAct: pyAct, yPred: pyPred });
  });
}

function bindEvents() {
  // Step tabs
  document.querySelectorAll(".step-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".step-tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const target = btn.dataset.step;
      document.querySelectorAll(".pipeline-stage").forEach((sec) => {
        sec.hidden = target !== "all" && sec.dataset.stage !== target;
      });
    });
  });

  // Feature category filter pills
  document.querySelectorAll("#featureCategoryPills .pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      document.querySelectorAll("#featureCategoryPills .pill").forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      activeCategory = pill.dataset.group;
      renderFeatureTable();
    });
  });

  // Search box
  document.getElementById("featureSearchInput").addEventListener("input", (e) => {
    searchQuery = e.target.value;
    renderFeatureTable();
  });

  // Click on Fig. 6 canvas to select sample
  const canvas = document.getElementById("fig6Canvas");
  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const cx = (e.clientX - rect.left) * scaleX;
    const cy = (e.clientY - rect.top) * scaleY;

    let best = null;
    let bestDist = 999999;
    fig6HitPoints.forEach((pt) => {
      const d1 = Math.hypot(cx - pt.x, cy - pt.yAct);
      const d2 = Math.hypot(cx - pt.x, cy - pt.yPred);
      const d = Math.min(d1, d2);
      if (d < bestDist) {
        bestDist = d;
        best = pt.sample;
      }
    });
    if (best && bestDist <= 28) {
      selectSample(best);
    }
  });

  // Animated walkthrough button
  document.getElementById("runAnimationBtn").addEventListener("click", async () => {
    // Ensure all stages are visible
    document.querySelector('.step-tab[data-step="all"]').click();
    const stages = ["stage1Card", "stage2Card", "stage3Card", "stage4Card", "stage5Card"];
    for (const id of stages) {
      const el = document.getElementById(id);
      if (!el || el.hidden) continue;
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      el.classList.add("highlight-pulse");
      await new Promise((r) => setTimeout(r, 900));
      el.classList.remove("highlight-pulse");
    }
  });

  // Custom image upload -> live browser canvas pipeline
  document.getElementById("customImageInput").addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    processCustomImageFile(file);
  });
}

function processCustomImageFile(file) {
  const reader = new FileReader();
  reader.onload = (ev) => {
    const img = new Image();
    img.onload = () => {
      const customSample = runBrowserPipelineOnImage(img, file.name);
      selectSample(customSample);
    };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
}

function runBrowserPipelineOnImage(imgEl, filename) {
  const canvas = document.getElementById("offscreenCanvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  // Scale down to max 420px for fast browser pixel extraction
  const maxW = 420;
  const scale = imgEl.width > maxW ? maxW / imgEl.width : 1;
  const W = Math.max(1, Math.round(imgEl.width * scale));
  const H = Math.max(1, Math.round(imgEl.height * scale));
  canvas.width = W;
  canvas.height = H;
  ctx.drawImage(imgEl, 0, 0, W, H);
  const origDataUrl = canvas.toDataURL("image/jpeg", 0.85);

  // Crop region proportional to (586, 0, 4252, 3164) on 4896x3264 if aspect matches
  const x1 = Math.floor(W * (586 / 4896));
  const x2 = Math.ceil(W * (4252 / 4896));
  const y1 = 0;
  const y2 = Math.ceil(H * (3164 / 3264));
  const cW = Math.max(1, x2 - x1);
  const cH = Math.max(1, y2 - y1);

  const cropImgData = ctx.getImageData(x1, y1, cW, cH);
  const src = cropImgData.data;

  // Helper canvas for stage snapshots
  const stageCanvas = document.createElement("canvas");
  stageCanvas.width = cW;
  stageCanvas.height = cH;
  const sCtx = stageCanvas.getContext("2d");
  sCtx.putImageData(cropImgData, 0, 0);
  const croppedUrl = stageCanvas.toDataURL("image/jpeg", 0.85);

  const hueImg = sCtx.createImageData(cW, cH);
  const denImg = sCtx.createImageData(cW, cH);
  const grnImg = sCtx.createImageData(cW, cH);
  const brnImg = sCtx.createImageData(cW, cH);

  const allPixels = [];
  const grnPixels = [];
  let artifactCount = 0;
  let validCount = 0;
  let grnCount = 0;
  let brnCount = 0;

  for (let i = 0; i < src.length; i += 4) {
    const r = src[i];
    const g = src[i + 1];
    const b = src[i + 2];
    const [h180, s255, v255] = rgbToOpenCvHsv(r, g, b);

    // Hue colormap pixel
    const [hr, hg, hb] = hueToRainbowRgb(h180);
    hueImg.data[i] = hr;
    hueImg.data[i + 1] = hg;
    hueImg.data[i + 2] = hb;
    hueImg.data[i + 3] = 255;

    if (h180 >= 100 && h180 <= 140) {
      artifactCount++;
      denImg.data[i + 3] = 255;
      grnImg.data[i + 3] = 255;
      brnImg.data[i + 3] = 255;
      continue;
    }

    denImg.data[i] = r;
    denImg.data[i + 1] = g;
    denImg.data[i + 2] = b;
    denImg.data[i + 3] = 255;

    if (r === 0 && g === 0 && b === 0) continue;
    validCount++;
    const lab = rgbToOpenCvLab(r, g, b);
    const pxObj = { r, g, b, h360: h180 * 2, s255, v255, L: lab[0], a: lab[1], bLab: lab[2] };
    allPixels.push(pxObj);

    if (h180 >= 25 && h180 <= 85 && s255 >= 35 && v255 >= 35) {
      grnCount++;
      grnPixels.push(pxObj);
      grnImg.data[i] = r;
      grnImg.data[i + 1] = g;
      grnImg.data[i + 2] = b;
      grnImg.data[i + 3] = 255;
      brnImg.data[i + 3] = 255;
    } else {
      brnCount++;
      brnImg.data[i] = r;
      brnImg.data[i + 1] = g;
      brnImg.data[i + 2] = b;
      brnImg.data[i + 3] = 255;
      grnImg.data[i + 3] = 255;
    }
  }

  sCtx.putImageData(hueImg, 0, 0);
  const hueUrl = stageCanvas.toDataURL("image/jpeg", 0.85);
  sCtx.putImageData(denImg, 0, 0);
  const denUrl = stageCanvas.toDataURL("image/jpeg", 0.85);
  sCtx.putImageData(grnImg, 0, 0);
  const grnUrl = stageCanvas.toDataURL("image/jpeg", 0.85);
  sCtx.putImageData(brnImg, 0, 0);
  const brnUrl = stageCanvas.toDataURL("image/jpeg", 0.85);

  const allFeats = computeSegmentFeatures(allPixels);
  const grnFeats = computeSegmentFeatures(grnPixels.length > 10 ? grnPixels : allPixels);

  const featsMeta = DEMO_DATA.model.features;
  const rawVals = [];
  const zVals = [];
  const contribs = [];

  featsMeta.forEach((f) => {
    const pool = f.segment === "All" ? allFeats : grnFeats;
    const val = Number(pool[f.name] ?? f.train_mean);
    const z = (val - f.train_mean) / f.train_std;
    rawVals.push(val);
    zVals.push(z);
    contribs.push(z * f.beta_std);
  });

  const pred =
    DEMO_DATA.model.intercept_std + contribs.reduce((acc, v) => acc + v, 0);

  return {
    test_rank: -1,
    id: -1,
    filename,
    actual: null,
    predicted: Math.max(5.0, Math.min(35.0, pred)),
    abs_error: 0,
    pct_error: 0,
    featured: true,
    artifact_removed_pct: (artifactCount / Math.max(1, cW * cH)) * 100,
    green_segment_pct: (grnCount / Math.max(1, validCount)) * 100,
    brown_segment_pct: (brnCount / Math.max(1, validCount)) * 100,
    images: {
      original: origDataUrl,
      cropped: croppedUrl,
      hue_map: hueUrl,
      denoised: denUrl,
      green: grnUrl,
      brown: brnUrl,
    },
    raw_features: rawVals,
    z_scores: zVals,
    z_contributions: contribs,
  };
}

function rgbToOpenCvHsv(r, g, b) {
  const rf = r / 255;
  const gf = g / 255;
  const bf = b / 255;
  const max = Math.max(rf, gf, bf);
  const min = Math.min(rf, gf, bf);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === rf) h = 60 * (((gf - bf) / d) % 6);
    else if (max === gf) h = 60 * ((bf - rf) / d + 2);
    else h = 60 * ((rf - gf) / d + 4);
  }
  if (h < 0) h += 360;
  const s = max === 0 ? 0 : d / max;
  return [Math.round(h / 2), Math.round(s * 255), Math.round(max * 255)];
}

function rgbToOpenCvLab(r, g, b) {
  const lin = (c) => {
    const v = c / 255;
    return v > 0.04045 ? Math.pow((v + 0.055) / 1.055, 2.4) : v / 12.92;
  };
  const lr = lin(r), lg = lin(g), lb = lin(b);
  const X = (lr * 0.4124564 + lg * 0.3575761 + lb * 0.1804375) / 0.95047;
  const Y = lr * 0.2126729 + lg * 0.7151522 + lb * 0.0721750;
  const Z = (lr * 0.0193339 + lg * 0.1191920 + lb * 0.9503041) / 1.08883;
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f(X), fy = f(Y), fz = f(Z);
  const L = (116 * fy - 16) * (255 / 100);
  const a = 500 * (fx - fy) + 128;
  const bLab = 200 * (fy - fz) + 128;
  return [L, a, bLab];
}

function hueToRainbowRgb(h180) {
  const h = (h180 / 180) * 360;
  const c = 1;
  const x = 1 - Math.abs(((h / 60) % 2) - 1);
  let r = 0, g = 0, b = 0;
  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function computeSegmentFeatures(pxList) {
  const n = pxList.length;
  if (n === 0) return {};
  const arr = (fn) => Float64Array.from(pxList, fn);
  const R = arr((p) => p.r);
  const G = arr((p) => p.g);
  const B = arr((p) => p.b);
  const H = arr((p) => p.h360);
  const S = arr((p) => p.s255);
  const V = arr((p) => p.v255);
  const L = arr((p) => p.L);
  const a = arr((p) => p.a);
  const b = arr((p) => p.bLab);

  const stats = (xs) => {
    let sum = 0;
    for (let i = 0; i < xs.length; i++) sum += xs[i];
    const mean = sum / xs.length;
    let m2 = 0, m3 = 0, m4 = 0;
    for (let i = 0; i < xs.length; i++) {
      const d = xs[i] - mean;
      const d2 = d * d;
      m2 += d2;
      m3 += d2 * d;
      m4 += d2 * d2;
    }
    const variance = m2 / Math.max(1, xs.length - 1);
    const std = Math.sqrt(variance);
    const popSd = Math.sqrt(m2 / xs.length) || 1;
    const skew = (m3 / xs.length) / Math.pow(popSd, 3);
    const kurt = (m4 / xs.length) / Math.pow(popSd, 4);
    return { mean, std, skew, kurt };
  };

  const entropy256 = (xs) => {
    const counts = new Uint32Array(256);
    let tot = 0;
    for (let i = 0; i < xs.length; i++) {
      const v = Math.floor(xs[i]);
      if (v >= 0 && v < 256) {
        counts[v]++;
        tot++;
      }
    }
    if (tot === 0) return 0;
    let ent = 0;
    for (let k = 0; k < 256; k++) {
      if (counts[k] > 0) {
        const p = counts[k] / tot;
        ent -= p * Math.log2(p);
      }
    }
    return ent;
  };

  const histBins = (xs, nbins, maxVal) => {
    const counts = new Uint32Array(nbins);
    for (let i = 0; i < xs.length; i++) {
      let idx = Math.floor((xs[i] / maxVal) * nbins);
      if (idx >= nbins) idx = nbins - 1;
      if (idx < 0) idx = 0;
      counts[idx]++;
    }
    return Array.from(counts, (c) => c / xs.length);
  };

  const rS = stats(R), gS = stats(G), bS = stats(B);
  const hS = stats(H), sS = stats(S), vS = stats(V);
  const lS = stats(L), aS = stats(a), blS = stats(b);
  const hBins = histBins(H, 8, 360);
  const sBins = histBins(S, 8, 255);
  const vBins = histBins(V, 5, 255);

  const gFrac = arr((p) => {
    const s = p.r + p.g + p.b;
    return s > 0 ? p.g / s : 0;
  });
  const rbRatio = arr((p) => p.r / (p.b !== 0 ? p.b : 1));
  const gbRatio = arr((p) => p.g / (p.b !== 0 ? p.b : 1));
  const exg = arr((p) => 2 * p.g - p.r - p.b);
  const exr = arr((p) => 1.4 * p.r - p.g);
  const tgi = arr((p) => -0.5 * (190 * (p.r - p.g) - 120 * (p.r - p.b)));
  const exb = arr((p) => 2 * p.b - p.r - p.g);
  const rg = arr((p) => p.r - p.g);
  const yb = arr((p) => 0.5 * (p.r + p.g) - p.b);
  const rgS = stats(rg), ybS = stats(yb);

  const exgS = stats(exg), exrS = stats(exr), tgiS = stats(tgi), exbS = stats(exb);

  return {
    R_skew: rS.skew,
    V_skew: vS.skew,
    R_std: rS.std,
    Entropy_R: entropy256(R),
    G_std: gS.std,
    Entropy_G: entropy256(G),
    B_std: bS.std,
    Entropy_B: entropy256(B),
    H_std: hS.std,
    H_skew: hS.skew,
    H_bin0: hBins[0],
    H_bin1: hBins[1],
    H_bin5: hBins[5],
    S_std: sS.std,
    S_kurt: sS.kurt,
    S_bin0: sBins[0],
    S_bin2: sBins[2],
    S_bin7: sBins[7],
    Entropy_S: entropy256(S),
    V_std: vS.std,
    V_bin0: vBins[0],
    V_bin2: vBins[2],
    Entropy_V: entropy256(V),
    L_std: lS.std,
    a_mean: aS.mean,
    a_std: aS.std,
    b_mean: blS.mean,
    b_std: blS.std,
    g_mean: stats(gFrac).mean,
    R_B_ratio_mean: stats(rbRatio).mean,
    G_B_ratio_mean: stats(gbRatio).mean,
    ExG_mean: exgS.mean,
    ExG_std: exgS.std,
    ExR_std: exrS.std,
    TGI_mean: tgiS.mean,
    TGI_std: tgiS.std,
    ExB_mean: exbS.mean,
    Colorfulness_index:
      Math.hypot(rgS.std, ybS.std) + 0.3 * Math.hypot(rgS.mean, ybS.mean),
  };
}

function formatNum(val) {
  if (val == null || Number.isNaN(val)) return "—";
  const abs = Math.abs(val);
  if (abs >= 100) return val.toFixed(1);
  if (abs >= 10) return val.toFixed(2);
  return val.toFixed(4);
}

document.addEventListener("DOMContentLoaded", initDemo);
