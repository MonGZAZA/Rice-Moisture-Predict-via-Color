let DATA = null;
let currentFilter = "diverse";
let activeSample = null;
let activeStageKey = "cropped";

async function init() {
  if (window.DEMO_DATA) {
    DATA = window.DEMO_DATA;
  } else {
    const res = await fetch("data/demo_data.json");
    DATA = await res.json();
  }

  const filterSelect = document.getElementById("filterSelect");
  filterSelect.addEventListener("change", () => {
    currentFilter = filterSelect.value;
    renderSampleList();
  });

  document.getElementById("fileInput").addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) runUploadedImage(file);
  });

  renderSampleList();
  renderAllTable();

  const initial = DATA.test_samples.find((s) => s.id === 27) || DATA.test_samples[0];
  selectSample(initial);
}

function getFilteredSamples() {
  const all = DATA.test_samples;
  if (currentFilter === "diverse") return all.filter((s) => s.featured);
  if (currentFilter === "dry") return all.filter((s) => s.actual < 15);
  if (currentFilter === "mid") return all.filter((s) => s.actual >= 15 && s.actual <= 22);
  if (currentFilter === "wet") return all.filter((s) => s.actual > 22);
  return all;
}

function renderSampleList() {
  const listEl = document.getElementById("sampleList");
  listEl.innerHTML = "";
  const items = getFilteredSamples();
  document.getElementById("sampleCountLabel").textContent = `${items.length} shown`;

  items.forEach((s) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "sample-item" + (activeSample && activeSample.id === s.id ? " active" : "");
    btn.dataset.id = String(s.id);
    btn.innerHTML = `
      <img src="${s.images.cropped}" alt="${s.filename}" loading="lazy" />
      <div class="si-name">${s.filename}</div>
      <div class="si-gt">GT: ${s.actual.toFixed(1)}% | Out: ${s.predicted.toFixed(1)}%</div>
    `;
    btn.addEventListener("click", () => selectSample(s));
    listEl.appendChild(btn);
  });
}

function selectSample(s) {
  activeSample = s;
  activeStageKey = s.images && s.images.original ? "original" : "cropped";

  document.querySelectorAll(".sample-item").forEach((el) => {
    el.classList.toggle("active", Number(el.dataset.id) === s.id);
  });
  document.querySelectorAll("#allTableBody tr").forEach((tr) => {
    tr.classList.toggle("active-row", Number(tr.dataset.id) === s.id);
  });

  document.getElementById("activeFilename").textContent = s.filename;
  renderImagePane(s);

  document.getElementById("resFile").textContent = s.filename;
  const hasGT = s.actual != null;
  document.getElementById("resActual").textContent = hasGT ? `${s.actual.toFixed(2)} %` : "N/A";
  document.getElementById("resPred").textContent = `${s.predicted.toFixed(2)} %`;

  if (hasGT) {
    const signed = s.predicted - s.actual;
    const sign = signed >= 0 ? "+" : "";
    document.getElementById("resSignedErr").textContent = `${sign}${signed.toFixed(2)} %`;
    document.getElementById("resAbsErr").textContent = `${s.abs_error.toFixed(2)} %`;
    document.getElementById("resRelErr").textContent = `${s.pct_error.toFixed(2)} %`;
  } else {
    document.getElementById("resSignedErr").textContent = "—";
    document.getElementById("resAbsErr").textContent = "—";
    document.getElementById("resRelErr").textContent = "—";
  }

  document.getElementById("resMasks").textContent =
    s.green_segment_pct != null
      ? `Green ${s.green_segment_pct}% / Brown ${s.brown_segment_pct}%`
      : "—";

  renderFeaturesTable(s);
}

function renderImagePane(s) {
  const img = document.getElementById("previewImg");
  const btnWrap = document.getElementById("viewButtons");
  btnWrap.innerHTML = "";

  const stageDefs = [
    ["original", "Original"],
    ["cropped", "Cropped"],
    ["denoised", "Denoised"],
    ["green", "Green Mask"],
    ["brown", "Brown Mask"],
  ];

  const available = stageDefs.filter(([k]) => Boolean(s.images && s.images[k]));
  if (!s.images || !s.images[activeStageKey]) {
    activeStageKey = available.length ? available[0][0] : "cropped";
  }

  img.src = (s.images && s.images[activeStageKey]) || "";

  if (available.length > 1) {
    available.forEach(([key, label]) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "view-btn" + (key === activeStageKey ? " active" : "");
      b.textContent = label;
      b.addEventListener("click", () => {
        activeStageKey = key;
        renderImagePane(s);
      });
      btnWrap.appendChild(b);
    });
  }
}

function renderFeaturesTable(s) {
  const tbody = document.getElementById("featTableBody");
  tbody.innerHTML = "";

  DATA.model.features.forEach((f, i) => {
    const raw = s.raw_features[i];
    const z = s.z_scores[i];
    const contrib = s.z_contributions[i];
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>${f.segment}</td>
      <td><code>${f.name}</code></td>
      <td>${Number(raw).toFixed(4)}</td>
      <td>${Number(f.train_mean).toFixed(4)}</td>
      <td>${Number(f.train_std).toFixed(4)}</td>
      <td>${z >= 0 ? "+" : ""}${Number(z).toFixed(3)}</td>
      <td>${f.beta_std >= 0 ? "+" : ""}${Number(f.beta_std).toFixed(4)}</td>
      <td>${contrib >= 0 ? "+" : ""}${Number(contrib).toFixed(4)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderAllTable() {
  const tbody = document.getElementById("allTableBody");
  tbody.innerHTML = "";

  DATA.test_samples.forEach((s) => {
    const tr = document.createElement("tr");
    tr.dataset.id = String(s.id);
    tr.innerHTML = `
      <td>${s.filename}</td>
      <td>${s.actual.toFixed(2)}</td>
      <td>${s.predicted.toFixed(2)}</td>
      <td>${s.abs_error.toFixed(2)}</td>
      <td>${s.pct_error.toFixed(2)}%</td>
      <td>${s.green_segment_pct}% / ${s.brown_segment_pct}%</td>
    `;
    tr.addEventListener("click", () => selectSample(s));
    tbody.appendChild(tr);
  });
}

/* Run exact 38-feature HSV + CIELab + OLS model in-browser for uploaded images */
function runUploadedImage(file) {
  const reader = new FileReader();
  reader.onload = (ev) => {
    const img = new Image();
    img.onload = () => {
      const sample = analyzeUploadedImage(img, file.name);
      selectSample(sample);
    };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
}

function analyzeUploadedImage(img, filename) {
  const canvas = document.getElementById("workCanvas");
  const ctx = canvas.getContext("2d");

  const origUrl = scaleToDataUrl(img, 0, 0, img.width, img.height, 480);

  let sx = 0, sy = 0, sw = img.width, sh = img.height;
  if (img.width >= 4300 && img.height >= 3164) {
    sx = 586; sy = 0; sw = 4252 - 586; sh = 3164;
  }
  const scale = Math.min(1, 420 / sw);
  const w = Math.max(1, Math.round(sw * scale));
  const h = Math.max(1, Math.round(sh * scale));
  canvas.width = w;
  canvas.height = h;
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, w, h);
  const croppedUrl = canvas.toDataURL("image/jpeg", 0.85);

  const px = ctx.getImageData(0, 0, w, h).data;
  const denImg = ctx.createImageData(w, h);
  const grnImg = ctx.createImageData(w, h);
  const brnImg = ctx.createImageData(w, h);

  let validCount = 0, grnCount = 0, brnCount = 0;
  const allPix = { R: [], G: [], B: [], H: [], S: [], V: [], L: [], a: [], b: [] };
  const grnPix = { R: [], G: [], B: [], H: [], S: [], V: [], L: [], a: [], b: [] };

  for (let i = 0; i < px.length; i += 4) {
    const r = px[i], g = px[i + 1], b = px[i + 2];
    const [H, S, V] = rgbToOpenCvHsv(r, g, b);
    const notBlack = !(r === 0 && g === 0 && b === 0);
    const notBlue = H < 100 || H > 140;
    const isValid = notBlack && notBlue;

    denImg.data[i + 3] = 255;
    grnImg.data[i + 3] = 255;
    brnImg.data[i + 3] = 255;

    if (isValid) {
      validCount++;
      denImg.data[i] = r; denImg.data[i + 1] = g; denImg.data[i + 2] = b;
      const [L, a, bb] = rgbToOpenCvLab(r, g, b);
      const hDeg = H * 2.0;

      allPix.R.push(r); allPix.G.push(g); allPix.B.push(b);
      allPix.H.push(hDeg); allPix.S.push(S); allPix.V.push(V);
      allPix.L.push(L); allPix.a.push(a); allPix.b.push(bb);

      // config.py: GREEN_HSV_LOWER = (25, 35, 35), GREEN_HSV_UPPER = (85, 255, 255)
      if (H >= 25 && H <= 85 && S >= 35 && V >= 35) {
        grnCount++;
        grnImg.data[i] = r; grnImg.data[i + 1] = g; grnImg.data[i + 2] = b;
        grnPix.R.push(r); grnPix.G.push(g); grnPix.B.push(b);
        grnPix.H.push(hDeg); grnPix.S.push(S); grnPix.V.push(V);
        grnPix.L.push(L); grnPix.a.push(a); grnPix.b.push(bb);
      } else {
        brnCount++;
        brnImg.data[i] = r; brnImg.data[i + 1] = g; brnImg.data[i + 2] = b;
      }
    }
  }

  ctx.putImageData(denImg, 0, 0);
  const denoisedUrl = canvas.toDataURL("image/jpeg", 0.85);
  ctx.putImageData(grnImg, 0, 0);
  const greenUrl = canvas.toDataURL("image/jpeg", 0.85);
  ctx.putImageData(brnImg, 0, 0);
  const brownUrl = canvas.toDataURL("image/jpeg", 0.85);

  const allStats = extractPaperFeatures(allPix);
  const grnStats = extractPaperFeatures(grnPix.R.length > 10 ? grnPix : allPix);

  const rawFeatures = [];
  const zScores = [];
  const contribs = [];
  let yPred = DATA.model.intercept_std;

  DATA.model.features.forEach((f) => {
    const src = f.segment === "All" ? allStats : grnStats;
    const val = src[f.name] ?? f.train_mean;
    const z = (val - f.train_mean) / f.train_std;
    const c = z * f.beta_std;
    rawFeatures.push(val);
    zScores.push(z);
    contribs.push(c);
    yPred += c;
  });

  const denom = Math.max(1, validCount);
  return {
    id: -1,
    filename: `${filename} (uploaded)`,
    actual: null,
    predicted: yPred,
    abs_error: null,
    pct_error: null,
    featured: false,
    green_segment_pct: Number(((grnCount / denom) * 100).toFixed(2)),
    brown_segment_pct: Number(((brnCount / denom) * 100).toFixed(2)),
    images: {
      original: origUrl,
      cropped: croppedUrl,
      denoised: denoisedUrl,
      green: greenUrl,
      brown: brownUrl,
    },
    raw_features: rawFeatures,
    z_scores: zScores,
    z_contributions: contribs,
  };
}

function scaleToDataUrl(img, sx, sy, sw, sh, maxW) {
  const c = document.createElement("canvas");
  const s = Math.min(1, maxW / sw);
  c.width = Math.max(1, Math.round(sw * s));
  c.height = Math.max(1, Math.round(sh * s));
  c.getContext("2d").drawImage(img, sx, sy, sw, sh, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.85);
}

function rgbToOpenCvHsv(r, g, b) {
  const rf = r / 255, gf = g / 255, bf = b / 255;
  const max = Math.max(rf, gf, bf), min = Math.min(rf, gf, bf);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === rf) h = 60 * (((gf - bf) / d) % 6);
    else if (max === gf) h = 60 * ((bf - rf) / d + 2);
    else h = 60 * ((rf - gf) / d + 4);
  }
  if (h < 0) h += 360;
  return [Math.round(h / 2), Math.round(max === 0 ? 0 : (d / max) * 255), Math.round(max * 255)];
}

function rgbToOpenCvLab(r, g, b) {
  const lin = (c) => {
    const v = c / 255;
    return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  };
  const R = lin(r), G = lin(g), B = lin(b);
  const X = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.95047;
  const Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B;
  const Z = (0.0193339 * R + 0.1191920 * G + 0.9503041 * B) / 1.08883;
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f(X), fy = f(Y), fz = f(Z);
  return [((116 * fy - 16) * 255) / 100, 500 * (fx - fy) + 128, 200 * (fy - fz) + 128];
}

function extractPaperFeatures(pix) {
  const n = pix.R.length;
  if (!n) return {};
  const stats = (arr) => {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += arr[i];
    const m = sum / n;
    let sq = 0, m3 = 0, m4 = 0;
    for (let i = 0; i < n; i++) {
      const d = arr[i] - m;
      sq += d * d;
      m3 += d * d * d;
      m4 += d * d * d * d;
    }
    const sd = n > 1 ? Math.sqrt(sq / (n - 1)) : 0;
    const sigma = Math.sqrt(sq / n);
    const sk = sigma > 1e-9 ? m3 / n / Math.pow(sigma, 3) : 0;
    const ku = sigma > 1e-9 ? m4 / n / Math.pow(sigma, 4) : 0;
    return [m, sd, sk, ku];
  };
  const histBins = (arr, k, rangeMax) => {
    const h = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      let b = Math.floor((arr[i] / rangeMax) * k);
      if (b >= k) b = k - 1;
      if (b < 0) b = 0;
      h[b]++;
    }
    return h.map((c) => c / n);
  };
  const entropy = (arr, rangeMax) => {
    const h = new Array(256).fill(0);
    for (let i = 0; i < n; i++) {
      let b = Math.floor((arr[i] / rangeMax) * 256);
      if (b >= 256) b = 255;
      if (b < 0) b = 0;
      h[b]++;
    }
    let e = 0;
    for (let i = 0; i < 256; i++) {
      if (h[i] > 0) {
        const p = h[i] / n;
        e -= p * Math.log2(p);
      }
    }
    return e;
  };

  const f = {};
  for (const [name, arr] of [["R", pix.R], ["G", pix.G], ["B", pix.B]]) {
    const [m, sd, sk] = stats(arr);
    f[`${name}_mean`] = m;
    f[`${name}_std`] = sd;
    f[`${name}_skew`] = sk;
    f[`Entropy_${name}`] = entropy(arr, 256);
  }
  for (const [name, arr, rng, k, erng] of [
    ["H", pix.H, 360, 8, 360],
    ["S", pix.S, 255, 8, 256],
    ["V", pix.V, 255, 5, 256],
  ]) {
    const [, sd, sk, ku] = stats(arr);
    f[`${name}_std`] = sd;
    f[`${name}_skew`] = sk;
    f[`${name}_kurt`] = ku;
    histBins(arr, k, rng).forEach((v, idx) => {
      f[`${name}_bin${idx}`] = v;
    });
    f[`Entropy_${name}`] = entropy(arr, erng);
  }
  for (const [name, arr] of [["L", pix.L], ["a", pix.a], ["b", pix.b]]) {
    const [m, sd] = stats(arr);
    f[`${name}_mean`] = m;
    f[`${name}_std`] = sd;
  }

  let gFracSum = 0, rbSum = 0, gbSum = 0;
  const exgArr = new Float64Array(n);
  const exrArr = new Float64Array(n);
  const tgiArr = new Float64Array(n);
  const exbArr = new Float64Array(n);
  const rgArr = new Float64Array(n);
  const ybArr = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const R = pix.R[i], G = pix.G[i], B = pix.B[i];
    const s = R + G + B;
    gFracSum += s > 0 ? G / s : 0;
    rbSum += R / (B !== 0 ? B : 1);
    gbSum += G / (B !== 0 ? B : 1);
    exgArr[i] = 2 * G - R - B;
    exrArr[i] = 1.4 * R - G;
    tgiArr[i] = -0.5 * (190 * (R - G) - 120 * (R - B));
    exbArr[i] = 2 * B - R - G;
    rgArr[i] = R - G;
    ybArr[i] = 0.5 * (R + G) - B;
  }

  f.g_mean = gFracSum / n;
  f.R_B_ratio_mean = rbSum / n;
  f.G_B_ratio_mean = gbSum / n;

  const [exgM, exgS] = stats(exgArr);
  const [, exrS] = stats(exrArr);
  const [tgiM, tgiS] = stats(tgiArr);
  const [exbM] = stats(exbArr);
  f.ExG_mean = exgM;
  f.ExG_std = exgS;
  f.ExR_std = exrS;
  f.TGI_mean = tgiM;
  f.TGI_std = tgiS;
  f.ExB_mean = exbM;

  const [rgM, rgS] = stats(rgArr);
  const [ybM, ybS] = stats(ybArr);
  f.Colorfulness_index = Math.sqrt(rgS * rgS + ybS * ybS) + 0.3 * Math.sqrt(rgM * rgM + ybM * ybM);
  return f;
}

document.addEventListener("DOMContentLoaded", init);
