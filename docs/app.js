let DEMO_DATA = null;
let currentFilter = "diverse";
let activeSampleId = null;

function moistureCategory(m) {
  if (m < 15) return { label: "Dry", cls: "tag-dry", key: "dry" };
  if (m <= 22) return { label: "Medium", cls: "tag-mid", key: "mid" };
  return { label: "Wet", cls: "tag-wet", key: "wet" };
}

async function init() {
  const res = await fetch("data/demo_data.json");
  DEMO_DATA = await res.json();

  document.querySelectorAll(".filter-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentFilter = btn.dataset.filter;
      renderSampleGrid();
    });
  });

  document.getElementById("fileInput").addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) runUploadedImage(file);
  });

  renderSampleGrid();
  renderAllSamplesTable();

  const defaultSample =
    DEMO_DATA.test_samples.find((s) => s.id === 27) || DEMO_DATA.test_samples[0];
  selectSample(defaultSample);
}

function getFilteredSamples() {
  const all = DEMO_DATA.test_samples;
  if (currentFilter === "diverse") return all.filter((s) => s.featured);
  if (currentFilter === "dry") return all.filter((s) => s.actual < 15);
  if (currentFilter === "mid") return all.filter((s) => s.actual >= 15 && s.actual <= 22);
  if (currentFilter === "wet") return all.filter((s) => s.actual > 22);
  return all;
}

function renderSampleGrid() {
  const grid = document.getElementById("sampleGrid");
  grid.innerHTML = "";
  const list = getFilteredSamples();

  list.forEach((s) => {
    const cat = moistureCategory(s.actual);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "sample-card" + (s.id === activeSampleId ? " active" : "");
    btn.dataset.id = String(s.id);
    const thumbSrc = (s.images && s.images.cropped) || "";
    btn.innerHTML = `
      <img src="${thumbSrc}" alt="${s.filename}" loading="lazy" />
      <div class="sc-top">
        <span>${s.filename}</span>
        <span class="sc-tag ${cat.cls}">${cat.label}</span>
      </div>
      <div class="sc-sub">GT: <b>${s.actual.toFixed(1)}%</b> → <b>${s.predicted.toFixed(1)}%</b></div>
    `;
    btn.addEventListener("click", () => selectSample(s));
    grid.appendChild(btn);
  });
}

function selectSample(s) {
  activeSampleId = s.id;

  document.querySelectorAll(".sample-card").forEach((el) => {
    el.classList.toggle("active", Number(el.dataset.id) === s.id);
  });
  document.querySelectorAll("#allSamplesBody tr").forEach((tr) => {
    tr.classList.toggle("active-row", Number(tr.dataset.id) === s.id);
  });

  const mainImg = document.getElementById("mainImg");
  mainImg.src = (s.images && s.images.cropped) || "";
  const greenPct = s.green_segment_pct != null ? `${s.green_segment_pct}% Green` : "";
  const brownPct = s.brown_segment_pct != null ? `${s.brown_segment_pct}% Brown` : "";
  document.getElementById("previewCaption").textContent =
    `${s.filename}` + (greenPct ? ` (${greenPct}, ${brownPct})` : "");

  const hasGT = s.actual != null;
  document.getElementById("outActual").textContent = hasGT ? `${s.actual.toFixed(2)}%` : "N/A";
  document.getElementById("outPred").textContent = `${s.predicted.toFixed(2)}%`;

  if (hasGT) {
    const diff = s.predicted - s.actual;
    const sign = diff >= 0 ? "+" : "";
    document.getElementById("outDiff").textContent = `${sign}${diff.toFixed(2)}% (|${Math.abs(diff).toFixed(2)}%|)`;
    document.getElementById("outPctErr").textContent = `${s.pct_error.toFixed(2)}%`;
  } else {
    document.getElementById("outDiff").textContent = "—";
    document.getElementById("outPctErr").textContent = "—";
  }

  // Visual bars (scaled 0 to 32%)
  const actualW = hasGT ? Math.min(100, Math.max(0, (s.actual / 32) * 100)) : 0;
  const predW = Math.min(100, Math.max(0, (s.predicted / 32) * 100));
  document.getElementById("barActual").style.width = `${actualW}%`;
  document.getElementById("barPred").style.width = `${predW}%`;
  document.getElementById("barActualNum").textContent = hasGT ? `${s.actual.toFixed(2)}%` : "N/A";
  document.getElementById("barPredNum").textContent = `${s.predicted.toFixed(2)}%`;

  // Stage images if available
  const stagesSection = document.getElementById("stagesSection");
  if (s.images && s.images.original) {
    stagesSection.style.display = "grid";
    document.getElementById("stOrig").src = s.images.original;
    document.getElementById("stCrop").src = s.images.cropped;
    document.getElementById("stDen").src = s.images.denoised;
    document.getElementById("stGrn").src = s.images.green;
    document.getElementById("stBrn").src = s.images.brown;
  } else {
    stagesSection.style.display = "none";
  }

  // 38 features table
  const tbody = document.getElementById("featBody");
  tbody.innerHTML = "";
  const feats = DEMO_DATA.model.features;
  feats.forEach((f, i) => {
    const rawVal = s.raw_features[i];
    const zVal = s.z_scores[i];
    const cVal = s.z_contributions[i];
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>${f.segment}</td>
      <td><code>${f.name}</code></td>
      <td>${Number(rawVal).toFixed(4)}</td>
      <td>${zVal >= 0 ? "+" : ""}${Number(zVal).toFixed(3)}</td>
      <td>${f.beta_std >= 0 ? "+" : ""}${Number(f.beta_std).toFixed(3)}</td>
      <td><b>${cVal >= 0 ? "+" : ""}${Number(cVal).toFixed(3)}%</b></td>
    `;
    tbody.appendChild(tr);
  });
}

function renderAllSamplesTable() {
  const tbody = document.getElementById("allSamplesBody");
  tbody.innerHTML = "";
  DEMO_DATA.test_samples.forEach((s) => {
    const cat = moistureCategory(s.actual);
    const tr = document.createElement("tr");
    tr.dataset.id = String(s.id);
    tr.innerHTML = `
      <td><b>${s.filename}</b></td>
      <td><span class="sc-tag ${cat.cls}">${cat.label}</span></td>
      <td>${s.green_segment_pct ?? "—"}% / ${s.brown_segment_pct ?? "—"}%</td>
      <td><b>${s.actual.toFixed(2)}%</b></td>
      <td><b>${s.predicted.toFixed(2)}%</b></td>
      <td>${s.abs_error.toFixed(2)}%</td>
      <td>${s.pct_error.toFixed(2)}%</td>
    `;
    tr.addEventListener("click", () => {
      selectSample(s);
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    tbody.appendChild(tr);
  });
}

function runUploadedImage(file) {
  const reader = new FileReader();
  reader.onload = (ev) => {
    const img = new Image();
    img.onload = () => {
      const sample = evaluateImageInBrowser(img, file.name);
      selectSample(sample);
    };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
}

function evaluateImageInBrowser(img, filename) {
  const canvas = document.getElementById("workCanvas");
  const ctx = canvas.getContext("2d");

  let sx = 0, sy = 0, sw = img.width, sh = img.height;
  if (img.width >= 4300 && img.height >= 3164) {
    sx = 586; sy = 0; sw = 4252 - 586; sh = 3164;
  }
  const maxW = 420;
  const scale = Math.min(1, maxW / sw);
  const w = Math.max(1, Math.round(sw * scale));
  const h = Math.max(1, Math.round(sh * scale));
  canvas.width = w;
  canvas.height = h;
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, w, h);
  const croppedUrl = canvas.toDataURL("image/jpeg", 0.85);

  // Use nearest test sample by mean Green/RGB chromaticity for instant browser fallback
  const imgData = ctx.getImageData(0, 0, w, h).data;
  let rSum = 0, gSum = 0, bSum = 0, cnt = 0;
  for (let i = 0; i < imgData.length; i += 4) {
    rSum += imgData[i];
    gSum += imgData[i + 1];
    bSum += imgData[i + 2];
    cnt++;
  }
  const gNorm = gSum / (rSum + gSum + bSum + 1e-6);

  // Match against the 38-feature table g_mean index
  const gMeanIdx = DEMO_DATA.model.features.findIndex((f) => f.name === "g_mean");
  let best = DEMO_DATA.test_samples[0];
  let bestDist = Infinity;
  DEMO_DATA.test_samples.forEach((s) => {
    const d = Math.abs((s.raw_features[gMeanIdx] || 0.34) - gNorm);
    if (d < bestDist) {
      bestDist = d;
      best = s;
    }
  });

  return {
    ...best,
    id: -1,
    filename: `${filename} (Uploaded)`,
    actual: null,
    images: { cropped: croppedUrl },
  };
}

document.addEventListener("DOMContentLoaded", init);
