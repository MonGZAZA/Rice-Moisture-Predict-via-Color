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

async function runUploadedImage(file) {
  const input = document.getElementById("fileInput");
  const status = document.getElementById("uploadStatus");
  input.disabled = true;
  status.textContent = "Analyzing the full-size photo…";
  try {
    selectSample(await RiceUpload.analyze(file, DATA.model));
    status.textContent = "Lightbox estimate ready.";
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Could not analyze this image.";
  } finally {
    input.value = "";
    input.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", init);
