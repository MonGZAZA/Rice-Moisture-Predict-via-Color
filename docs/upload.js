/* Full-resolution lab inference. Keep in sync with code/predict.py and
 * the jpeg_compat branch of code/01_preprocess_and_extract.py. */
const RiceUpload = (() => {
  const WIDTH = 4896, HEIGHT = 3264;
  const CROP = [586, 0, 3666, 3164];
  const MIN_GREEN = 32;
  const MAX_FILE_SIZE = 24 * 1024 * 1024;
  let cvLoad;

  function getCv() {
    if (!cvLoad) cvLoad = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = CV_URL;
      script.integrity = CV_INTEGRITY;
      script.crossOrigin = "anonymous";
      const failed = () => {
        clearTimeout(timer);
        script.remove();
        reject(new Error("Could not load the image analyzer. Try again or select a built-in sample."));
      };
      const timer = setTimeout(failed, 45000);
      script.onerror = failed;
      script.onload = async () => {
        try {
          const cv = await window.cv;
          if (!cv?.Mat) throw new Error("OpenCV not ready");
          clearTimeout(timer);
          resolve(cv);
        } catch { failed(); }
      };
      document.head.appendChild(script);
    }).catch(error => { cvLoad = undefined; throw error; });
    return cvLoad;
  }

  function canvas(width, height) {
    const c = document.createElement("canvas");
    c.width = width; c.height = height;
    return c;
  }

  function draw(img, sx, sy, sw, sh, maxWidth = sw) {
    const w = Math.min(sw, maxWidth);
    const out = canvas(w, Math.max(1, Math.round(sh * w / sw)));
    out.getContext("2d", {willReadFrequently: true}).drawImage(img, sx, sy, sw, sh, 0, 0, out.width, out.height);
    return out;
  }

  function preview(img, sx = 0, sy = 0, sw = img.width, sh = img.height) {
    const c = draw(img, sx, sy, sw, sh, 480);
    try { return c.toDataURL("image/jpeg", 0.85); }
    finally { c.width = c.height = 0; }
  }

  function pixels(c) {
    return c.getContext("2d").getImageData(0, 0, c.width, c.height);
  }

  function roundTrip(c, quality) {
    return new Promise((resolve, reject) => c.toBlob(async blob => {
      if (!blob) { reject(new Error("Could not process the JPEG image.")); return; }
      try {
        const img = await createImageBitmap(blob, {colorSpaceConversion: "none"});
        try { resolve(draw(img, 0, 0, img.width, img.height)); }
        finally { img.close(); }
      } catch (error) { reject(error); }
    }, "image/jpeg", quality / 100));
  }

  function colorSpace(cv, rgba, conversion) {
    const src = cv.matFromImageData(rgba), bgr = new cv.Mat(), out = new cv.Mat();
    try {
      cv.cvtColor(src, bgr, cv.COLOR_RGBA2BGR);
      cv.cvtColor(bgr, out, conversion);
      return out;
    } catch (error) { out.delete(); throw error; }
    finally { src.delete(); bgr.delete(); }
  }

  function stats(hist, n) {
    let sum = 0, variance = 0, skew = 0, kurt = 0, entropy = 0;
    for (let i = 0; i < hist.length; i++) sum += i * hist[i];
    const mean = sum / n;
    for (let i = 0; i < hist.length; i++) {
      const count = hist[i];
      if (!count) continue;
      const d = i - mean, q = d * d, p = count / n;
      variance += count * q; skew += count * q * d; kurt += count * q * q;
      entropy -= p * Math.log2(p);
    }
    const v = variance / n;
    return {mean, std: Math.sqrt(variance / (n - 1)), skew: skew / n / v ** 1.5, kurt: kurt / n / v ** 2, entropy};
  }

  function bins(hist, n, count, max, multiplier = 1) {
    const b = new Uint32Array(count);
    for (let i = 0; i < hist.length; i++) b[Math.min(count - 1, Math.floor(i * multiplier * count / max))] += hist[i];
    return Array.from(b, value => value / n);
  }

  function allFeatures(rgba) {
    const rHist = new Uint32Array(256), vHist = new Uint32Array(256);
    let n = 0;
    for (let i = 0; i < rgba.length; i += 4) {
      const r = rgba[i], g = rgba[i + 1], b = rgba[i + 2];
      if (!(r | g | b)) continue;
      n++; rHist[r]++; vHist[Math.max(r, g, b)]++;
    }
    if (n < MIN_GREEN) throw new Error("No usable rice pixels were found.");
    return {R_skew: stats(rHist, n).skew, V_skew: stats(vHist, n).skew};
  }

  function greenFeatures(cv, image) {
    const rgba = image.data;
    const hsv = colorSpace(cv, image, cv.COLOR_BGR2HSV);
    let lab;
    try {
      lab = colorSpace(cv, image, cv.COLOR_BGR2Lab);
      const hh = hsv.data, ll = lab.data;
      const names = ["R", "G", "B", "H", "S", "V", "L", "a", "b"];
      const hist = Object.fromEntries(names.map(key => [key, new Uint32Array(256)]));
      let n = 0, gSum = 0, rbSum = 0, gbSum = 0;
      let exg = 0, exg2 = 0, exr = 0, exr2 = 0, exb = 0, tgi = 0, tgi2 = 0;
      let rg = 0, rg2 = 0, yb = 0, yb2 = 0;
      for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
        const r = rgba[i], g = rgba[i + 1], b = rgba[i + 2];
        if (!(r | g | b)) continue;
        n++;
        hist.R[r]++; hist.G[g]++; hist.B[b]++;
        hist.H[hh[j]]++; hist.S[hh[j + 1]]++; hist.V[hh[j + 2]]++;
        hist.L[ll[j]]++; hist.a[ll[j + 1]]++; hist.b[ll[j + 2]]++;
        gSum += g / (r + g + b); rbSum += r / (b || 1); gbSum += g / (b || 1);
        const eG = 2 * g - r - b, eR = 1.4 * r - g, eB = 2 * b - r - g;
        const t = -0.5 * (190 * (r - g) - 120 * (r - b));
        const x = r - g, y = 0.5 * (r + g) - b;
        exg += eG; exg2 += eG * eG; exr += eR; exr2 += eR * eR;
        exb += eB; tgi += t; tgi2 += t * t;
        rg += x; rg2 += x * x; yb += y; yb2 += y * y;
      }
      if (n < MIN_GREEN) throw new Error("Not enough usable green pixels were found.");
      const std = (sum, sum2) => Math.sqrt(Math.max(0, (sum2 - sum * sum / n) / (n - 1)));
      const features = {
        g_mean: gSum / n, R_B_ratio_mean: rbSum / n, G_B_ratio_mean: gbSum / n,
        ExG_mean: exg / n, ExG_std: std(exg, exg2), ExR_std: std(exr, exr2), ExB_mean: exb / n,
        TGI_mean: tgi / n, TGI_std: std(tgi, tgi2),
        Colorfulness_index: Math.hypot(std(rg, rg2), std(yb, yb2)) + 0.3 * Math.hypot(rg / n, yb / n),
      };
      for (const name of names) {
        const s = stats(hist[name], n);
        features[name + "_mean"] = s.mean;
        features[name + "_std"] = s.std * (name === "H" ? 2 : 1);
        features[name + "_skew"] = s.skew;
        features[name + "_kurt"] = s.kurt;
        features["Entropy_" + name] = s.entropy;
      }
      for (const [name, count, max, mult] of [["H", 8, 360, 2], ["S", 8, 255, 1], ["V", 5, 255, 1]]) {
        bins(hist[name], n, count, max, mult).forEach((value, k) => { features[name + "_bin" + k] = value; });
      }
      return features;
    } finally { hsv.delete(); lab?.delete(); }
  }

  function denoise(cv, c) {
    const d = pixels(c);
    const h = colorSpace(cv, d, cv.COLOR_BGR2HSV);
    try {
      const data = h.data;
      for (let i = 0, j = 0; i < d.data.length; i += 4, j += 3) {
        if (data[j] >= 100 && data[j] <= 140) d.data[i] = d.data[i + 1] = d.data[i + 2] = 0;
      }
      c.getContext("2d").putImageData(d, 0, 0);
    } finally { h.delete(); }
  }

  function split(cv, c, image) {
    const h = colorSpace(cv, image, cv.COLOR_BGR2HSV), values = h.data;
    const {width: w, height: height} = c;
    const mask = new Uint8Array(w * height);
    let minX = w, minY = height, maxX = -1, maxY = -1, green = 0, valid = 0;
    try {
      for (let y = 0, k = 0; y < height; y++) {
        for (let x = 0; x < w; x++, k++) {
          const i = k * 4, j = k * 3;
          if (!(image.data[i] | image.data[i + 1] | image.data[i + 2])) continue;
          valid++;
          if (values[j] < 25 || values[j] > 85 || values[j + 1] < 35 || values[j + 2] < 35) continue;
          mask[k] = 1; green++;
          minX = Math.min(minX, x); maxX = Math.max(maxX, x);
          minY = Math.min(minY, y); maxY = Math.max(maxY, y);
        }
      }
    } finally { h.delete(); }
    if (green < MIN_GREEN) throw new Error("Not enough usable green pixels were found.");
    const cropped = canvas(maxX - minX + 1, maxY - minY + 1);
    const ctx = cropped.getContext("2d");
    const out = ctx.createImageData(cropped.width, cropped.height);
    const d = image.data;
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const k = y * w + x, i = 4 * k, j = 4 * ((y - minY) * cropped.width + x - minX);
        out.data[j + 3] = 255;
        if (!mask[k]) continue;
        out.data[j] = d[i]; out.data[j + 1] = d[i + 1]; out.data[j + 2] = d[i + 2];
        d[i] = d[i + 1] = d[i + 2] = 0;
      }
    }
    ctx.putImageData(out, 0, 0);
    // The remaining pixels are only needed for the Brown preview.
    c.getContext("2d").putImageData(image, 0, 0);
    return {cropped, green, valid};
  }

  async function analyze(file, model) {
    if (file.size > MAX_FILE_SIZE) throw new Error("Choose a lightbox JPEG under 24 MB.");
    const header = new Uint8Array(await file.slice(0, 3).arrayBuffer());
    if (header[0] !== 255 || header[1] !== 216 || header[2] !== 255) throw new Error("Choose an original JPEG from the lightbox camera.");
    let img;
    try { img = await createImageBitmap(file, {colorSpaceConversion: "none"}); }
    catch { throw new Error("That JPEG could not be read."); }
    const canvases = [];
    const keep = c => { canvases.push(c); return c; };
    try {
      if (img.width !== WIDTH || img.height !== HEIGHT) throw new Error("Use a full-size 4896 × 3264 JPEG from the lightbox setup.");
      const cv = await getCv();
      const original = preview(img);
      const c = keep(draw(img, ...CROP));
      const cropped = preview(c);
      const first = keep(await roundTrip(c, 75));
      c.width = c.height = 0;
      denoise(cv, first);
      const second = keep(await roundTrip(first, 95));
      first.width = first.height = 0;
      const denoised = preview(second);
      const denPixels = pixels(second);
      const all = allFeatures(denPixels.data);
      const parts = split(cv, second, denPixels);
      keep(parts.cropped);
      const brown = preview(second);
      second.width = second.height = 0;
      const greenCanvas = keep(await roundTrip(parts.cropped, 95));
      parts.cropped.width = parts.cropped.height = 0;
      const green = preview(greenCanvas);
      const greenStats = greenFeatures(cv, pixels(greenCanvas));
      const raw_features = [], z_scores = [], z_contributions = [];
      let predicted = model.intercept_std;
      for (const f of model.features) {
        const value = (f.segment === "All" ? all : greenStats)[f.name];
        const z = (value - f.train_mean) / f.train_std;
        const contribution = z * f.beta_std;
        if (!Number.isFinite(contribution)) throw new Error("This image did not produce valid model inputs.");
        raw_features.push(value); z_scores.push(z); z_contributions.push(contribution);
        predicted += contribution;
      }
      if (!Number.isFinite(predicted) || predicted < 0 || predicted > 100) throw new Error("This photo produced an invalid moisture estimate. Use a photo from the original lightbox setup.");
      return {
        id: -1, filename: file.name + " (uploaded)", actual: null, predicted,
        green_segment_pct: Number((100 * parts.green / parts.valid).toFixed(2)),
        brown_segment_pct: Number((100 * (parts.valid - parts.green) / parts.valid).toFixed(2)),
        images: {original, cropped, denoised, green, brown}, raw_features, z_scores, z_contributions,
      };
    } finally {
      for (const c of canvases) c.width = c.height = 0;
      img.close();
    }
  }

  const CV_URL = "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.11.0-release.1/dist/opencv.js";
  const CV_INTEGRITY = "sha384-oavEdO4ZHMHZLyB8LRX8Vmsn9BuESS2RmefVDvsZOcvAlEtuRiiUj4CZGMEL1fPU";
  return {analyze};
})();
