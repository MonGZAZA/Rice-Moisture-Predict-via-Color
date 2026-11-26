/* Install Playwright separately, then run 'node tests/demo-upload.cjs' from the repo root.
 * Set OPENCV_JS to a local copy of the pinned dependency to test offline.
 * Set CHROMIUM_EXECUTABLE for a custom Chromium binary. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require("playwright");

const root = path.resolve(__dirname, "..");
const photos = process.env.RICE_PHOTO_ROOT || root;
const origin = "http://rice-test.local";
const cvUrl = "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.11.0-release.1/dist/opencv.js";
const types = {".html":"text/html", ".css":"text/css", ".js":"text/javascript"};
const references = [
  ["original", "0245.jpg", 13.480347161414777],
  ["original", "0069.jpg", 16.80371853897718],
  ["original", "0146.jpg", 24.222161489852407],
  ["later", "0297.jpg", 29.091419685321263],
  ["later", "0320.jpg", 18.250640956204297],
  ["later", "0334.jpg", 17.01492892729086],
  ["later", "0356.jpg", 15.509665282879988],
  ["later", "0363.jpg", 30.41029320363389],
  ["later", "0392.jpg", 14.228286563646487],
  ["later", "0412.jpg", 16.825588647905757],
  ["later", "0433.jpg", 27.966479321071546],
  ["later", "0445.jpg", 27.34963628526993],
  ["later", "0456.jpg", 15.955963053677163],
  ["later", "0469.jpg", 22.570338144940575],
  ["later", "0482.jpg", 18.340482036043984],
  ["later", "0494.jpg", 15.956908655632038],
  ["later", "0502.jpg", 16.798649915893616],
  ["later", "0512.jpg", 14.46992321221435],
];
const TOLERANCE_PP = 0.10;

async function run() {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_EXECUTABLE,
    args: ["--no-sandbox"],
  });
  try {
    const page = await browser.newPage({viewport:{width:390,height:844}});
    const errors = [];
    page.on("pageerror", error => errors.push(String(error)));
    await page.route("**/*", async route => {
      const u = new URL(route.request().url());
      if (u.href === cvUrl) {
        if (!process.env.OPENCV_JS) return route.continue();
        return route.fulfill({path:process.env.OPENCV_JS, contentType:"text/javascript", headers:{"Access-Control-Allow-Origin":"*"}});
      }
      if (u.origin !== origin) return route.abort();
      const f = u.pathname === "/" ? "index.html" : u.pathname.slice(1);
      if (!["index.html","app.js","demo_data.js","styles.css","upload.js"].includes(f)) return route.fulfill({status:404});
      return route.fulfill({path:path.join(root,"docs",f), contentType:types[path.extname(f)]});
    });
    await page.goto(origin);
    await page.locator("#filterSelect").selectOption("all");
    await page.locator(".sample-item[data-id='245']").click();
    assert.equal(await page.locator("#resPred").textContent(),"13.93 %");
    assert.equal(await page.locator(".sample-item").count(),45);
    let maxDifference=0;
    for (const [location, filename, reference] of references) {
      const file = location==="original"
        ? path.join(photos,process.env.RICE_PHOTO_ROOT?"original":"data/pictures",filename)
        : path.join(photos,process.env.RICE_PHOTO_ROOT?"later":"data/extra_dataset/pictures",filename);
      await page.locator("#fileInput").setInputFiles(file);
      await page.waitForFunction(()=>!document.getElementById("fileInput").disabled,null,{timeout:60000});
      assert.equal(await page.locator("#uploadStatus").textContent(),"Lightbox estimate ready.",filename);
      assert.equal(await page.locator("#resFile").textContent(),filename+" (uploaded)");
      const prediction=await page.evaluate(()=>activeSample.predicted);
      const delta=Math.abs(prediction-reference);
      maxDifference=Math.max(maxDifference,delta);
      assert.ok(delta<=TOLERANCE_PP,`${filename}: ${prediction} vs. Python ${reference}`);
      console.log(`${filename}: browser ${prediction.toFixed(4)}; Python ${reference.toFixed(4)}`);
    }
    const lastEstimate=await page.locator("#resPred").textContent();
    for (const [w,h,msg] of [[10,10,/full-size/],[4896,3264,/No usable rice pixels/]]) {
      const buf=await page.evaluate(async ([w,h])=>{
        const c=document.createElement("canvas"); c.width=w; c.height=h;
        c.getContext("2d").fillRect(0,0,w,h);
        const blob=await new Promise(r=>c.toBlob(r,"image/jpeg"));
        return Array.from(new Uint8Array(await blob.arrayBuffer()));
      },[w,h]);
      await page.locator("#fileInput").setInputFiles({name:"black.jpg",mimeType:"image/jpeg",buffer:Buffer.from(buf)});
      await page.waitForFunction(()=>!document.getElementById("fileInput").disabled);
      assert.match(await page.locator("#uploadStatus").textContent(),msg);
      assert.equal(await page.locator("#resPred").textContent(),lastEstimate);
    }
    for (const [buffer, msg] of [[Buffer.from("not an image"),/original JPEG/],[Buffer.from([255,216,255,0]),/could not be read/]]) {
      await page.locator("#fileInput").setInputFiles({name:"invalid.jpg",mimeType:"image/jpeg",buffer});
      await page.waitForFunction(()=>!document.getElementById("fileInput").disabled);
      assert.match(await page.locator("#uploadStatus").textContent(),msg);
      assert.equal(await page.locator("#resPred").textContent(),lastEstimate);
    }
    assert.deepEqual(errors,[]);
    console.log(`PASS: ${references.length} real photos, max Python delta ${maxDifference.toFixed(4)} pp; four invalid inputs and built-in samples.`);
  } finally { await browser.close(); }
}
run().catch(error=>{console.error(error);process.exitCode=1;});
