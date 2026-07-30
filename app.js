// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const MODEL_URL = 'model/yolov8n-seg.onnx';
const INPUT_SIZE = 640;
const MASK_SIZE = 160; // proto grid size (640 / 4)
const NUM_CLASSES = 80;
const IOU_THRESHOLD = 0.45;

let session = null;
let currentImage = null; // an HTMLImageElement, the source photo at natural size

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const dropZone = document.getElementById('dropZone');
const dropContent = document.getElementById('dropContent');
const fileInput = document.getElementById('fileInput');
const chooseBtn = document.getElementById('chooseBtn');
const cameraBtn = document.getElementById('cameraBtn');
const cameraVideo = document.getElementById('cameraVideo');
const cameraControls = document.getElementById('cameraControls');
const snapBtn = document.getElementById('snapBtn');
const cancelCameraBtn = document.getElementById('cancelCameraBtn');

const resultPanel = document.getElementById('resultPanel');
const outputCanvas = document.getElementById('outputCanvas');
const confSlider = document.getElementById('confSlider');
const confValue = document.getElementById('confValue');
const resetBtn = document.getElementById('resetBtn');
const detectionsEl = document.getElementById('detections');

const statusLine = document.getElementById('statusLine');
const progressBar = document.getElementById('progressBar');

let cameraStream = null;

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------
async function loadModel() {
  statusLine.textContent = 'Downloading model (~13 MB, once per visit)…';
  progressBar.style.width = '15%';
  try {
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'],
    });
    progressBar.style.width = '100%';
    statusLine.textContent = 'Model ready. Drop a photo to begin.';
    setTimeout(() => { progressBar.style.width = '0%'; }, 600);
  } catch (err) {
    console.error(err);
    statusLine.textContent = 'Could not load the model. Check your connection and reload the page.';
  }
}
loadModel();

// ---------------------------------------------------------------------------
// UI wiring — photo input
// ---------------------------------------------------------------------------
chooseBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
dropZone.addEventListener('click', () => { if (!cameraStream) fileInput.click(); });

fileInput.addEventListener('change', () => {
  if (fileInput.files && fileInput.files[0]) {
    loadImageFile(fileInput.files[0]);
  }
});

['dragenter', 'dragover'].forEach(evt =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  })
);
['dragleave', 'drop'].forEach(evt =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
  })
);
dropZone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) loadImageFile(file);
});

function loadImageFile(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    currentImage = img;
    runPipeline();
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

// ---------------------------------------------------------------------------
// UI wiring — camera capture
// ---------------------------------------------------------------------------
cameraBtn.addEventListener('click', async (e) => {
  e.stopPropagation();
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    cameraVideo.srcObject = cameraStream;
    dropContent.hidden = true;
    cameraVideo.hidden = false;
    cameraControls.hidden = false;
  } catch (err) {
    statusLine.textContent = 'Could not access the camera. You can still upload a photo instead.';
  }
});

cancelCameraBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  stopCamera();
});

snapBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  const tmp = document.createElement('canvas');
  tmp.width = cameraVideo.videoWidth;
  tmp.height = cameraVideo.videoHeight;
  tmp.getContext('2d').drawImage(cameraVideo, 0, 0);
  const img = new Image();
  img.onload = () => {
    currentImage = img;
    stopCamera();
    runPipeline();
  };
  img.src = tmp.toDataURL('image/png');
});

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  cameraVideo.hidden = true;
  cameraControls.hidden = true;
  dropContent.hidden = false;
}

resetBtn.addEventListener('click', () => {
  currentImage = null;
  resultPanel.hidden = true;
  dropZone.hidden = false;
  detectionsEl.innerHTML = '';
});

confSlider.addEventListener('input', () => {
  confValue.textContent = confSlider.value;
  if (currentImage) runPipeline();
});

// ---------------------------------------------------------------------------
// Preprocessing — letterbox resize to 640x640
// ---------------------------------------------------------------------------
function letterbox(image, size = INPUT_SIZE) {
  const origW = image.naturalWidth || image.width;
  const origH = image.naturalHeight || image.height;
  const scale = Math.min(size / origW, size / origH);
  const newW = Math.round(origW * scale);
  const newH = Math.round(origH * scale);
  const padLeft = Math.floor((size - newW) / 2);
  const padTop = Math.floor((size - newH) / 2);

  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgb(114,114,114)';
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(image, 0, 0, origW, origH, padLeft, padTop, newW, newH);

  return { canvas, scale, padLeft, padTop, newW, newH, origW, origH };
}

function canvasToTensor(canvas) {
  const ctx = canvas.getContext('2d');
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const size = canvas.width * canvas.height;
  const floatData = new Float32Array(size * 3);
  // NCHW, RGB, normalized to [0,1]
  for (let i = 0; i < size; i++) {
    floatData[i] = data[i * 4] / 255;                 // R
    floatData[size + i] = data[i * 4 + 1] / 255;       // G
    floatData[size * 2 + i] = data[i * 4 + 2] / 255;   // B
  }
  return new ort.Tensor('float32', floatData, [1, 3, canvas.height, canvas.width]);
}

// ---------------------------------------------------------------------------
// Postprocessing — decode YOLOv8-seg output, NMS, mask assembly
// ---------------------------------------------------------------------------
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-6);
}

function decodeDetections(output0Data, confThreshold) {
  // output0: (1, 116, 8400) flattened. Row c, anchor i -> output0Data[c*8400 + i]
  const numAnchors = 8400;
  const stride = numAnchors;
  const detections = [];

  for (let i = 0; i < numAnchors; i++) {
    let bestScore = 0;
    let bestClass = -1;
    for (let c = 0; c < NUM_CLASSES; c++) {
      const score = output0Data[(4 + c) * stride + i];
      if (score > bestScore) { bestScore = score; bestClass = c; }
    }
    if (bestScore < confThreshold) continue;

    const cx = output0Data[0 * stride + i];
    const cy = output0Data[1 * stride + i];
    const w = output0Data[2 * stride + i];
    const h = output0Data[3 * stride + i];

    const maskCoeffs = new Float32Array(32);
    for (let k = 0; k < 32; k++) {
      maskCoeffs[k] = output0Data[(4 + NUM_CLASSES + k) * stride + i];
    }

    detections.push({
      x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2,
      score: bestScore, classId: bestClass, maskCoeffs,
    });
  }
  return detections;
}

function nms(detections, iouThreshold) {
  const byClass = {};
  for (const d of detections) {
    (byClass[d.classId] = byClass[d.classId] || []).push(d);
  }
  const kept = [];
  for (const classId in byClass) {
    const boxes = byClass[classId].sort((a, b) => b.score - a.score);
    const used = new Array(boxes.length).fill(false);
    for (let i = 0; i < boxes.length; i++) {
      if (used[i]) continue;
      kept.push(boxes[i]);
      for (let j = i + 1; j < boxes.length; j++) {
        if (!used[j] && iou(boxes[i], boxes[j]) > iouThreshold) used[j] = true;
      }
    }
  }
  return kept;
}

// Build a colored, alpha-masked overlay for one detection and draw it onto ctx
// at the detection's location in the ORIGINAL image.
function drawMaskOverlay(ctx, det, protoData, letterboxInfo) {
  const { scale, padLeft, padTop } = letterboxInfo;
  const g = MASK_SIZE / INPUT_SIZE; // 640 -> 160 scale factor (0.25)

  // Bounding box in 640-space, clamped
  const bx1 = Math.max(0, det.x1), by1 = Math.max(0, det.y1);
  const bx2 = Math.min(INPUT_SIZE, det.x2), by2 = Math.min(INPUT_SIZE, det.y2);

  // Corresponding box in 160-space (proto grid)
  const gx1 = Math.floor(bx1 * g), gy1 = Math.floor(by1 * g);
  const gx2 = Math.ceil(bx2 * g), gy2 = Math.ceil(by2 * g);

  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = MASK_SIZE;
  maskCanvas.height = MASK_SIZE;
  const mctx = maskCanvas.getContext('2d');
  const imgData = mctx.createImageData(MASK_SIZE, MASK_SIZE);

  const color = classColor(det.classId);
  const rgb = hslToRgb(color);

  const planeSize = MASK_SIZE * MASK_SIZE;
  for (let y = 0; y < MASK_SIZE; y++) {
    for (let x = 0; x < MASK_SIZE; x++) {
      const idx = y * MASK_SIZE + x;
      let alpha = 0;
      if (x >= gx1 && x < gx2 && y >= gy1 && y < gy2) {
        let sum = 0;
        for (let k = 0; k < 32; k++) {
          sum += det.maskCoeffs[k] * protoData[k * planeSize + idx];
        }
        const v = sigmoid(sum);
        alpha = v > 0.5 ? 150 : 0;
      }
      const p = idx * 4;
      imgData.data[p] = rgb[0];
      imgData.data[p + 1] = rgb[1];
      imgData.data[p + 2] = rgb[2];
      imgData.data[p + 3] = alpha;
    }
  }
  mctx.putImageData(imgData, 0, 0);

  // Upscale mask (160x160, representing the 640x640 letterboxed frame)
  // to a 640x640 canvas, then map the unpadded region back onto the
  // original image using the same transform used for letterboxing.
  const upscaled = document.createElement('canvas');
  upscaled.width = INPUT_SIZE;
  upscaled.height = INPUT_SIZE;
  const uctx = upscaled.getContext('2d');
  uctx.imageSmoothingEnabled = true;
  uctx.drawImage(maskCanvas, 0, 0, MASK_SIZE, MASK_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

  const { newW, newH, origW, origH } = letterboxInfo;
  ctx.drawImage(
    upscaled,
    padLeft, padTop, newW, newH,   // source: the unpadded region in 640-space
    0, 0, origW, origH             // dest: the full original image
  );
}

function hslToRgb(hslStr) {
  // hslStr like 'hsl(120, 70%, 50%)'
  const m = hslStr.match(/hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)/);
  const h = +m[1] / 360, s = +m[2] / 100, l = +m[3] / 100;
  if (s === 0) { const v = Math.round(l * 255); return [v, v, v]; }
  const hue2rgb = (p, q, t) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const r = hue2rgb(p, q, h + 1 / 3), gC = hue2rgb(p, q, h), b = hue2rgb(p, q, h - 1 / 3);
  return [Math.round(r * 255), Math.round(gC * 255), Math.round(b * 255)];
}

// ---------------------------------------------------------------------------
// Full pipeline: preprocess -> run model -> decode -> draw
// ---------------------------------------------------------------------------
async function runPipeline() {
  if (!currentImage || !session) return;

  dropZone.hidden = true;
  resultPanel.hidden = false;
  statusLine.textContent = 'Running segmentation…';
  progressBar.style.width = '40%';

  const lb = letterbox(currentImage, INPUT_SIZE);
  const inputTensor = canvasToTensor(lb.canvas);

  const results = await session.run({ images: inputTensor });
  const output0 = results['output0'].data;   // Float32Array, 116*8400
  const output1 = results['output1'].data;   // Float32Array, 32*160*160

  progressBar.style.width = '75%';

  const confThreshold = parseFloat(confSlider.value);
  const raw = decodeDetections(output0, confThreshold);
  const detections = nms(raw, IOU_THRESHOLD);

  // Draw original image at natural size
  outputCanvas.width = lb.origW;
  outputCanvas.height = lb.origH;
  const ctx = outputCanvas.getContext('2d');
  ctx.drawImage(currentImage, 0, 0, lb.origW, lb.origH);

  // Draw mask overlays
  for (const det of detections) {
    drawMaskOverlay(ctx, det, output1, lb);
  }

  // Draw boxes + labels on top
  const scaleX = lb.origW / INPUT_SIZE, scaleY = lb.origH / INPUT_SIZE;
  for (const det of detections) {
    // unletterbox box coords
    const ox1 = (det.x1 - lb.padLeft) / lb.scale;
    const oy1 = (det.y1 - lb.padTop) / lb.scale;
    const ox2 = (det.x2 - lb.padLeft) / lb.scale;
    const oy2 = (det.y2 - lb.padTop) / lb.scale;

    const color = classColor(det.classId);
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, lb.origW / 400);
    ctx.strokeRect(ox1, oy1, ox2 - ox1, oy2 - oy1);

    const label = `${COCO_CLASSES[det.classId]} ${Math.round(det.score * 100)}%`;
    ctx.font = `${Math.max(12, lb.origW / 70)}px sans-serif`;
    const textW = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(ox1, Math.max(0, oy1 - 20), textW + 10, 20);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, ox1 + 5, Math.max(14, oy1 - 5));
  }

  renderDetectionChips(detections);

  progressBar.style.width = '100%';
  statusLine.textContent = detections.length
    ? `Found ${detections.length} object${detections.length === 1 ? '' : 's'}.`
    : 'No objects detected above this confidence threshold.';
  setTimeout(() => { progressBar.style.width = '0%'; }, 500);
}

function renderDetectionChips(detections) {
  const counts = {};
  for (const d of detections) {
    counts[d.classId] = (counts[d.classId] || 0) + 1;
  }
  detectionsEl.innerHTML = '';
  Object.keys(counts).forEach(classId => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.style.background = classColor(parseInt(classId));
    chip.textContent = `${counts[classId]}× ${COCO_CLASSES[classId]}`;
    detectionsEl.appendChild(chip);
  });
}
