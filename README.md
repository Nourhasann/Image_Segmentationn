
# Segment — browser-only image segmentation website

A real, permanently free website. Visitors upload a photo or use their camera,
and object segmentation runs **entirely in their browser** — nothing is
uploaded to any server, so there's no backend to pay for or keep running.

## What's inside

- `index.html` / `style.css` — the page and its design
- `app.js` — loads the model and runs it (preprocessing, inference, decoding, drawing)
- `labels.js` — the 80 object categories the model recognizes
- `model/yolov8n-seg.onnx` — the AI model itself (~13 MB), converted from
  YOLOv8 so it can run via [ONNX Runtime Web](https://onnxruntime.ai/) directly
  in JavaScript

## How to publish it (pick one — both are free forever)

### Option A — GitHub Pages
1. Create a new GitHub repository
2. Upload all the files
3. Go to the repo's **Settings → Pages**, set the source to your main branch
4. Your site appears at `https://your-username.github.io/your-repo-name/`

### Option B — Hugging Face Static Space
1. Go to huggingface.co/new-space, choose the **Static** SDK (free, no restrictions)
2. Upload all the files in this folder
3. Your site appears at `https://huggingface.co/spaces/your-username/your-space-name`

That's it — no build step, no server, no account tier limits, no sleep/cold starts.

## How it works, briefly

The original YOLOv8 model (a PyTorch file) was converted to the **ONNX**
format, an open standard that many languages and runtimes understand —
including JavaScript, via ONNX Runtime Web. The browser downloads the model
once, then runs all the math (image preprocessing, detection, and mask
generation) using WebAssembly, entirely on the visitor's own device.


