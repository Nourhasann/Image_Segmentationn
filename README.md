---
title: Segment
emoji: "✂️"
colorFrom: gray
colorTo: blue
sdk: static
pinned: false
---

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
2. Upload all the files in this folder, **keeping the `model/` folder structure**
3. Go to the repo's **Settings → Pages**, set the source to your main branch
4. Your site appears at `https://your-username.github.io/your-repo-name/`

### Option B — Hugging Face Static Space
1. Go to huggingface.co/new-space, choose the **Static** SDK (free, no restrictions)
2. Upload all the files in this folder, keeping the `model/` folder structure
3. Your site appears at `https://huggingface.co/spaces/your-username/your-space-name`

That's it — no build step, no server, no account tier limits, no sleep/cold starts.

## Testing it locally first (optional)

Browsers block loading local files directly (`file://`) for security reasons,
so to preview it on your own computer before publishing, serve it with a
simple local server:

```bash
cd this-folder
python3 -m http.server 8000
```
Then open `http://localhost:8000` in your browser.

## How it works, briefly

The original YOLOv8 model (a PyTorch file) was converted to the **ONNX**
format, an open standard that many languages and runtimes understand —
including JavaScript, via ONNX Runtime Web. The browser downloads the model
once, then runs all the math (image preprocessing, detection, and mask
generation) using WebAssembly, entirely on the visitor's own device.

## Notes

- First-time load takes a few seconds while the ~13 MB model downloads;
  after that, the browser caches it.
- Speed depends on the visitor's device. It's smooth on most modern laptops
  and phones; very old/low-power devices may take a bit longer per image.
- Everything runs client-side — no image is ever sent anywhere, and there's
  no usage limit, no paid tier, and no server to keep alive.
- If you'd like a bigger/more accurate model (slower) or a smaller/faster one
  swapped in, that's a one-line change in `app.js` plus re-exporting that
  model size — let me know if you want that done.
