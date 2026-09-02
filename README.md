# Image-Based Vehicle Lane-Change Detection

Research internship project — **Multimedia Lab, Yuan Ze University, Taiwan**
Supervisor: Prof. Duan-Yu Chen · Jun – Aug 2025
Author: Nutt Bhanidch (KMUTT, Electronics & Infocommunication Engineering)

Detecting *"that car just changed lanes"* from ordinary road video needs two facts at the
same time — **where the lanes are** and **where the cars are** — and they come from two
different models whose outputs disagree frame to frame. This repo is the pipeline that
reconciles them.

📄 **[Final presentation (PDF)](docs/internship-presentation.pdf)**

---

## Pipeline

```
                ┌── CLRerNet ─────────────► lane boundaries (ordered point sequences)
                │                                          │
camera video ───┤                                          ├──► lane-change rule ──► event
                │                                          │      (temporal
                └── YOLOv11-s + tracker ──► vehicle boxes ─┘       consistency)
                                            + stable IDs
```

| Stage | Model | What it produces |
|---|---|---|
| Lane detection | CLRerNet (DLA-34 / ResNet-18, CULane) | lane lines as ordered point sequences, drawn as continuous polylines |
| Vehicle detection | YOLOv11-s | boxes + class + confidence for cars, trucks, buses, motorcycles |
| Tracking | `HomemadeStrongSORT` — written from scratch in this repo | one stable ID per vehicle across frames |
| Decision | custom rule | each vehicle's **bottom edge** is tracked; when it crosses a lane boundary **consistently over multiple frames**, a lane change is flagged |

The last row is the part that matters. A single-frame crossing test produces a storm of
false positives — jitter in either model is enough to trip it. Requiring the crossing to
hold across consecutive frames is what makes the output usable.

### Why a hand-written tracker

`HomemadeStrongSORT` (in the notebooks) is a from-scratch centroid + IoU association
tracker with a disappearance counter, not a wrapper around a library. It was written
this way to keep the association logic inspectable while debugging why IDs were
swapping between adjacent vehicles — which turned out to be the root cause of most of
the early false lane-change events.

---

## Repository layout

```
notebooks/
  lane_change_detection.ipynb              ← main integrated pipeline
  lane_change_detection_v1.ipynb           ← earlier version, kept for comparison
  yolo_homemade_tracker.ipynb              ← the tracker in isolation
  yolo_homemade_tracker_experimental.ipynb ← tracker + RAFT optical flow experiments
  early_experiments.ipynb                  ← first-week exploration
clrernet/
  detect_video.py                          ← run CLRerNet over a video file
  clr_resnet18_culane.py                   ← ResNet-18 CULane config used
docs/
  internship-presentation.pdf              ← final presentation to the lab
results/                                   ← output frames from each stage
requirements.txt
```

---

## Results

| | |
|---|---|
| ![lane detection baseline](results/lane_detection_baseline.png) | ![integrated 1](results/integrated_example_1.png) |
| Lane detection alone | Lanes + vehicles + IDs |
| ![integrated 2](results/integrated_example_2.png) | ![own footage](results/own_footage_1.png) |
| Integrated system | Run on footage shot by hand, not from the dataset |

More in [`results/`](results/).

🎬 **[`results/lane_change_demo.mp4`](results/lane_change_demo.mp4)** — 8 seconds of the
integrated system running: lane boundaries, tracked vehicles with stable IDs, and the
lane-change counter, all on one frame.

> Two notes on the video files. The longer demo (~131 MB) is not here — over GitHub's
> file size limit. And every output clip is encoded as **MPEG-4 Part 2 (mp4v)**, which is
> what `cv2.VideoWriter` produces by default; browsers cannot play that, so the clip needs
> converting to H.264 before it will play anywhere but a desktop player:
>
> ```bash
> ffmpeg -i results/lane_change_demo.mp4 -c:v libx264 -crf 23 -movflags +faststart out.mp4
> ```
>
> Some of the other output files in the original working folder are also **truncated and
> will not open at all** — worth checking before relying on one.

---

## Running it

The notebooks were written for a lab machine with a CUDA GPU. CLRerNet in particular
needs a pinned, fairly old stack (`torch==1.8.0`, `mmcv==1.2.5`) — see
[`requirements.txt`](requirements.txt). Getting that environment to build is genuinely
the hardest part of reproducing this; the upstream repo ships a Dockerfile and using it
is strongly recommended over a bare `pip install`.

```bash
# 1. lane detection weights + code (upstream, not vendored here)
git clone https://github.com/hirotomusiker/CLRerNet
# follow their README / docker setup, then download a CULane checkpoint

# 2. run lane detection over a video
python clrernet/detect_video.py     # edit the paths at the bottom of the file first

# 3. the integrated pipeline
jupyter notebook notebooks/lane_change_detection.ipynb
```

`detect_video.py` still contains the absolute paths of the lab's DGX machine at the
bottom of the file. They are left as-is rather than cleaned up, because they document
what the code actually ran against.

---

## What is mine and what is not

**Mine:** everything in `notebooks/` and `clrernet/detect_video.py`, the lane-change
decision rule, the tracker, and the integration.

**Not mine:**

- **CLRerNet** — *"CLRerNet: Improving Confidence of Lane Detection with LaneIoU"*,
  Hiroto Honda and Yusuke Uchida, WACV 2024.
  [github.com/hirotomusiker/CLRerNet](https://github.com/hirotomusiker/CLRerNet) ·
  Apache License 2.0. **Not vendored into this repo** — clone it separately.
- **YOLOv11** — Ultralytics, AGPL-3.0.
- CULane dataset for the pre-trained lane weights.

---

## Honest limitations

1. **No quantitative benchmark.** The deliverable was a working end-to-end pipeline
   demonstrated on video, not a number on a public benchmark. There is no mAP or
   lane-change F1 to quote here, and this README is not going to invent one.
2. **Tuned on a small set of clips.** The temporal-consistency thresholds were set by
   watching failures on the clips available in the lab, not by a search over a held-out
   set. Expect them to need retuning on different camera heights and frame rates.
3. **Daytime, clear-weather footage only.** Nothing here was tested at night or in rain.
4. **The tracker is a teaching-grade StrongSORT**, not the published one — no appearance
   embeddings, no Kalman filter. It holds IDs well enough for this task and no further.

---

## License

[Apache License 2.0](LICENSE) — see [NOTICE](NOTICE) for third-party attributions.

Apache 2.0 was chosen partly because CLRerNet, the lane detection model this
builds on, is under the same licence. Note that YOLOv11 is **AGPL-3.0**, which
is strong copyleft: if you deploy a network service built on it, that obligation
reaches your own source too. Listed here so nobody is surprised by it later.
