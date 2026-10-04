"""Create loss curve, before/after snapshot, and inference memory footprint."""
import json, os, resource, time
import cv2, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E
from dataset import list_images, label_path_for, polygons_to_mask, read_polygons
from inference import load_model, predict_mask, DEFAULT_DATA

def sc(tag, key):
    a = E(f"runs/lane_unet/{tag}"); a.Reload(); return [e.value for e in a.Scalars(key)]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for t in ("train", "val"):
    ax[0].plot(range(1, 31), sc(f"loss_{t}", "loss"), label=t)
    ax[1].plot(range(1, 31), sc(f"iou_{t}", "iou"), label=t)
ax[0].set(title="Loss (0.5 BCE + 0.5 Dice)", xlabel="epoch"); ax[1].set(title="IoU", xlabel="epoch")
for a in ax: a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("assets/loss_curve.png", dpi=150)

model = load_model("checkpoints/best.pt", torch.device("cpu"))
files = list_images(f"{DEFAULT_DATA}/images/test")
ev = json.load(open("evaluation_results.json"))["per_image"]
ious = {r["image"]: r["iou"] for r in ev}
picks = [files[0], files[len(files) // 2], files[-1]]
fig, ax = plt.subplots(len(picks), 4, figsize=(14, 3.2 * len(picks)))
for i, f in enumerate(picks):
    bgr = cv2.imread(f); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pred = predict_mask(model, bgr, torch.device("cpu"))
    gt = polygons_to_mask(read_polygons(label_path_for(f)), *pred.shape)
    ov = rgb.copy(); ov[pred > 0] = (0.5 * ov[pred > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    name = os.path.splitext(os.path.basename(f))[0]
    for a, im, t in zip(ax[i], [rgb, gt, pred, ov], ["Input (before)", "Ground truth", "Predicted mask", f"Overlay (after) IoU={ious[name]:.3f}"]):
        a.imshow(im, cmap="gray"); a.set_title(t, fontsize=9); a.axis("off")
plt.tight_layout(); plt.savefig("assets/inference_example.png", dpi=120)

# memory footprint (CPU inference)
n = sum(p.numel() for p in model.parameters())
torch.save(model.state_dict(), "/tmp/w.pt"); wsize = os.path.getsize("/tmp/w.pt") / 2**20
x = torch.rand(1, 3, 48, 48)
act = []
hooks = [m.register_forward_hook(lambda m, i, o: act.append(o.numel() * 4)) for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
with torch.no_grad(): model(x)
for h in hooks: h.remove()
with torch.no_grad():
    for _ in range(20): model(x)
    t = time.time()
    for _ in range(200): model(x)
    ms = (time.time() - t) / 200 * 1000
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # bytes on macOS
res = {"params": n, "weights_fp32_MB": round(n * 4 / 2**20, 3), "state_dict_file_MB": round(wsize, 3),
       "activation_sum_MB_batch1": round(sum(act) / 2**20, 3), "cpu_latency_ms_per_48x48": round(ms, 2),
       "process_peak_RSS_MB": round(rss, 1)}
json.dump(res, open("memory_footprint.json", "w"), indent=2); print(res)
