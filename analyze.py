import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from metrics import sketch_scores

SKETCHES = ["alien_1", "alien_2", "alien_3"]
SCALES = [0.0, 0.5, 1.0, 1.3]
STYLES = ["oil", "photo", "watercolor"]

def make_row(images, labels, size=256):
    """Make a row of images with labels below each image."""
    row = Image.new("RGB", (size * len(images), size + 30), "white")
    draw = ImageDraw.Draw(row)
    font = ImageFont.load_default(size=14)
    for i, (img, label) in enumerate(zip(images, labels)):
        row.paste(img.convert("RGB").resize((size, size)),(i * size, 0))
        draw.text((i * size + 5, size + 5), label, fill="black", font=font)
    return row

grids = Path("outputs/grids")
grids.mkdir(exist_ok=True)
rows = []
for name in SKETCHES:
    folder = Path("outputs") / name
    sketch = Image.open(f"sketches/{name}.jpeg")
    control = Image.open(folder / "control.png")

    # Grid 1: sketch strength
    images, labels = [sketch, control], ["sketch", "control"]
    for scale in SCALES:
        img = Image.open(folder / f"scale_{scale}.png")
        recall, precision = sketch_scores(control, img)
        rows.append({"sketch": name, "experiment": "scale", "setting": scale,
                     "recall": round(recall, 3), "precision": round(precision, 3)})
        images.append(img)
        labels.append(f"scale {scale} | recall {recall:.2f}")
    make_row(images, labels).save(grids / f"{name}_scale.png")

    # Grid 2: styles
    images, labels = [sketch], ["sketch"]
    for style in STYLES:
        img = Image.open(folder / f"style_{style}.png")
        recall, precision = sketch_scores(control, img)
        rows.append({"sketch": name, "experiment": "style", "setting": style,
                     "recall": round(recall, 3), "precision": round(precision, 3)})
        images.append(img)
        labels.append(f"{style} | recall {recall:.2f}")
    make_row(images, labels).save(grids / f"{name}_styles.png")

# Save all scores
with open("outputs/metrics.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Plot: recall vs scale
plt.figure(figsize=(6, 4))
for name in SKETCHES:
    recalls = [r["recall"] for r in rows if r["sketch"] == name and r["experiment"] == "scale"]
    plt.plot(SCALES, recalls, marker="o", label=name)
plt.xlabel("ControlNet conditioning scale")
plt.ylabel("Sketch recall")
plt.title("Sketch preservation vs conditioning scale")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig(grids / "recall_vs_scale.png", dpi=150, bbox_inches="tight")

print("Done! Grids and plot saved in outputs/grids/")
