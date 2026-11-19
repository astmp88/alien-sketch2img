import csv
from pathlib import Path
from preprocess import preprocess_sketch
from generate import load_pipeline, generate

ALIEN = "a cute purple alien with two antennae and a big smiling face"
CAT = "a cute brown with white lines cat with big eyes and a happy face"

SCENES = {
    "alien_1": f"a flying saucer UFO with a glass dome, {ALIEN} inside waving both hands, light beam shining down, night forest",
    "alien_2": f"{ALIEN} is out of the UFO, he is surprised to see {CAT} inside the UFO, {CAT} is happy to see {ALIEN}, night forest, light beam shining down",
    "alien_3": f"{ALIEN} now is inside of the UFO with {CAT}, they are both happy, night forest, light beam shining down",
}

STYLES = {
    "oil": "oil painting, romantic art style",
    "photo": "photorealistic, highly detailed, cinematic lighting",
    "watercolor": "watercolor illustration, soft colors, children's book",
}

SCALES = [0.0, 0.5, 1.0, 1.3]

pipe = load_pipeline()
results = []

for name, scene in SCENES.items():
    control = preprocess_sketch(f"sketches/{name}.jpeg")
    folder = Path("outputs") / name
    folder.mkdir(parents=True, exist_ok=True)
    control.save(folder / "control.png")

    # Experiment 1: sketch strength (same style)
    for scale in SCALES:
        prompt = f"{scene}, {STYLES['oil']}"
        image, seconds = generate(pipe, control, prompt, scale=scale)
        image.save(folder / f"scale_{scale}.png")
        results.append({"sketch": name, "experiment": "scale", "scale": scale,
                        "style": "oil", "seconds": round(seconds, 2)})

    # Experiment 2: styles (same strength)
    for style, words in STYLES.items():
        prompt = f"{scene}, {words}"
        image, seconds = generate(pipe, control, prompt, scale=1.0)
        image.save(folder / f"style_{style}.png")
        results.append({"sketch": name, "experiment": "style", "scale": 1.0,
                        "style": style, "seconds": round(seconds, 2)})

with open("outputs/results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"Done! {len(results)} images generated.")