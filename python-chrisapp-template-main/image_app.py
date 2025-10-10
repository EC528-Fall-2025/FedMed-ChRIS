import torch
from torchvision import models

@chris_plugin(title="pl-resnet", category="analysis")
def main(options, inputdir: Path, outputdir: Path):
    outputdir.mkdir(parents=True, exist_ok=True)

    print("Loading ResNet18 model (first run may download weights)...", flush=T>    weights = models.ResNet18_Weights.DEFAULT
    net = models.resnet18(weights=weights).eval()
    preprocess = weights.transforms()
    labels = weights.meta["categories"]

    results = []
    for p in inputdir.rglob("*.jpg"):
        try:
            img = Image.open(p).convert("RGB")
        except UnidentifiedImageError:
            continue
        with torch.no_grad():
            probs = torch.softmax(net(preprocess(img).unsqueeze(0)), dim=1)[0]
        idx = int(probs.argmax())
        results.append({"file": p.name, "label": labels[idx], "score": float(pr>
    (outputdir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Done. Wrote {len(results)} results to {outputdir/'results.json'}", >
if __name__ == "__main__":
    main()
