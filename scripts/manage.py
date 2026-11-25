"""Cross-platform Docker download/build/export/deploy CLI; never executes .env as shell code."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kev_service.catalog import MODELS  # noqa: E402


def config():
    values = {}
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                key, separator, value = line.partition("=")
                if not separator:
                    raise ValueError(".env entries must use KEY=value")
                values[key.strip()] = value.strip().strip("\"'")
    # Match Compose's environment-over-.env precedence.
    values.update({k: v for k, v in os.environ.items() if k.startswith("KEV_")})
    return values


def run(*args):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["download", "build", "export", "deploy", "test"])
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--archive", default="kev-ranker-image.tar")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--pull", action="store_true")
    args = parser.parse_args()
    env = config()
    model = args.model or env.get("KEV_BUNDLE_MODEL", "kev-0.8b")
    if model not in MODELS:
        parser.error("KEV_BUNDLE_MODEL is not a standard model")
    cuda = args.cuda or env.get("KEV_DEVICE", "cpu") == "cuda"
    default_index = "https://download.pytorch.org/whl/cu128" if cuda else "https://download.pytorch.org/whl/cpu"
    index = default_index if args.cuda else env.get("KEV_TORCH_INDEX", default_index)
    if cuda and index.endswith("/cpu"):
        parser.error("CUDA requires KEV_TORCH_INDEX=https://download.pytorch.org/whl/cu128")
    image = env.get("KEV_IMAGE", "kev-ranker:1.0-cuda" if cuda else "kev-ranker:1.0-cpu")
    build_args = ["--build-arg", f"TORCH_INDEX={index}", "--platform", "linux/amd64"]
    if args.action == "download":
        tools = "kev-ranker-model-tools:1.0"
        (ROOT / "model-cache").mkdir(exist_ok=True)
        # Downloading does not need GPU access; always use CPU tooling.
        run("docker", "build", "--target", "model-tools", "--platform", "linux/amd64", "-t", tools, ".")
        run("docker", "run", "--rm", "--mount", f"type=bind,source={ROOT / 'model-cache'},target=/model-cache",
            tools, "--model", model)
    elif args.action in ("build", "test"):
        if args.action == "build" and not (ROOT / "model-cache" / f"{model}.tar").exists():
            parser.error(f"run download --model {model} first")
        run("docker", "build", *build_args, "--target", "test" if args.action == "test" else "production",
            "--build-arg", f"KEV_BUNDLE_MODEL={model}", "-t", image + "-test" if args.action == "test" else image, ".")
    elif args.action == "export":
        run("docker", "save", "--output", args.archive, image)
    else:
        if not (ROOT / ".env").exists():
            parser.error("copy .env.example to .env before deploying")
        if args.load and args.pull:
            parser.error("choose either --load or --pull")
        if args.load:
            run("docker", "load", "--input", args.archive)
        compose = ["docker", "compose", "-f", "compose.yaml"]
        if cuda:
            compose += ["-f", "compose.cuda.yaml"]
        run(*compose, "up", "-d", "--no-build", "--pull", "always" if args.pull else "never",
            "--wait", "--wait-timeout", env.get("KEV_STARTUP_TIMEOUT", "600"))


if __name__ == "__main__":
    main()

