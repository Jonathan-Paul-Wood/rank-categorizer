"""Run in the model-tools Docker target: downloads are resumed in the host cache."""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from .catalog import MODELS, UPSTREAM_COMMIT
from .storage import write_json

TOKENIZER_FILES = ["config.json", "tokenizer*", "vocab*", "merges.txt", "special_tokens_map.json",
                   "added_tokens.json", "*.jinja", "chat_template*", "LICENSE*", "README.md"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(directory):
    manifest = json.loads((directory / "bundle.json").read_text())
    pin = MODELS[manifest["model_id"]]
    if manifest["revision"] != pin["revision"] or manifest["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("staged bundle pins changed; download again")
    if not manifest.get("files"):
        raise ValueError("bundle has no checksums")
    for name, checksum in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != checksum:
            raise ValueError(f"bundle checksum mismatch: {name}")
    return manifest


def pack(directory, archive):
    temporary = archive.with_suffix(".tmp")
    with tarfile.open(temporary, "w") as output:
        def include(member):
            if ".locks" in Path(member.name).parts:
                return None
            if member.isfile():
                member.mode |= 0o444
            elif member.isdir():
                member.mode |= 0o555
            return member
        output.add(directory, arcname=".", filter=include)
    temporary.replace(archive)


def download(model_id, root):
    import torch
    from huggingface_hub import HfApi, snapshot_download
    pin = MODELS[model_id]
    directory = root / model_id
    directory.mkdir(parents=True, exist_ok=True)
    # A failed/resumed download must not remain marked complete.
    (directory / "bundle.json").unlink(missing_ok=True)
    ck = directory / "checkpoint"
    snapshot_download(pin["repo"], revision=pin["revision"], local_dir=ck,
                      allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja", "LICENSE*", "README.md"])
    meta = torch.load(ck / "head.pt", map_location="cpu", weights_only=True)
    base = meta["base"]
    revision = meta.get("base_revision")
    # Older checkpoints may not carry a base pin. Resolve once and record it locally for offline use.
    if not revision:
        revision = HfApi().model_info(base).sha
        meta["base_revision"] = revision
        torch.save(meta, ck / "head.pt")
    patterns = list(TOKENIZER_FILES)
    if (ck / "adapter_config.json").exists():
        patterns += ["*.safetensors", "model.safetensors.index.json", "generation_config.json"]
    snapshot_download(base, revision=revision, cache_dir=directory / "hf" / "hub", allow_patterns=patterns)
    # local_dir metadata can contain download credentials/locks and is unnecessary at runtime.
    shutil.rmtree(ck / ".cache", ignore_errors=True)
    manifest = {"model_id": model_id, **pin, "base": base, "base_revision": revision,
                "upstream_commit": UPSTREAM_COMMIT,
                "files": {str(p.relative_to(directory)).replace("\\", "/"): sha256(p)
                          for p in sorted(directory.rglob("*")) if p.is_file() and ".locks" not in p.parts}}
    write_json(directory / "bundle.json", manifest)
    # Cache symlinks created by Linux containers are not valid Windows build-context entries.
    # A portable tar preserves them without doubling multi-GB blobs through dereferencing.
    pack(directory, root / f"{model_id}.tar")
    print(f"Prepared {model_id}: {directory}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="kev-0.8b")
    parser.add_argument("--root", type=Path, default=Path("/model-cache"))
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--copy-to", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--pack", type=Path)
    args = parser.parse_args()
    if args.pack:
        verify(args.pack)
        pack(args.pack, args.root / f"{args.model}.tar")
    elif args.archive:
        if not args.copy_to:
            parser.error("--archive requires --copy-to")
        args.copy_to.mkdir(parents=True)
        with tarfile.open(args.archive) as source:
            source.extractall(args.copy_to, filter="data")
        verify(args.copy_to)
    elif args.verify:
        verify(args.verify)
        if args.copy_to:
            shutil.copytree(args.verify, args.copy_to, symlinks=True)
    else:
        download(args.model, args.root)


if __name__ == "__main__":
    main()

