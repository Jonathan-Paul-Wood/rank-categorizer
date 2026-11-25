import json
import re
from pathlib import Path

from .catalog import MODELS

ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class Registry:
    def __init__(self, settings):
        self.settings = settings
        self.bundle = json.loads((settings.bundle_dir / "bundle.json").read_text(encoding="utf-8"))
        self.bundled_id = self.bundle["model_id"]
        if self.bundled_id not in MODELS or self.bundle["revision"] != MODELS[self.bundled_id]["revision"]:
            raise ValueError("bundle does not match pinned model catalog; rebuild image")
        self.models = settings.data_dir / "models"
        self.jobs = settings.data_dir / "jobs"
        self.models.mkdir(parents=True, exist_ok=True)
        self.jobs.mkdir(parents=True, exist_ok=True)
        # Resolve on startup, not on the first inference request.
        self.active_path = self.resolve(settings.model_id)

    def custom(self):
        entries = []
        for path in sorted(self.models.glob("ft-*/model.json")):
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
                if entry["id"] == path.parent.name and ID.fullmatch(entry["id"]):
                    entries.append(entry)
            except (OSError, ValueError, KeyError):
                continue
        return entries

    def compatible(self, entry):
        return (entry.get("parent_model") == self.bundled_id
                and entry.get("parent_revision") == self.bundle["revision"]
                and entry.get("base_revision") == self.bundle["base_revision"])

    def resolve(self, model_id):
        if model_id == self.bundled_id:
            path = self.settings.bundle_dir / "checkpoint"
        else:
            entry = next((item for item in self.custom() if item["id"] == model_id), None)
            if not entry or not self.compatible(entry):
                raise ValueError("KEV_MODEL is absent or incompatible with this image; select/rebuild its standard base")
            path = self.models / model_id / "checkpoint"
        if not (path / "head.pt").is_file():
            raise ValueError("selected model checkpoint is incomplete")
        return path

    def catalog(self):
        entries = [{"id": key, "kind": "standard", "bundled": key == self.bundled_id,
                    "selectable": key == self.bundled_id, "active": key == self.settings.model_id,
                    "fine_tunable": key != "kev-27b", **value} for key, value in MODELS.items()]
        entries.extend({**entry, "kind": "custom", "bundled": False,
                        "selectable": self.compatible(entry), "active": entry["id"] == self.settings.model_id}
                       for entry in self.custom())
        return entries

