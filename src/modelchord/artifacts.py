"""Content-bound manifests and safe tensors. No pickle or remote-code loaders."""
from __future__ import annotations
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from .models import MLP, from_spec

MAX_BYTES = 64 * 1024 * 1024

class CompatibilityError(ValueError):
    pass

def canonical(value: Any) -> bytes:
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def state_digest(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode()); h.update(str(v.dtype).encode()); h.update(canonical(list(v.shape)))
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@dataclass(frozen=True)
class Artifact:
    path: Path
    manifest: dict[str, Any]

    @classmethod
    def open(cls, path: str | Path) -> "Artifact":
        path = Path(path).resolve()
        manifest_path = path / "manifest.json"
        if manifest_path.stat().st_size > 1024*1024:
            raise CompatibilityError("Oversized manifest")
        m = json.loads(manifest_path.read_text())
        required = {"schema", "architecture", "base_id", "contract", "license", "tensors", "weights_sha256", "provenance", "ties"}
        if set(m) != required or m["schema"] != "modelchord/v1":
            raise CompatibilityError("Unsupported or malformed manifest")
        try:
            expected = from_spec(m["architecture"]).state_dict()
        except (ValueError, TypeError, KeyError) as exc:
            raise CompatibilityError(str(exc)) from exc
        if m["ties"] != []:
            raise CompatibilityError("Tied tensors unsupported in mlp/v1; untie only with a new declared contract")
        if not isinstance(m["base_id"],str) or len(m["base_id"]) != 64:
            raise CompatibilityError("Missing immutable base identity")
        if not isinstance(m["license"],str) or m["license"] not in {"MIT","Apache-2.0","CC-BY-4.0","CC0-1.0"}:
            raise CompatibilityError("Unsupported redistribution license; export a recipe after legal review")
        contract = m["contract"]
        if not isinstance(contract,dict) or not {"family","representation","preprocessing","labels","tokenizer"} <= contract.keys():
            raise CompatibilityError("Incomplete representation/preprocessing/tokenizer contract")
        if contract["labels"] != m["architecture"]["classes"]:
            raise CompatibilityError("Head class order disagrees with declared labels")
        w = path / "model.safetensors"
        if w.stat().st_size > MAX_BYTES or digest(w) != m["weights_sha256"]:
            raise CompatibilityError("Weights too large or hash mismatch")
        if set(m["tensors"]) != set(expected):
            raise CompatibilityError("Tensor identities differ from registry")
        with safe_open(w,framework="pt",device="cpu") as f:
            if set(f.keys()) != set(expected):
                raise CompatibilityError("Tensor identities differ from manifest")
            for k,v in expected.items():
                tensor = f.get_tensor(k)
                if m["tensors"][k] != {"shape":list(v.shape),"dtype":"float32"} or tensor.shape != v.shape or tensor.dtype != torch.float32:
                    raise CompatibilityError(f"Shape/precision mismatch: {k}")
                if not torch.isfinite(tensor).all():
                    raise CompatibilityError(f"Nonfinite tensor: {k}")
        canonical(m)
        return cls(path,m)

    def tensor(self, name: str) -> torch.Tensor:
        # Revalidate at each streamed read: an opened handle is no permission to
        # trust subsequently replaced source bytes.
        if digest(self.path / "model.safetensors") != self.manifest["weights_sha256"]:
            raise CompatibilityError("Source weights changed after opening")
        if canonical(json.loads((self.path / "manifest.json").read_text())) != canonical(self.manifest):
            raise CompatibilityError("Source manifest changed after opening")
        with safe_open(self.path / "model.safetensors",framework="pt",device="cpu") as f:
            return f.get_tensor(name)

    def state(self) -> dict[str,torch.Tensor]:
        return {k:self.tensor(k) for k in self.manifest["tensors"]}

    def model(self) -> MLP:
        model = from_spec(self.manifest["architecture"])
        model.load_state_dict(self.state(),strict=True)
        return model.eval()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.manifest["architecture"]["input_dim"] or not 1 <= len(x) <= 65536:
            raise ValueError("Input must be a nonempty bounded [batch,input_dim] tensor")
        if x.dtype != torch.float32 or not torch.isfinite(x).all() or x.abs().max() > 1e6:
            raise ValueError("Input must contain finite float32 values with magnitude <=1e6")
        with torch.inference_mode():
            logits = self.model()(x)
            if not torch.isfinite(logits).all():
                raise ValueError("Inference produced nonfinite outputs; reduce input magnitude or reject unstable weights")
            return logits

def validate_compatibility(base: Artifact, experts: list[Artifact]) -> dict[str,Any]:
    if state_digest(base.state()) != base.manifest["base_id"]:
        raise CompatibilityError("Supplied base weights do not match their immutable base identity")
    if not experts:
        raise CompatibilityError("At least one expert is required")
    if len(experts)>16:
        raise CompatibilityError("At most 16 experts per bounded build")
    for expert in experts:
        for key in ("base_id","architecture","contract","tensors","ties","license"):
            if canonical(expert.manifest[key]) != canonical(base.manifest[key]):
                raise CompatibilityError(f"{expert.path.name}: incompatible {key}")
    return {"compatible":True,"expert_count":len(experts),"base_id":base.manifest["base_id"],"contract_sha256":hashlib.sha256(canonical(base.manifest["contract"])).hexdigest()}

def export_artifact(path: str | Path, model: MLP, contract: dict[str,Any], *, base_id: str | None = None,
                    provenance: dict[str,Any] | None = None, license: str = "MIT") -> Artifact:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Artifact is immutable; choose a fresh output path: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    state = {k:v.detach().cpu().float().contiguous() for k,v in model.state_dict().items()}
    tmp = Path(tempfile.mkdtemp(prefix=".modelchord-",dir=path.parent))
    try:
        save_file(state,tmp/"model.safetensors")
        m = {"schema":"modelchord/v1","architecture":model.spec,"base_id":base_id or state_digest(state),
             "contract":contract,"license":license,"tensors":{k:{"shape":list(v.shape),"dtype":"float32"} for k,v in state.items()},
             "weights_sha256":digest(tmp/"model.safetensors"),"provenance":provenance or {},"ties":[]}
        (tmp/"manifest.json").write_bytes(canonical(m)+b"\n")
        Artifact.open(tmp)
        tmp.rename(path)
    except BaseException:
        shutil.rmtree(tmp)
        raise
    return Artifact.open(path)
