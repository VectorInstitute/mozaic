"""Step 3: the final model, the best epoch's LoRA merged into the full Qwen2.5-Omni → <run>/final/.

LLaMA-Factory saves only the LoRA, and its own export keeps only the thinker. This follows its
scripts/qwen_omni_merge.py (merge_lora): the released model with the LoRA merged into its thinker,
saved with the released talker and processor, so evaluation runs it with weights: <run>/final.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from mozaic.common.io import share_tree, write_json
from mozaic.training.mllms.inputs import inputs_of
from mozaic.training.mllms.run_config import load, parse_args

log = logging.getLogger(__name__)


def checkpoints(out: Path) -> dict[int, Path]:
    """{epoch: checkpoint folder}, from each checkpoint's trainer_state.json."""
    found = {}
    for path in (out / "checkpoints").glob("checkpoint-*"):
        found[round(json.loads((path / "trainer_state.json").read_text())["epoch"])] = path
    return found


def merge(weights: str, adapter: Path, dest: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForTextToWaveform, AutoProcessor

    model = AutoModelForTextToWaveform.from_pretrained(weights, torch_dtype=torch.bfloat16, device_map="cpu")
    model.thinker = PeftModel.from_pretrained(model.thinker, str(adapter)).merge_and_unload()
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    AutoProcessor.from_pretrained(weights).save_pretrained(tmp)
    if (Path(weights) / "spk_dict.pt").exists():  # the talker's voices
        shutil.copy(Path(weights) / "spk_dict.pt", tmp / "spk_dict.pt")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)


def main() -> None:
    args = parse_args("Step 3: merge the best epoch (or --epoch) into the final model.")
    config, out = load(args)
    found = checkpoints(out)
    if not found:
        raise SystemExit(f"no checkpoint in {out / 'checkpoints'}: train first")
    if args.epoch:
        epoch, chosen_by = args.epoch, f"--epoch {args.epoch}"
    else:
        best = out / "val_scores.json"
        if not best.exists():
            raise SystemExit(f"no {best} (written by validation during training); or pick one with --epoch")
        epoch, chosen_by = json.loads(best.read_text())["epoch"], f"val {config['train']['select_on']}"
    if epoch not in found:
        raise SystemExit(f"no checkpoint for epoch {epoch}; saved: {sorted(found)}")

    info = out / "final.json"
    if info.exists() and (out / "final").exists() and json.loads(info.read_text())["epoch"] == epoch:
        log.info("final model is already epoch %d", epoch)
        return
    info.unlink(missing_ok=True)  # never a final.json for another epoch's model
    getattr(inputs_of(config), "merge", merge)(config["weights"], found[epoch], out / "final")
    share_tree(out / "final")
    write_json(info, {"epoch": epoch, "checkpoint": found[epoch].name, "chosen_by": chosen_by})
    log.info("final model: epoch %d (%s), chosen by %s → %s", epoch, found[epoch].name, chosen_by, out / "final")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    main()
