"""Build a task's question files into the dataset root.

    python -m mozaic.annotation.run_annotation --task t1
    python -m mozaic.annotation.run_annotation --task t1 --size 200
    python -m mozaic.annotation.run_annotation --task t1 --size 200 --seed 7
    python -m mozaic.annotation.run_annotation --task t3 --size 40      # t3: size set in config.yaml
    python -m mozaic.annotation.run_annotation --task t1 --subset small # metadata_small.jsonl -> qas_small/

T4 is annotated by models (Gemini explains, Claude verifies and rewrites), on MOZAIC-Small only; every run names one step (t4_explanation/README.md):

    python -m mozaic.annotation.run_annotation --task t4 --subset small --select                  # pick the clips
    python -m mozaic.annotation.run_annotation --task t4 --subset small --annotate --golden-only  # golden only
    python -m mozaic.annotation.run_annotation --task t4 --subset small --annotate                # every clip
    python -m mozaic.annotation.run_annotation --task t4 --subset small --build                   # final files
    python -m mozaic.annotation.run_annotation --task t4 --subset small --annotate --golden-only --out DIR  # a trial
"""

from __future__ import annotations

import argparse
import logging

from mozaic.common.paths import use_subset

T4_STEPS = ("select", "annotate", "build")


def main() -> None:
    p = argparse.ArgumentParser(description="Build a task's question files.")
    p.add_argument("--task", required=True, choices=["t1", "t2", "t3", "t4"])
    p.add_argument("--size", type=int, help="write a balanced sample of the test split instead of the split files")
    p.add_argument("--seed", type=int, help="sample seed (default in the task config)")
    p.add_argument("--out", help="folder to write to instead of qas/<task> (for experiments)")
    p.add_argument("--subset", help="run on a subset, e.g. small (default: the full dataset)")
    t4 = p.add_argument_group("t4 (annotated by a model): one step per run")
    t4.add_argument("--select", action="store_true", help="pick the clips: train, val, test and the golden set; no model")
    t4.add_argument("--annotate", action="store_true", help="Gemini explains each clip, Claude verifies and rewrites; save each reply")
    t4.add_argument("--build", action="store_true", help="selection + saved replies -> final files, no calls")
    t4.add_argument("--golden-only", action="store_true", help="with --annotate: only the golden clips")
    t4.add_argument("--retry-failed", action="store_true",
                    help="with --annotate: ask again the clips that failed: Gemini from round 1, or only the rewriter")
    t4.add_argument("--limit", type=int, help="with --annotate: only the first N clips")
    args = p.parse_args()
    use_subset(args.subset)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for name in ("httpx", "httpx2"):  # openai >= 3 sends requests through httpx2
        logging.getLogger(name).setLevel(logging.WARNING)

    steps = [s for s in T4_STEPS if getattr(args, s)]
    if args.task == "t4":
        if len(steps) != 1:
            p.error("t4 needs one step: --select, --annotate or --build")
        if args.size or args.seed is not None:
            p.error("t4 takes no --size or --seed: the golden set is set in t4_explanation/config.yaml")
        if (args.golden_only or args.retry_failed or args.limit) and not args.annotate:
            p.error("--golden-only, --retry-failed and --limit go with --annotate")
        from mozaic.annotation.t4_explanation.build_questions import use_out
        use_out(args.out)  # every step in --out instead of qas_small/t4_explanation/
        if args.select:
            from mozaic.annotation.t4_explanation.build_questions import select
            select()
        elif args.annotate:
            from mozaic.annotation.t4_explanation.annotate import annotate
            annotate(golden_only=args.golden_only, limit=args.limit, retry_failed=args.retry_failed)
        elif args.build:
            from mozaic.annotation.t4_explanation.build_questions import build_final
            build_final()
        return
    if steps or args.golden_only or args.retry_failed or args.limit:
        p.error("the t4 flags are for --task t4")

    if args.task == "t1":
        from mozaic.annotation.t1_spot_the_fake.build_questions import build
    elif args.task == "t2":
        from mozaic.annotation.t2_manipulation_type.build_questions import build
    elif args.task == "t3":
        from mozaic.annotation.t3_identity_check.build_questions import build
    build(size=args.size, seed=args.seed, out=args.out)


if __name__ == "__main__":
    main()
