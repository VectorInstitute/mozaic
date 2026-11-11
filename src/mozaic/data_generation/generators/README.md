# generators

The four generators that make MOZAIC's fake streams: three for the picture, one for the voice. Every pair of a
build gets one render (a fake picture of the driver's clip) and one converted voice, and its four clips are
made from them:

    RVRA   the driver's crop                with its real sound
    RVFA   the driver's crop                with the converted voice
    FVRA   the render                       with the real sound
    FVFA   the render                       with the converted voice

Each generator has a job maker (`make_jobs.py`, run in the core environment, writes a job list and checks every
input) and a runner (`run.py`, run in the generator's own environment from its clone in the git-ignored
external-repos/, one shard per GPU task; it imports nothing from mozaic, so every value it needs is on the job).
Settings live in data_generation/configs/<generator>.yaml.

    generators/
    ├── reference_frame.py   the face photo of the video generators: the best of 24 frames of the person's own clip
    ├── facefusion/          FaceFusion, inswapper_128_fp16: swaps the face only, the rest is the driver's crop
    ├── s2v/                 Wan2.2-S2V-14B: the face photo + the real speech -> a new talking video
    ├── animate/             Wan2.2-Animate-14B, replace mode: the whole frame redrawn from the face photo,
    │                        moving as the driver moves (77-frame crops only)
    └── seed_vc/             Seed-VC: the driver's words in the other person's voice; and the vocoded copies of
                             real voices (BigVGAN, Seed-VC's vocoder), which stay labelled real

## What the video generators share

- **One build, one video generator**: the manifest names it per row (video_method), and its render_key prefix
  says which (`cross` and `self` for Animate, `ffcross` and `ffself` for FaceFusion, `s2vcross` and `s2vself`
  for S2V).
- **One render per render_key**: FVRA and FVFA of a pair share it; the self-drive control (the driver re-made
  as themselves) is a render of its own.
- **The same face photo**: reference_frame.py scores 24 frames spread over the person's own VoxCeleb2 clip
  (sharpness of the face x both eyes found, on a frontal face) and keeps the best; the photo is that frame,
  <data>/new_generated/refs/<render_key>.png. Never a still from another dataset.
- **Seeds**: S2V and Animate get one seed per render, blake2b of "42:<render_key>"; FaceFusion takes none.
- **Outputs**: <data>/new_generated/renders/<render_key>.mp4, with its sidecar <render_key>.json (how it was
  made) and <render_key>.done. A render the generator refused (<render_key>.refused.json) or that failed
  (<render_key>.failed.json; each runner retries the generation once first) drops its whole pair: no class of it
  ships (settings.refused_pair_ids).
- **Job lists**: <data>/new_generated/manifests/<build>/<generator>_jobs<suffix>/jobs.jsonl, the suffix
  `_pairs<N>_<sha8>` for a batch of pairs (settings.batch_suffix_for).

## Order

    crop/             the driver's crops (video and sound)
    make_jobs.py      the job list, after the manifest check (validate_report.json says PASS)
    run.py            the renders: scripts/data_generation/render_<generator>.sh, one GPU per array task
    seed_vc/          the converted voices and the vocoded copies
    final_clips/           the mux: every clip, real or fake, through the same steps to one format

Each generator's README says what it does to a clip, its settings, where its code and weights come from, and
how to run it.
