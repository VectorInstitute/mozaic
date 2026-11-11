# Seed-VC voices

The fake voices of MOZAIC, and the vocoded copies of its real voices. Both run on the code of
[Seed-VC](https://github.com/Plachtaa/seed-vc) (GPL-3.0; cloned and called, never copied here).

**Converted voice (RVFA, FVFA).** The driver's own words in the other person's voice. Seed-VC, a
diffusion transformer, hears two sounds: the driver's cropped sound (the words and their timing) and ONE
reference clip of the other person (the voice): the first of the pair's `voice_ref_clip_ids`, a whole
clip, not cropped (Seed-VC reads at most its first 25 s). The voice keeps the source timing
(`length_adjust` 1.0), so it has the crop's length and the lips in the video stay in sync. RVFA (real face)
and FVFA (fake face) of a pair use the same converted voice: one conversion per `conversion_key`. In the
same-speaker arm, an RVFA control, the reference comes from another recording of the driver: the identity
does not change, so only the conversion's own trace is left.

**Vocoded copy (real voices).** Every real voice also goes through BigVGAN, the vocoder Seed-VC ends in,
with Seed-VC's own mel step and settings, but without the conversion. The copy keeps the same words, voice
and timing and stays labelled real (`audio_variant` vocoded_real): it is the sound of the vocoded view of
RVRA and FVRA. Every converted voice went through this vocoder, so real and fake voices share the vocoder's
trace, and a detector cannot call a voice fake for the vocoder alone.

    seed_vc/
    ├── make_conversion_pairs.py   pair list → conversion_pairs.jsonl, one line per converted voice    .venv
    ├── make_jobs.py               conversion_pairs.jsonl → seedvc_jobs/jobs.jsonl                       .venv
    ├── run.py                     jobs.jsonl → conversions/, one shard per GPU                          .venv_seed_vc
    ├── make_vocode_jobs.py        pair list → vocode_jobs.jsonl, one line per real crop                 .venv
    └── vocode.py                  vocode_jobs.jsonl → vocoded/                                          .venv_seed_vc

The makers run in the core environment (`python -m ...`). run.py and vocode.py run in Seed-VC's own
environment as plain scripts and import nothing from mozaic, only Seed-VC's code from its clone.

## Settings

Set in configs/seed_vc.yaml and run.yaml, read through settings.py.

| | |
|---|---|
| Model | Seed-VC's default path: `DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth`, no pitch input (`f0_condition` off), on a GPU; the diffusion steps in float16 (Seed-VC's own autocast) |
| Diffusion steps | 5 |
| Guidance (`inference_cfg_rate`) | 0.3 |
| Length (`length_adjust`) | 1.0, the source timing |
| Reference | one clip of the target person, whole |
| Voice | 22,050 Hz wav (Seed-VC's rate); the mux brings it to 16 kHz with every other sound |
| Seed | one per voice: blake2b of `"42:<conversion_key>"` (8 bytes, as a number) mod 2^31 − 1; torch is seeded with it right before every conversion |
| Vocoder | `nvidia/bigvgan_v2_22khz_80band_256x`, the one Seed-VC's default path ends in |
| Vocoded copy | 16 kHz → 22,050 Hz → mel (n_fft and window 1024, hop 256, 80 bands, fmin 0, no fmax, center off; read from Seed-VC's config) → BigVGAN → 22,050 Hz → 16 kHz, on CPU |

Seed-VC's only random draw is its starting noise, and nothing in Seed-VC seeds it. With one seed per voice,
set right before it is made, a voice does not depend on its shard or its place in the list.

## Files

Under new_generated/ (the data folder of configs/paths.yaml):

    manifests/<build>/conversion_pairs.jsonl     one line per converted voice: its two wavs, seed, settings, the clips that use it
    manifests/<build>/seedvc_jobs/jobs.jsonl     one job per voice, for run.py
    manifests/<build>/vocode_jobs.jsonl          one job per real crop, for vocode.py
    conversions/<conversion_key>.wav             the converted voice; .json how it was made; .done written last
    conversions/<conversion_key>.failed.json     Seed-VC failed twice: a documented drop, the pair ships in no class
    vocoded/<crop_key>.wav                       the vocoded copy (16 kHz); .json its stages and sample rates

make_conversion_pairs and make_vocode_jobs refuse a build whose pair list did not pass its check
(manifests/<build>/validate_report.json, verdict PASS), and stop when a wav they need is missing
(`--allow-missing`: write the list anyway, not for real runs); make_jobs leaves out a pair whose wavs are
missing. `--pair-ids-file` limits a list to some pairs: MOZAIC was made in batches of pairs, almost all of
200, each batch's lists named with the suffix of `settings.batch_suffix_for`
(conversion_pairs_pairs200_<8 hex>.jsonl, seedvc_jobs_pairs200_<8 hex>/, vocode_jobs_pairs200_<8 hex>.jsonl),
passed with `--out` and `--out-dir`.

## Code and weights

Seed-VC's code, cloned into external-repos/seed-vc (git-ignored, never pushed) at the commit MOZAIC used:

    git clone https://github.com/Plachtaa/seed-vc external-repos/seed-vc
    git -C external-repos/seed-vc checkout 51383efd921027683c89e5348211d93ff12ac2a8

Its own environment, .venv_seed_vc, on Python 3.12:

    uv venv .venv_seed_vc --python 3.12
    uv pip install --python .venv_seed_vc/bin/python --torch-backend cu126 \
        -r src/mozaic/data_generation/generators/seed_vc/requirements.txt

That is torch 2.6.0, torchaudio 2.6.0 and torchvision 0.21.0 for CUDA 12.6 (Seed-VC's own list asks for
torch 2.4.0, which has no CUDA 12.6 build), then the rest of Seed-VC's requirements.txt without
sounddevice and FreeSimpleGUI, which only its live microphone demo uses. mozaic is not installed there.

Seed-VC loads every model it has when it starts, even the ones this path does not use, and downloads what
is missing. Its own loader always saves into a relative ./checkpoints folder, so run.py and vocode.py work
from inside the clone: the first four rows below land in external-repos/seed-vc/checkpoints/. BigVGAN and
Whisper go to the Hugging Face cache (HF_HOME, by default ~/.cache/huggingface; the jobs need the same
HF_HOME). Where the jobs have no internet, download them first, once, from inside the clone and one model
at a time (building Seed-VC's wrapper loads them all at once and can run out of memory on a small machine):

    cd external-repos/seed-vc
    ../../.venv_seed_vc/bin/python - <<'EOF'
    from hf_utils import load_custom_model_from_hf
    from modules.bigvgan import bigvgan
    from transformers import AutoFeatureExtractor, WhisperModel

    load_custom_model_from_hf("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                              "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
    load_custom_model_from_hf("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
                              "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")
    load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin")
    load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt")
    for name in ("nvidia/bigvgan_v2_22khz_80band_256x", "nvidia/bigvgan_v2_44khz_128band_512x"):
        bigvgan.BigVGAN.from_pretrained(name, use_cuda_kernel=False)
    WhisperModel.from_pretrained("openai/whisper-small")
    AutoFeatureExtractor.from_pretrained("openai/whisper-small")
    EOF
    cd ../..

| Weights (Hugging Face) | What for | Version MOZAIC used |
|---|---|---|
| Plachta/Seed-VC: `DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth`, `config_dit_mel_seed_uvit_whisper_small_wavenet.yml` | the conversion model, and the mel settings of both steps | 257283f9f41585055e8f858fba4fd044e5caed6e |
| Plachta/Seed-VC: `DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth`, `config_dit_mel_seed_uvit_whisper_base_f0_44k.yml` | loaded, not used (pitch input off) | 257283f9f41585055e8f858fba4fd044e5caed6e |
| funasr/campplus: `campplus_cn_common.bin` | the reference's speaker embedding | e4b6ede7ce16997aff4ae69fbca1f0175e2afede |
| lj1995/VoiceConversionWebUI: `rmvpe.pt` | pitch; loaded, not used | e6d0c1a17da07c33557852f9dfa2bd44cc75737d |
| openai/whisper-small | the words (content features) | 973afd24965f72e36ca33b3055d56a652f456b4d |
| nvidia/bigvgan_v2_22khz_80band_256x | the vocoder, of the voices and the vocoded copies | 633ff708ed5b74903e86ff1298cf4a98e921c513 |
| nvidia/bigvgan_v2_44khz_128band_512x | loaded, not used | 95a9d1dcb12906c03edd938d77b9333d6ded7dfb |

The loaders take each model's latest version; the last column is the one in MOZAIC's caches.

## Run

From the repo root; `<data>` is the data folder of configs/paths.yaml.

    # converted voices: the pairs and the jobs (core .venv), then one GPU per shard
    python -m mozaic.data_generation.generators.seed_vc.make_conversion_pairs --build-id <build>
    python -m mozaic.data_generation.generators.seed_vc.make_jobs \
        --pairs <data>/new_generated/manifests/<build>/conversion_pairs.jsonl \
        --out-dir <data>/new_generated/manifests/<build>/seedvc_jobs
    sbatch scripts/data_generation/convert_seed_vc.sh        # JOBS and OUT at its top

    # vocoded copies: the jobs (core .venv), then the vocoder on CPU
    python -m mozaic.data_generation.generators.seed_vc.make_vocode_jobs --build-id <build>
    sbatch scripts/data_generation/vocode.sh                 # JOBS at its top

Without Slurm, in .venv_seed_vc, the commands the two scripts run:

    python src/mozaic/data_generation/generators/seed_vc/run.py --jobs <jobs.jsonl> \
        --upstream $PWD/external-repos/seed-vc --out-dir <data>/new_generated/conversions
    SEEDVC_UPSTREAM=$PWD/external-repos/seed-vc python src/mozaic/data_generation/generators/seed_vc/vocode.py \
        --jobs <vocode_jobs.jsonl> --device cpu

- Absolute paths only: run.py and vocode.py work from inside the Seed-VC folder. The job lists hold
  absolute paths when configs/paths.yaml does, so keep its folders absolute.
- A rerun continues: run.py skips a voice with a .done, vocode.py a copy that exists (`--overwrite` makes
  it again). run.py tries a failing voice once more, then writes its .failed.json and goes on.
- run.py stops when it finds no GPU (on CPU a voice takes minutes instead of seconds); `--device cpu` runs
  it on CPU on purpose. `--dry-run` prints the jobs left.
- One GPU per shard is enough (an A40 or smaller): Seed-VC's models are small next to the video generators.

## Same input, same bytes

- Converted voices: seeded per voice, so the same job list on the same kind of GPU, with the same
  software, gives the same files.
- Vocoded copies: MOZAIC's were made on CPU (8 cores), so vocode.sh passes `--device cpu`; on a GPU the
  numbers differ slightly.
- The job lists hold the absolute paths of the wavs, so lists made with another data folder differ in
  those paths only.
