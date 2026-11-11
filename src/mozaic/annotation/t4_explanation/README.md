# T4: Explanation

Which parts of a clip, if any, were generated or altered by AI, where, and what in the
clip shows it? The evaluated model sees the frames and hears the sound of one clip, and
answers with an explanation.

**What is true comes from metadata; how it is said comes from models.** The facts (what is
fake, where, for how long) and the signs (gate scores past a cutoff) are taken from metadata by
code, as numbered claims. Models then turn the claims into text, in one call per clip and kind of
part it has: its fake parts, and its real parts. The steps of a fake part:

| step | model | sees | does | code |
|---|---|---|---|---|
| annotator | Gemini (`gemini-3.6-flash`) | the clip (8 timed frames + its sound), the claims, the person's gender | writes the sentences of each fake part: the scene, what is AI-generated and for how long, where the signs show; cites the claims per sentence | `models/annotator.py`, `prompts/annotator.py` |
| checks | code | the reply and the claims | citations, length, defect words, method words, leaks, claim numbers, times | `checks/rules.py` |
| verifier | Claude (`claude-sonnet-5-5`) | text only: the claims and the sentences | says per sentence whether it adds anything no cited claim gives | `models/verifier.py`, `prompts/verifier.py` |
| rewriter | Claude (`claude-sonnet-5-5`) | text only: the checked sentences | rewrites each part as one paragraph, the way a person would: no repeats, no dashes | `models/rewriter.py`, `prompts/rewriter.py` |
| checks + verifier, again | code, then Claude | the rewrite and the checked sentences | code: no new defect, every sign in its own words, no dash, the same rules; the verifier: anything the rewrite adds, drops or changes | `checks/rules.py`, `prompts/verifier.py` |

**Real parts** go through the same steps, in their own call (`kind: real`), so a real part reads like
a fake one: the scene, then the verdict. Their claims are a verdict ("the video is a natural recording,
with nothing AI-generated") and a scene; there is nothing to find, so the annotator describes content
only (who, where, what they do or say, never how a face, the light or the voice looks or sounds), and
the checks fail any defect word, any word calling the part made or changed by AI, and a reply or
rewrite without "natural recording" and "nothing AI-generated". RVRA has a real call only, FVFA a fake
call only, RVFA and FVRA one of each, in either order. Real parts are asked as many sentences as a
fake part with no sign, so length does not tell real from fake.

    metadata_small.jsonl
      → facts/metadata_facts.py   facts (what is fake, where) and signs (scores past a cutoff, worded by strength)
      → facts/claims.py           the numbered claims: per fake part verdict, fact, signs, scene; per real part verdict, scene
      → annotate.py               per clip and kind: annotator → checks → verifier (up to 2 rounds), then rewriter → checks → verifier
      → build_questions.py        selection + passing replies → t4_*.jsonl

    t4_explanation/
    ├── README.md
    ├── config.yaml             vocoded view, golden set, the three models
    ├── annotate.py             the steps per clip, replies.jsonl, stats.json (resumable)
    ├── build_questions.py      select(): splits + golden; build_final(): replies → files
    ├── facts/
    │   ├── metadata_facts.py   metadata row → facts and signs
    │   ├── claims.py           facts → numbered claims, of the fake parts and of the real parts
    │   └── artifacts.py        the artifact categories (closed list)
    ├── models/
    │   ├── client.py           the OpenAI-compatible client every model uses
    │   ├── annotator.py        Gemini: the clip (timed frames + sound) + its prompt
    │   ├── verifier.py         Claude: its prompt only
    │   └── rewriter.py         Claude: a system prompt + the checked sentences
    ├── prompts/
    │   ├── annotator.py        what Gemini is told
    │   ├── verifier.py         what the verifier is told (sentences vs claims, rewrite vs sentences), how its reply is read
    │   └── rewriter.py         what the rewriter is told
    └── checks/
        └── rules.py            the code checks on the reply and on the rewrite

## Why these models

Gemini sees and hears the clip, so it can describe it (who, where, what happens) and say
where a listed sign shows, which a text model cannot. But a model that watches can also
invent a defect. So it never decides what is wrong:

- every defect must be a claim from metadata, cited by the sentence that names it;
- the code checks catch the known words (a defect family no cited claim has, "overall",
  a method name, "the original video", "looks natural", ...);
- the verifier catches the rest: a defect in other words, an added timing, a judgement.

The verifier is another model, so the annotator never grades its own text, and it reads text
only, so it judges the sentences against the claims, not against what it sees.

The annotator's sentences each cite their claims, so joined they read as a list. The rewriter
turns them into one paragraph per part. It sees only the checked sentences, and its paragraphs are
checked against them twice: by code (`check_rewrite`: no defect they do not name, each sign in
its own words, "AI-generated" and "AI-altered" kept, no dash, the same rules as the reply), then by
the verifier with its own prompt (`REWRITE_PROMPT`): it reads the checked sentences as its claims,
knows the rewriter never saw the clip, and fails anything the rewrite adds (a scene detail too),
drops or changes. Only a rewrite that passes both is used.

The person's gender (`gender` in metadata, `m` or `f`, one per clip: both people of a pair share
it) is given to the annotator, so the text names the person right.

**Not checked by any model or rule:** the scene (clothes, setting, lighting, objects). It is
neutral description, never a defect, but it can be wrong: the human review of the golden set
checks it.

## Rounds and failures

Per clip (`tries: 2` in config.yaml), at most 2 Gemini calls and 4 Claude calls:

    round 1: Gemini writes → code checks → (if they pass) Claude verifies
    round 2: only if round 1 failed: Gemini again, with round 1's problems listed
    rewrite: only if a round passed: Claude rewrites → code checks → (if they pass) Claude verifies

A clip that fails both rounds is saved as failed (`ok: false`), with every round's reply and
problems. It is never replaced by other text: `--annotate --retry-failed` asks it again from
round 1, and `--build` leaves it out until it passes. A clip whose sentences passed but whose
rewrite failed is saved with that rewrite's problems; `--build` leaves it out too, and
`--retry-failed` asks only the rewriter again (no Gemini call), keeping the failed rewrites in
`rewrites_failed`. After `rewrite_tries` (3) failed rewrites of the same sentences, `--retry-failed`
asks Gemini again from round 1 instead: new sentences, then a new rewrite. A failed call (the API down) is not
saved, so the next run asks again; the run stops after 3 failed calls in a row.

The sentences are checked when they are written (their result is `ok`); what ships is the
rewrite, and `--build` and `--retry-failed` run the code checks again on it, so a stricter check
needs no new call: a rewrite that fails it now counts as failed, and `--retry-failed` asks only
the rewriter again.

**Stats** (`build/stats.json`, after every `--annotate`), for the report:

    clips             explained, passed, failed
    passed_in_round   how many passed in round 1, how many a second round fixed
    rounds_failed_by  checks, verifier, cut off
    problems          each problem kind and how often (says warp, talks about deciding, verifier: adds, ...)
    rewrites          passed, failed, fixed by a retry, attempts failed by checks / verifier / cut off, and why
    calls, tokens     per model
    asked_again       clips --retry-failed took out to ask again (kept in replies_dropped.jsonl)

## Run

T4 is built on MOZAIC-Small only: every step refuses to run without `--subset small`. Every run names one step:

    # 1. which clips: train / val / test, and 400 golden clips from test; no model
    python -m mozaic.annotation.run_annotation --task t4 --subset small --select

    # 2. Gemini explains, Claude verifies and rewrites (resumable; the same command continues)
    python -m mozaic.annotation.run_annotation --task t4 --subset small --annotate --golden-only --limit 8
    python -m mozaic.annotation.run_annotation --task t4 --subset small --annotate --golden-only
    python -m mozaic.annotation.run_annotation --task t4 --subset small --annotate --golden-only --retry-failed
    python -m mozaic.annotation.run_annotation --task t4 --subset small --annotate          # every clip

    # 3. final files from the saved replies; no calls
    python -m mozaic.annotation.run_annotation --task t4 --subset small --build

The order: `--select`, then `--annotate --golden-only`, `--build` and a human review of
t4_golden.jsonl; only if the golden set holds up, `--annotate` for every clip and `--build` again.

    --select            pick the clips: train, val, test and the golden set; no model
    --annotate          explain every selected clip, golden first: each kind of part with no saved reply
      --golden-only     only the golden clips
      --limit N         only the first N clips
      --retry-failed    ask again the clips that failed: Gemini from round 1, or only the rewriter
    --build             selection + passing replies -> final files; no calls

`--out DIR` runs any step in DIR instead of qas_small/t4_explanation/, for trials on a few
clips: put their selection rows in DIR/build/selection/golden.jsonl (and empty train/val,
test = golden to `--build`), then `--annotate --golden-only --out DIR`.

Run it with `.venv_gemini` (scripts/annotation/annotate_t4.sh). Every model is called through an
OpenAI-compatible endpoint (`base_url` in config.yaml: the public Gemini and Anthropic URLs, or another endpoint), with its
key in the `.env` from common/config.yaml: `GEMINI_API_KEY` (Gemini) and
`ANTHROPIC_API_KEY` (Claude: the verifier and the rewriter). Never commit them.

## Facts, signs and claims

**Facts** (`facts/metadata_facts.py`, `facts()`): which parts are fake (`face_real`, `voice_real`), the region the
method changes (inner face, whole frame, voice), and what was changed (fixed text per
generator). Every fake covers the whole clip: there is no "when".

**Signs** (`SIGNS`): a fake video's quality scores (`sharp` and `melt` relative to the real
clip, 1 = the same as real). A score past its cutoff gives one sign, tagged with an artifact
category; past its strong cutoff too, the sign is worded as strong. Real clips have no
scores, so no signs. The wording says what one sees in the clip alone: the scores compare
the clip with its real recording, which the evaluated model never sees.

| score | cutoff | category | sign | strong (past) |
|---|---|---|---|---|
| `sharp` | < 0.7 | unnatural blur | the face looks a little soft and blurry | < 0.6: the face looks clearly soft and blurry, without fine detail |
| `melt` | > 1.25 | shape or structure distortion | the face wobbles or warps as it moves | > 1.5: the face strongly wobbles and warps as it moves |
| `melt` | < 0.8 | unnatural human movement | the face barely moves or changes expression | < 0.65: the face hardly moves or changes expression at all |

- `melt` gives no sign for S2V (`OFF`), as in the quality gate: S2V invents its own head
  movement, so its face changes differ from the real clip's by design, not by warping.
- `flicker` gives no sign: it is not relative to the real clip, and every released clip is
  within the real clips' range (the gate's limit is their 99th percentile).
- `sharp` < 0.7 keeps the clearly softer faces; 0.85 was FaceFusion's median (half of it).
  The strong cutoffs split the test clips with that sign near their quartiles.

**Claims** (`claims_of()`), per fake part, in order: the verdict ("the video is
AI-generated"), the fact (what is generated, "from the start of the clip to the end"), each
sign, and a scene slot ("what is on screen" / "what is heard") that the scene sentences cite.
The verdict, fact and signs must all be cited.

## Coverage and limits

A fake part has a sign only when a score passes a cutoff:

| set | clips | fake videos with a sign | FaceFusion | Animate | S2V | fake parts with a sign |
|---|---|---|---|---|---|---|
| train | 6,664 | 774 / 4,490 | 9% | 60% | 11% | 774 / 7,822 |
| val | 1,332 | 114 / 898 | 7% | 35% | 11% | 114 / 1,564 |
| test | 3,334 | 350 / 2,226 | 7% | 56% | 11% | 350 / 3,893 |
| golden | 400 | 65 / 300 | 4% | 49% | 12% | 65 / 500 |

- Most fake parts say only what is AI-generated, where and for how long (plus the scene):
  that much is also scored exactly (fake parts and regions, evaluation/metrics/t4_metrics.py).
- Fake voices never have a sign, so the three audio categories in facts/artifacts.py are unused.
  There is no audio-visual category: whether a voice fits a face cannot be told from one clip.
- No sign is not "nothing visible": it only means no measured score passed a cutoff. A
  model that names a real artifact the gate did not measure is not wrong.
- Score the explanation against `gt_target` (which signs it names, whether it calls a real
  part fake), not against the wording: the wording is Gemini's.
- The annotator is a Gemini model: never evaluate it, the verifier or the rewriter on T4
  (inference refuses all three). A later Gemini may still favour Gemini-like wording; another reason to
  score `gt_target`, not the text.

## Human review

The golden set is reviewed by people before the other splits are explained. Each fake part
is labelled **good** or **bad**: bad if a sign it states is not there, it states something the
clip contradicts, or its scene description is wrong. The reviewers see the clip, its pair's
real clip, the facts, the evidence and the gate scores with their cutoffs.

## Files

    qas_small/t4_explanation/
    ├── t4_train.jsonl, t4_val.jsonl, t4_test.jsonl    step 3: explained rows only
    ├── t4_golden.jsonl                                step 3: the golden clips (also in t4_test.jsonl)
    └── build/
        ├── selection/{train,val,test,golden}.jsonl    step 1: one row per clip, facts only
        ├── settings.json                              the setup of this folder (models, prompts, claims, cutoffs)
        ├── replies.jsonl                              step 2: one row per clip and kind (fake, real; none: fake): claims, reply, ok, every round, the rewrite (and failed ones)
        ├── replies_dropped.jsonl                      the rows --retry-failed asked again, for the record
        └── stats.json                                 what failed, where, and what a second round fixed

Released: the four t4_*.jsonl files. build/ holds working files.

**Which clips.** Train, val and test hold every clip of the vocoded view in their split.
The golden set is 400 clips from test, 50 per cell (label × video generator), spread over
length and gender, one clip per pair, for human review only. `--select` again writes the
same clips, and refuses if the metadata or the config would change them.

**Vocoded view.** A real voice is its vocoded copy only (RVRA, FVRA vocoded); a converted
voice has one version (RVFA, FVFA main), already vocoded inside Seed-VC.

Evaluate a model on `t4_test.jsonl` with `--subset small` (evaluation/README.md). Inference
refuses `t4_golden.jsonl`.

## Record

    {
      "id": "t4_FVRA__animate__77f__vocoded__...",
      "task": "task4_explanation",
      "split": "test",
      "clips": ["videos/FVRA/animate/77f/vocoded/FVRA__animate__77f__vocoded__...mov"],
      "question": "Which parts of this audio-visual clip, if any, were generated or altered by AI, ...",
      "options": ["none", "inner face", "whole frame", "voice"],
      "gt_answer": "A close-up shot shows a bald man wearing a suit speaking against a backdrop with red lighting. ... During these head movements, the face strongly wobbles and warps as it moves. ... A man speaks about ... The audio is a natural recording, with nothing AI-generated.",
      "gt_label": "FVRA",
      "gt_target": {"fake_parts": ["video"], "regions": ["whole frame"],
                    "parts": {"video": {"fake": true, "region": "whole frame", "method": "animate",
                                        "changed": "...", "original": null, "signs": [...]},
                              "audio": {"fake": false, "region": "none", "method": "real", ...}},
                    "sections": {"video": "A close-up shot ...", "audio": "A man speaks about ..."},
                    "evidence": [{"modality": "video", "artifact": "shape or structure distortion",
                                  "meaning": "the face strongly wobbles and warps as it moves"}]},
      "meta": {"video_method": "animate", "audio_method": "bigvgan", "frames": 77, "arm": "vocoded",
               "source_id": "...", "target_id": "...", "pair_id": "...",
               "annotator": "gemini-3.6-flash", "verifier": "claude-sonnet-5-5", "rewriter": "claude-sonnet-5-5",
               "human_checked": false}                 # true in t4_golden.jsonl (reviewed)
    }
