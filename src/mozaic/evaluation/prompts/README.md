# prompts

One file per task, for MLLMs. Each file turns a question into a prompt and a reply
back into one of the options, so every MLLM on a task is asked the same way.

    prompts/
    ├── t1_prompts.py     T1: spot the fake (frames + audio, answer by text)
    ├── t2_prompts.py     T2: manipulation type (frames only, answer by text)
    ├── t3_prompts.py     T3: two clips, frames (face) or sound (voice), yes or no
    └── t4_prompts.py     T4: explanation (frames + audio): a video and an audio explanation

## What a task file has

    MODALITIES       what the model gets: ("video", "audio") or ("video",)
    PROMPT           the prompt template, filled with the question and options
    build_prompt()   question record → prompt text (t1, t4: and the modalities sent)
    template()       t1, t4: PROMPT with its first line for video only or audio only (FIRST_LINES, in t1_prompts)
    parse()          reply text → one of the options, or None if unreadable
    clip_inputs()    only for tasks with several clips: (label, frames?, sound?) per clip

run_inference picks the task's file in load_prompts. Only MLLMs use prompts; the
chosen option becomes pred_* fields in evaluation/predictions.py.

## Rules

- The prompt holds only the question and options. Never the id, clip path or metadata.
- Options keep the order of the question file, which is shuffled per clip. The
  "What each option means" list follows the same order.
- The reply must be exactly one option. Anything else is saved with pred_answer null
  and counts as wrong.
- Options come from the task's annotation code (and predictions.py maps them to labels), so the two never drift.
- Changing PROMPT changes its hash in inference_settings.json, so a run with the old
  prompt is never mixed with a new one. Use a new experiment_name.

## T1 prompt

32 frames and the soundtrack, then:

    The images are {n} frames sampled evenly from one video clip, in time order.
    The audio is that clip's soundtrack.

    {question}

    What each option means:
    - "the video": the video is AI-generated or edited; the audio is real
    - "the audio": the audio is AI-generated or edited; the video is real
    - "both": both the video and the audio are AI-generated or edited
    - "neither": both the video and the audio are real

    Options: {options}

    Reply in JSON only: {"answer": "<one of the options, copied exactly>"}

## T2 prompt

32 frames, no sound, then:

    The images are {n} frames sampled evenly from one video clip, in time order.
    There is no sound.

    {question}

    What each option means:
    - "real": a camera recording; nothing was generated
    - "face swap": only the inner face was replaced; hair, head shape, body,
      background and all movement are from the original recording
    - "fully generated": the whole frame was generated, including the person and
      the background
    - "full-body swap": the whole frame was redrawn from one photo of another person,
      copying the movement of the original recording

    Options: {options}

    Reply in JSON only: {"answer": "<one of the options, copied exactly>"}

Options are shuffled per clip in the question file. The descriptions are listed in
the same order as that clip's options, so the two always match.

## T3 prompt

"Clip 1:", clip 1's 32 frames (face) or its sound (voice), "Clip 2:", the same for
clip 2, then for a face question:

    Above are {n} frames sampled evenly from each clip, in time order: first clip 1,
    then clip 2. There is no sound.

    {question}

    What each option means:
    - "real, same person": the face in clip 1 is a real recording of the person in clip 2
    - "real, different person": the face in clip 1 is a real recording of someone else
    - "fake of this person": the face in clip 1 was swapped, changed or made by a
      computer to look like the person in clip 2
    - "fake of a different person": the face in clip 1 was swapped, changed or made by
      a computer, and looks like someone else

    Options: {options}

    Reply in JSON only: {"answer": "<one of the options, copied exactly>"}

and for a voice question:

    Above are the soundtracks of clip 1 and clip 2, in that order. There are no images.

    {question}

    What each option means:   (as for faces, with "voice", "converted" and "sound like")

    Options: {options}

    Reply in JSON only: {"answer": "<one of the options, copied exactly>"}

The question text comes from the question file. Options are shuffled per question in
the question file, and the descriptions follow the same order.

## T4 prompt

32 frames and the soundtrack of the clip alone: no facts and no reference clip (those
are for the T4 annotator). Then:

    The images are {n} frames sampled evenly from one video clip, in time order. The audio is that clip's soundtrack.

    Some clips are real recordings; normal processing such as compression does not count as AI
    generation. In others, AI generated or altered the video, the audio, or both.

    Explain the video and the audio of this clip, each on its own, in 3 to 5 full sentences per part:
    - video: say clearly whether the video is AI-generated: if so, what was generated; if not, that it
      is a natural recording, with nothing AI-generated.
    - audio: say clearly whether the audio is AI-generated: if so, what was generated or altered; if
      not, that it is a natural recording, with nothing AI-generated.
    Describe what is on screen (who is shown, where, what they do) and what is heard (who speaks and
    what about), and name any visible sign of manipulation.

    What can be generated or altered:
    - inner face: only the inner face (eyes, nose, mouth, skin) is AI-generated; the hair, head, body
      and background are untouched
    - whole frame: the whole frame, person and background, is AI-generated
    - voice: the voice is AI-altered

    Reply in JSON only, each value a paragraph: {"video": "<3 to 5 sentences>", "audio": "<3 to 5 sentences>"}

The answer has the shape of the reference: gt_target.sections holds one paragraph per
part, and a real part describes the clip and ends with the sentence the prompt asks for. The descriptions are
keyed by region and listed in the question's options' order, without "none"; a region
without a description fails at import. parse() returns {"video", "audio"} from the first JSON
object in the reply that has both (text or stray characters around it are ignored), or None if
either is missing or empty. A reply that is not JSON as written is read again with a comma before
"}" dropped and a quote inside a value escaped; predictions.py turns that into pred_answer (the two joined, as
gt_answer) and pred_target.sections. Run T4 on t4_test.jsonl only:
t4_golden.jsonl is for human review, and run_inference refuses it and the annotator, verifier and rewriter models.
