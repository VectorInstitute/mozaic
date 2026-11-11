# Gemini

Gemini through its OpenAI-compatible chat API, or through Google's Interactions API, which is Google's own and
not OpenAI-compatible. Set `base_url` to the public API, or to an OpenAI-compatible endpoint.

    gemini/
    ├── model.py           frames, audio and prompt in one user message
    ├── sound_check.py     2 calls: does the sound of both clips arrive (t3)
    └── requirements.txt

Settings live in evaluation/configs/gemini_3_8_flash/ and gemini_3_1_pro/, one config per task
(t<N>/zeroshot.yaml). The key is read from the env file (GEMINI_API_KEY). Both ways send it as
`Authorization: Bearer <key>`, as an OpenAI-compatible endpoint takes it; Google's own Gemini endpoint may expect it in
`x-goog-api-key` instead.

Two ways in (config `api`):

    chat            the OpenAI-style chat API (default, without `api`): our frames as JPEG images, the sound as a WAV
    interactions    Google's Interactions API (base_url .../v1beta): each clip as one video, the file as it is
                    when the task uses sound (t1), without its sound otherwise (t2, t3 face: the video stream
                    copied, not re-encoded), at fps = frames / the clip's seconds, so Gemini takes 32 frames
                    (video_resolution: low = 64 tokens each, Google's default; high = 256);
                    t3 voice: each sound alone, as a WAV. store: false, so Google keeps nothing. Its usage is
                    also saved under the chat names (prompt_tokens, completion_tokens = answer + thinking, as
                    billed), with served_model, the model that answered as Google names it.

With chat, audio is sent as a file part (data:audio/wav), because some OpenAI-compatible gateways
silently drop input_audio parts. To check the sound arrived, look at prompt_tokens in the
predictions: about 35,104 for a 3 s clip at 32 frames. Around 35,027 means no sound.
With the Interactions API, usage.input_tokens_by_modality shows it: a video is 32 frames x 64 tokens
plus its sound (77 tokens for a 3 s clip, 153 for 6 s); a sound alone is 77 or 153 audio tokens.

For the voice questions of t3, run the sound check first (2 paid calls). It
asks for the words in both sounds of one 77-frame and one 153-frame question and
checks that the 153-frame call has 152 more prompt tokens:

    python -m mozaic.evaluation.models.api.gemini.sound_check --config gemini_3_8_flash --task t3

## Environment

    uv venv .venv_gemini
    source .venv_gemini/bin/activate
    uv pip install -e .
    uv pip install -r src/mozaic/evaluation/models/api/gemini/requirements.txt