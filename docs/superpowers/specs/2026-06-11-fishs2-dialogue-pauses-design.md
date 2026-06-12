# FishS2 Dialogue Pause Design

## Goal

Make FishS2 multi-speaker TTS dialogue sound tighter by reducing the pause between speaker turns while keeping speaker labels valid for the ComfyUI workflow.

## Current Behavior

Multi-speaker reference-audio dialogue with an S2/Fish model resolves to `baseaudio/多人/s2-two` for two speakers or `baseaudio/三人/s2-three` for three speakers. Both workflows use `FishS2MultiSpeakerTTS`, whose `pause_after_speaker` input currently comes from the workflow default of `0.4`.

The adapter currently formats prompt text as adjacent lines:

```text
[speaker_1]: first line
[speaker_2]: second line
```

## Design

For FishS2 multi-speaker workflows only:

- Keep the normal `[speaker_n]: text` label format.
- Join speaker turns with a blank line so the prompt remains readable and dialogue-like.
- Inject `pause_after_speaker = 0.25` into `FishS2MultiSpeakerTTS` at runtime instead of editing workflow JSON files.

Single-speaker FishS2, pure text FishS2, and LongCat workflows stay unchanged.

## Testing

Add adapter tests that verify:

- FishS2 multi-speaker workflows receive `pause_after_speaker = 0.25`.
- Generated ComfyUI prompts for FishS2 multi-speaker requests contain a blank line between speaker turns.
- Existing workflow selection and reference-audio injection behavior still passes.
