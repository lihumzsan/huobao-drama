# TTS Regenerate Design

## Goal

Add a clear "重新生成" action for storyboard dubbing items that already have generated audio.

## Approved Behavior

- Storyboards without generated TTS keep showing "生成配音".
- Storyboards with generated TTS show "重新生成".
- Pending storyboard TTS jobs keep showing "生成中" and the button stays disabled.
- Clicking "重新生成" calls the existing `POST /storyboards/:id/generate-tts` endpoint.
- The backend already overwrites `tts_audio_url` with the newly generated file path, so no new endpoint or history model is needed.
- Batch generation remains scoped to missing audio only.

## Implementation Notes

Put the button-label decision in `frontend/app/utils/ttsGenerationStatus.ts` so it can be tested independently. The episode page will call that helper from its existing `ttsButtonText` wrapper.

## Testing

Extend `frontend/tests/tts-generation-status.test.ts` with focused assertions for:

- no audio returns "生成配音";
- existing audio returns "重新生成";
- pending job returns "生成中" even if an audio URL exists.
