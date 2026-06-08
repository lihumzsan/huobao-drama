# Upload Role Voice Design

## Goal

Allow a user to upload an audio file for a character and use that uploaded file as the character's actual TTS voice reference. The uploaded audio should affect both the character voice sample generation and later storyboard/dialogue audio generation.

## Confirmed Behavior

- Uploaded audio is not only a playable sample. It becomes the character voice.
- After upload, the character is considered assigned because `voice_style` points to the uploaded audio path.
- Existing generated sample audio is cleared when the role voice changes.
- The user can immediately generate a new voice sample to verify the uploaded voice.
- Later storyboard TTS and compose flows reuse the same `voice_style`, so they also use the uploaded voice.

## Recommended Approach

Reuse the existing character voice fields instead of creating a new custom voice library.

The backend already stores each character voice in `characters.voice_style`, and the ComfyUI TTS adapter already treats local audio paths such as `static/...mp3` as reference audio sources. Uploading a file can therefore be implemented as a narrow upload endpoint plus a frontend action that updates the character's `voice_style`.

## Backend Design

Add `POST /api/v1/upload/audio`.

Request:

- `multipart/form-data`
- field: `file`
- accepted extensions: `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.ogg`, `.opus`, `.webm`
- accepted MIME prefix: `audio/`, with extension fallback for browsers that send `application/octet-stream`

Response:

```json
{
  "path": "static/voice-uploads/<uuid>.<ext>",
  "url": "/static/voice-uploads/<uuid>.<ext>"
}
```

The endpoint will use the existing local storage helper and save files under `static/voice-uploads`. It will reject missing files, unsupported types, and oversized files. The intended limit is 30 MB, which is enough for short reference voice clips without allowing large accidental uploads.

No database schema change is required.

## Frontend Design

In the voice assignment card for each character:

- Add an `Upload voice` button near the existing sample generation button.
- Use a hidden file input per character or a shared hidden input with the selected character id.
- On file selection, call `POST /upload/audio`.
- On upload success, call `PUT /characters/:id` with:

```json
{
  "voice_style": "static/voice-uploads/<uuid>.<ext>",
  "voice_provider": "comfyui"
}
```

Then update local character state:

- `voice_style` and `voiceStyle` become the uploaded path.
- `voice_provider` and `voiceProvider` become `comfyui`.
- `voice_sample_url` and `voiceSampleUrl` are cleared.

The selected voice display should show a compact custom voice state when the value is an audio path:

- label: uploaded voice
- secondary text: original file name if available in local state, otherwise the stored filename
- action hint: generate a sample to confirm the role voice

## Provider Constraint

Uploaded audio voice references require a TTS provider that supports reference audio. In the current codebase, this is the ComfyUI adapter.

If the effective episode audio provider is not `comfyui`, the frontend should block upload and show a clear toast asking the user to switch the episode audio configuration to ComfyUI. This avoids saving a reference-audio path that a plain voice-id provider cannot use.

## Data Flow

1. User chooses an audio file on a character card.
2. Frontend validates that the effective audio provider is `comfyui`.
3. Frontend uploads the file to `/api/v1/upload/audio`.
4. Backend validates and stores the file under `static/voice-uploads`.
5. Frontend updates the character's `voice_style` to the returned path.
6. Generating a voice sample calls the existing character sample endpoint.
7. Existing TTS generation passes `voice_style` to the ComfyUI adapter.
8. The adapter uploads the local reference audio to ComfyUI and runs the voice-clone workflow.

## Error Handling

- Missing file: return `400 file is required`.
- Unsupported audio type: return `400 unsupported audio type`.
- Oversized file: return `400 audio file is too large`.
- Upload network failure: show a toast and leave the existing character voice unchanged.
- Character update failure after upload: show a toast. The uploaded static file may remain unused, which is acceptable for this scoped change because the project already stores generated static assets without immediate garbage collection.

## Testing Plan

Backend:

- Add a focused route/storage test if the current test setup can exercise Hono upload handlers directly.
- At minimum, run backend typecheck and existing TTS tests after implementation.

Frontend:

- Run the frontend build after changing the page and API composable.
- Manually verify the voice assignment screen can select an audio file, update character state, and still trigger sample generation.

Primary verification commands:

```bash
cd backend && npm run typecheck
cd backend && npx tsx --test tests/comfyui-tts.test.ts tests/tts-generation.test.ts
cd frontend && npm run build
```

## Non-Goals

- No reusable custom voice library.
- No new voice database table.
- No background cleanup for unused uploaded voice files.
- No support for reference audio on providers that only accept catalog voice ids.
