# ComfyUI Audio Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ComfyUI-backed audio generation using the local baseaudio workflows copied into this project.

**Architecture:** Keep the existing `generateTTS()` public contract returning `static/audio/...`. Add a ComfyUI TTS adapter with its own workflow selection, UI-workflow conversion, audio upload, prompt polling, and output download. Register it under `provider=comfyui` while leaving MiniMax behavior untouched.

**Tech Stack:** TypeScript, Node `fetch`/`FormData`, ComfyUI `/prompt`, `/history`, `/view`, and `/upload/image`, Node built-in test runner through `tsx --test`.

---

### Task 1: Copy Workflows

**Files:**
- Create: `configs/comfyui/workflows/baseaudio/**`

- [ ] Copy all JSON files from `D:\workspace\comfui\workflows\baseaudio` into `configs/comfyui/workflows/baseaudio`.
- [ ] Verify the project copy contains `音色/s2-se.json`, `单人/s2-one.json`, `单人/LongCat-one.json`, `多人/s2-two.json`, `多人/LongCat-two.json`, and `三人/s2-three.json`.

### Task 2: Add ComfyUI TTS Unit Tests

**Files:**
- Create: `backend/tests/comfyui-tts.test.ts`

- [ ] Write failing tests for workflow selection, target-text injection, reference-audio injection, audio history parsing, and ComfyUI view URL building.
- [ ] Run `npx tsx --test tests/comfyui-tts.test.ts` from `backend` and confirm the tests fail because the adapter module does not exist yet.

### Task 3: Implement ComfyUI TTS Adapter

**Files:**
- Create: `backend/src/services/adapters/comfyui-tts.ts`
- Modify: `backend/src/services/adapters/types.ts`
- Modify: `backend/src/services/adapters/registry.ts`

- [ ] Implement UI workflow conversion for audio output nodes.
- [ ] Implement `selectComfyUiAudioWorkflow`, `resolveComfyUiAudioWorkflow`, `findFirstOutputAudio`, and `buildComfyUiAudioViewUrl`.
- [ ] Implement `ComfyUiTTSAdapter.generateAudio()` to upload local or URL reference audio when needed, submit `/prompt`, poll `/history`, download `/view`, and return an audio buffer.
- [ ] Register `comfyui` in `ttsAdapters`.
- [ ] Run `npx tsx --test tests/comfyui-tts.test.ts` and make it pass.

### Task 4: Wire Generated Audio Saving

**Files:**
- Modify: `backend/src/services/tts-generation.ts`

- [ ] Extend `generateTTS()` to call adapter-level `generateAudio()` when present.
- [ ] Keep MiniMax response parsing unchanged.
- [ ] Save returned ComfyUI buffers to `static/audio`.
- [ ] Run the ComfyUI TTS tests and backend typecheck.

### Task 5: Final Verification

**Files:**
- Verify all changed backend files.

- [ ] Run `npx tsx --test tests/comfyui-tts.test.ts`.
- [ ] Run `npm run typecheck`.
- [ ] Inspect `git status --short` and report only files changed for this task, without reverting unrelated pre-existing work.
