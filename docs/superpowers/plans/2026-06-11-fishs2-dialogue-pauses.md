# FishS2 Dialogue Pause Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce FishS2 multi-speaker dialogue gaps by formatting prompts with blank lines and injecting a shorter ComfyUI pause parameter.

**Architecture:** Keep workflow JSON files unchanged. Add runtime behavior in the ComfyUI TTS adapter so only `FishS2MultiSpeakerTTS` nodes get `pause_after_speaker = 0.25`, and only S2/Fish multi-speaker requests format speaker turns with blank lines.

**Tech Stack:** TypeScript, Node test runner, ComfyUI workflow adapter.

---

### Task 1: Adapter Tests

**Files:**
- Modify: `backend/tests/comfyui-tts.test.ts`

- [ ] **Step 1: Write failing tests**

Add assertions for FishS2 pause injection and FishS2 prompt formatting.

- [ ] **Step 2: Run red verification**

Run: `cd backend && npx.cmd tsx --test tests/comfyui-tts.test.ts`

Expected: fail because `pause_after_speaker` is still `0.4` and prompt text currently has no blank line between speaker turns.

### Task 2: Adapter Implementation

**Files:**
- Modify: `backend/src/services/adapters/comfyui-tts.ts`

- [ ] **Step 1: Add FishS2 constants and helpers**

Define `FISHS2_DIALOGUE_PAUSE_AFTER_SPEAKER = 0.25`, detect `FishS2MultiSpeakerTTS`, and format FishS2 multi-speaker turns with `\n\n`.

- [ ] **Step 2: Inject pause after model path injection**

Call the pause injection helper from `resolveComfyUiAudioWorkflow()` before pruning unreachable nodes.

- [ ] **Step 3: Run green verification**

Run: `cd backend && npx.cmd tsx --test tests/comfyui-tts.test.ts`

Expected: all tests in the adapter file pass.
