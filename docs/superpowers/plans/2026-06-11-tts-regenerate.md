# TTS Regenerate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show "重新生成" for storyboard dubbing items that already have audio, while reusing the existing overwrite behavior.

**Architecture:** Add one pure frontend helper for the TTS button label and consume it from the episode page. The existing backend generation endpoint remains unchanged because it already replaces `tts_audio_url`.

**Tech Stack:** Nuxt 3, Vue 3, TypeScript utilities, Node test runner.

---

### Task 1: Add Regenerate Label Helper

**Files:**
- Modify: `frontend/app/utils/ttsGenerationStatus.ts`
- Test: `frontend/tests/tts-generation-status.test.ts`

- [x] **Step 1: Write the failing test**

Add assertions that `ttsActionLabel` returns:

```ts
assert.equal(ttsActionLabel({ hasAudio: false, job: null }), '生成配音')
assert.equal(ttsActionLabel({ hasAudio: true, job: null }), '重新生成')
assert.equal(ttsActionLabel({
  hasAudio: true,
  job: { status: 'processing', startedAt },
}), '生成中')
```

- [x] **Step 2: Run the targeted frontend test and verify it fails**

Run: `node --test tests/tts-generation-status.test.ts` from `frontend/`.

Expected: FAIL because `ttsActionLabel` is not exported.

- [x] **Step 3: Implement the helper**

Add this export to `frontend/app/utils/ttsGenerationStatus.ts`:

```ts
export function ttsActionLabel(params: { hasAudio: boolean; job?: Pick<TTSGenerationJob, 'status'> | null }) {
  if (isPendingTTSJob(params.job)) return '生成中'
  return params.hasAudio ? '重新生成' : '生成配音'
}
```

- [x] **Step 4: Run the targeted frontend test and verify it passes**

Run: `node --test tests/tts-generation-status.test.ts` from `frontend/`.

Expected: PASS.

### Task 2: Wire Page Button To Helper

**Files:**
- Modify: `frontend/app/pages/drama/[id]/episode/[episodeNumber].vue`
- Test: `frontend/tests/agent-flow.test.mjs`

- [x] **Step 1: Import the helper**

Extend the existing `~/utils/ttsGenerationStatus` import with `ttsActionLabel`.

- [x] **Step 2: Use the helper in `ttsButtonText`**

Replace the inline ternary with:

```js
function ttsButtonText(sb) {
  return ttsActionLabel({ hasAudio: hasTTS(sb), job: getTTSJob(sb?.id) })
}
```

- [x] **Step 3: Run focused frontend tests**

Run these from `frontend/`:

```bash
node --test tests/tts-generation-status.test.ts
node --test tests/agent-flow.test.mjs
```

Expected: both PASS.
