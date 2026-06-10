# Upload Role Voice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users upload an audio file on a character card and use it as that character's ComfyUI reference voice for samples and later TTS generation.

**Architecture:** Add a narrow backend audio upload endpoint that saves validated files under `static/voice-uploads`. Add a multipart upload helper in the frontend API module. Extend the existing episode voice assignment page so upload writes the returned static path into the existing `characters.voice_style` field with `voice_provider=comfyui`.

**Tech Stack:** Hono, TypeScript, Node test runner, Nuxt 3/Vue 3, existing local storage helpers.

---

### Task 1: Backend Audio Upload Endpoint

**Files:**
- Modify: `backend/src/routes/upload.ts`
- Create: `backend/tests/upload-audio.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/upload-audio.test.ts`:

```ts
import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

test('audio upload saves supported audio under voice uploads', async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'huobao-upload-'))
  const originalStoragePath = process.env.STORAGE_PATH
  process.env.STORAGE_PATH = tempRoot

  try {
    const { default: upload } = await import('../src/routes/upload.js')
    const form = new FormData()
    form.append('file', new File([new Uint8Array([1, 2, 3])], 'voice.mp3', { type: 'audio/mpeg' }))

    const resp = await upload.request('/audio', { method: 'POST', body: form })
    const json = await resp.json() as any

    assert.equal(resp.status, 200)
    assert.match(json.data.path, /^static\/voice-uploads\/.+\.mp3$/)
    assert.equal(json.data.url, `/${json.data.path}`)
    assert.equal(existsSync(path.join(tempRoot, 'voice-uploads', path.basename(json.data.path))), true)
  } finally {
    if (originalStoragePath === undefined) delete process.env.STORAGE_PATH
    else process.env.STORAGE_PATH = originalStoragePath
    await rm(tempRoot, { recursive: true, force: true })
  }
})

test('audio upload rejects unsupported file types', async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'huobao-upload-'))
  const originalStoragePath = process.env.STORAGE_PATH
  process.env.STORAGE_PATH = tempRoot

  try {
    const { default: upload } = await import('../src/routes/upload.js')
    const form = new FormData()
    form.append('file', new File([new Uint8Array([1, 2, 3])], 'voice.txt', { type: 'text/plain' }))

    const resp = await upload.request('/audio', { method: 'POST', body: form })
    const json = await resp.json() as any

    assert.equal(resp.status, 400)
    assert.match(json.message, /unsupported audio type/)
  } finally {
    if (originalStoragePath === undefined) delete process.env.STORAGE_PATH
    else process.env.STORAGE_PATH = originalStoragePath
    await rm(tempRoot, { recursive: true, force: true })
  }
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && npx tsx --test tests/upload-audio.test.ts`

Expected: fail with a 404 or non-200 response for `/audio`, because the route has not been implemented yet.

- [ ] **Step 3: Implement the minimal backend route**

In `backend/src/routes/upload.ts`:

```ts
import path from 'path'
```

Add constants:

```ts
const AUDIO_MAX_BYTES = 30 * 1024 * 1024
const AUDIO_EXTENSIONS = new Set(['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.opus', '.webm'])
```

Add helpers:

```ts
function isSupportedAudioFile(file: File): boolean {
  const ext = path.extname(file.name || '').toLowerCase()
  const mime = String(file.type || '').toLowerCase()
  return AUDIO_EXTENSIONS.has(ext) && (mime.startsWith('audio/') || mime === 'application/octet-stream' || !mime)
}
```

Add route:

```ts
app.post('/audio', async (c) => {
  const body = await c.req.parseBody()
  const file = body['file']

  if (!file || !(file instanceof File)) return badRequest(c, 'file is required')
  if (!isSupportedAudioFile(file)) return badRequest(c, 'unsupported audio type')
  if (file.size > AUDIO_MAX_BYTES) return badRequest(c, 'audio file is too large')

  const buffer = await file.arrayBuffer()
  const savedPath = await saveUploadedFile(buffer, 'voice-uploads', file.name)
  return success(c, { url: `/${savedPath}`, path: savedPath })
})
```

- [ ] **Step 4: Run backend upload tests**

Run: `cd backend && npx tsx --test tests/upload-audio.test.ts`

Expected: both tests pass.

### Task 2: Frontend Upload API Helper

**Files:**
- Modify: `frontend/app/composables/useApi.ts`

- [ ] **Step 1: Add multipart request helper**

Add a `uploadFile` helper next to `req`:

```ts
async function uploadFile<T = any>(path: string, file: File): Promise<T> {
  const form = new FormData()
  form.append('file', file)
  const resp = await fetch(`${BASE}${path}`, { method: 'POST', body: form })
  const json = await resp.json()
  if (!resp.ok || (json.code && json.code >= 400)) {
    throw new Error(json.message || `${resp.status}`)
  }
  return json.data ?? json
}
```

- [ ] **Step 2: Export upload API**

Add:

```ts
export const uploadAPI = {
  audio: (file: File) => uploadFile<{ path: string; url: string }>('/upload/audio', file),
}
```

### Task 3: Character Card Upload UI and State

**Files:**
- Modify: `frontend/app/pages/drama/[id]/episode/[episodeNumber].vue`

- [ ] **Step 1: Import the upload API**

Change the existing import to include `uploadAPI`.

- [ ] **Step 2: Add upload state helpers**

Add near the existing pending refs:

```js
const pendingVoiceUploadIds = ref([])
const uploadedVoiceNames = ref({})
const roleVoiceAccept = 'audio/*,.mp3,.wav,.m4a,.aac,.flac,.ogg,.opus,.webm'
```

Add helper functions:

```js
function isPendingVoiceUpload(id) {
  return pendingVoiceUploadIds.value.includes(id)
}

function isAudioVoice(value) {
  return /\.(aac|flac|m4a|mp3|ogg|opus|wav|webm)(\?.*)?$/i.test(String(value || ''))
}

function uploadedVoiceLabel(char) {
  const stored = char?.voice_style || char?.voiceStyle || ''
  return uploadedVoiceNames.value[char.id] || decodeURIComponent(String(stored).split('/').pop() || 'uploaded voice')
}
```

- [ ] **Step 3: Add upload handler**

Add:

```js
async function uploadRoleVoice(char, event) {
  const input = event?.target
  const file = input?.files?.[0]
  if (!file) return

  if (String(lockedAudioProvider.value || '').toLowerCase() !== 'comfyui') {
    toast.warning('上传音色需要先把当前集音频配置切换为 ComfyUI')
    input.value = ''
    return
  }

  try {
    if (!isPendingVoiceUpload(char.id)) pendingVoiceUploadIds.value.push(char.id)
    const result = await uploadAPI.audio(file)
    await characterAPI.update(char.id, { voice_style: result.path, voice_provider: 'comfyui' })

    const c = chars.value.find(ch => ch.id === char.id)
    if (c) {
      c.voice_style = result.path
      c.voiceStyle = result.path
      c.voice_provider = 'comfyui'
      c.voiceProvider = 'comfyui'
      c.voice_sample_url = ''
      c.voiceSampleUrl = ''
    }
    uploadedVoiceNames.value = { ...uploadedVoiceNames.value, [char.id]: file.name }
    toast.success('音色已上传')
  } catch (e) {
    toast.error(e.message)
  } finally {
    pendingVoiceUploadIds.value = pendingVoiceUploadIds.value.filter(id => id !== char.id)
    input.value = ''
  }
}
```

- [ ] **Step 4: Render custom voice status**

After the existing voice profile card, add a custom uploaded voice card when `isAudioVoice(c.voice_style || c.voiceStyle)` is true.

- [ ] **Step 5: Add the upload button**

In `.voice-actions-row`, add a label styled like the existing small buttons:

```vue
<label class="btn btn-sm voice-upload-control" :class="{ 'is-disabled': isPendingVoiceUpload(c.id) }">
  上传音色
  <input type="file" :accept="roleVoiceAccept" :disabled="isPendingVoiceUpload(c.id)" @change="uploadRoleVoice(c, $event)" />
</label>
```

### Task 4: Verification

**Files:**
- Verify touched backend and frontend files.

- [ ] **Step 1: Run focused backend tests**

Run: `cd backend && npx tsx --test tests/upload-audio.test.ts tests/comfyui-tts.test.ts tests/tts-generation.test.ts`

Expected: all tests pass.

- [ ] **Step 2: Run backend typecheck**

Run: `cd backend && npm run typecheck`

Expected: exit code 0.

- [ ] **Step 3: Run frontend build**

Run: `cd frontend && npm run build`

Expected: exit code 0.

- [ ] **Step 4: Review changed files**

Run: `git diff -- backend/src/routes/upload.ts backend/tests/upload-audio.test.ts frontend/app/composables/useApi.ts frontend/app/pages/drama/[id]/episode/[episodeNumber].vue`

Expected: diff only includes the upload voice implementation.
