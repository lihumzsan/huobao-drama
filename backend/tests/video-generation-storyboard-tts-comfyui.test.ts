import assert from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

const AUDIO_LIPSYNC_WORKFLOW = 'basevideo/Seedance2.0_Bernini_01+02+04_480p_10s_图生视频出音效对口型'

async function waitForVideoRecord(db: any, schema: any, id: number) {
  for (let i = 0; i < 50; i++) {
    const [record] = db.select().from(schema.videoGenerations).all()
      .filter((row: any) => row.id === id)
    if (record && record.status !== 'processing') return record
    await new Promise(resolve => setTimeout(resolve, 100))
  }
  const [record] = db.select().from(schema.videoGenerations).all()
    .filter((row: any) => row.id === id)
  return record
}

function containsValue(value: unknown, expected: string): boolean {
  if (value === expected) return true
  if (typeof value === 'string') return value.includes(expected)
  if (Array.isArray(value)) return value.some(item => containsValue(item, expected))
  if (value && typeof value === 'object') {
    return Object.values(value).some(item => containsValue(item, expected))
  }
  return false
}

test('uses storyboard TTS audio for ComfyUI audio lipsync video generation', async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'huobao-video-storyboard-tts-comfyui-'))
  const originalDbPath = process.env.DB_PATH
  const originalStoragePath = process.env.STORAGE_PATH
  const originalComfyVideoBaseUrl = process.env.COMFYUI_VIDEO_BASE_URL
  const originalComfyBaseUrl = process.env.COMFYUI_BASE_URL
  const originalFetch = globalThis.fetch
  const originalSetTimeout = globalThis.setTimeout
  const promptText = 'The doctor speaks calmly to the patient.'
  const audioUrl = 'data:audio/wav;base64,UklGRg=='
  let uploadCount = 0
  let promptPosted = false

  process.env.DB_PATH = path.join(tempRoot, 'huobao.db')
  process.env.STORAGE_PATH = path.join(tempRoot, 'static-root')
  process.env.COMFYUI_VIDEO_BASE_URL = 'http://127.0.0.1:8878'
  delete process.env.COMFYUI_BASE_URL
  globalThis.setTimeout = ((handler: TimerHandler, timeout?: number, ...args: any[]) => {
    return originalSetTimeout(handler, Math.min(Number(timeout) || 0, 1), ...args)
  }) as typeof setTimeout

  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = input instanceof Request ? input.url : String(input)

    if (url === 'http://127.0.0.1:8878/upload/image') {
      uploadCount += 1
      assert.equal(init?.method, 'POST')
      return new Response(JSON.stringify({
        name: uploadCount === 1 ? 'huobao-ref.png' : 'huobao-voice.wav',
      }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }

    if (url === 'http://127.0.0.1:8878/prompt') {
      promptPosted = true
      const body = JSON.parse(String(init?.body || '{}'))
      assert.equal(init?.method, 'POST')
      assert.ok(containsValue(body.prompt, promptText), 'ComfyUI workflow should include the storyboard video prompt')
      assert.ok(containsValue(body.prompt, 'huobao-ref.png'), 'ComfyUI workflow should include the uploaded reference image')
      assert.ok(containsValue(body.prompt, 'huobao-voice.wav'), 'ComfyUI workflow should include the uploaded TTS audio')
      assert.ok(Object.values(body.prompt).some((node: any) => {
        return node.class_type === 'VHS_VideoCombine'
          && String(node._meta?.title || '').includes('最终视频')
      }), 'ComfyUI workflow should retain the final VHS output node')
      return new Response(JSON.stringify({ prompt_id: 'prompt-video-audio-1' }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }

    if (url === 'http://127.0.0.1:8878/history/prompt-video-audio-1') {
      return new Response(JSON.stringify({
        promptVideo: {
          status: { status_str: 'success', completed: true },
          outputs: {
            '1503': {
              filenames: [
                { filename: 'Bernini-T10-lipsync_00001.mp4', subfolder: 'video', type: 'output' },
              ],
            },
          },
        },
      }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }

    if (url === 'http://127.0.0.1:8878/view?filename=Bernini-T10-lipsync_00001.mp4&subfolder=video&type=output') {
      return new Response(Buffer.from([1, 2, 3, 4]), {
        status: 200,
        headers: { 'content-type': 'video/mp4' },
      })
    }

    throw new Error(`Unexpected fetch: ${url}`)
  }) as typeof fetch

  try {
    const { db, schema } = await import('../src/db/index.js')
    const { generateVideo } = await import('../src/services/video-generation.js')

    const ts = new Date().toISOString()
    const storyboardRes = db.insert(schema.storyboards).values({
      episodeId: 1,
      storyboardNumber: 1,
      description: 'Dialogue shot',
      duration: 4,
      ttsAudioUrl: audioUrl,
      createdAt: ts,
      updatedAt: ts,
    }).run()

    const id = await generateVideo({
      storyboardId: Number(storyboardRes.lastInsertRowid),
      dramaId: 1,
      prompt: promptText,
      imageUrl: 'data:image/png;base64,iVBORw0KGgo=',
      duration: 4,
    })
    const record = await waitForVideoRecord(db, schema, id)

    assert.equal(uploadCount, 2)
    assert.equal(promptPosted, true)
    assert.equal(record.provider, 'comfyui')
    assert.equal(record.model, AUDIO_LIPSYNC_WORKFLOW)
    assert.equal(record.audioUrl, audioUrl)
    assert.equal(record.status, 'completed')
    assert.match(record.localPath, /^static\/videos\/.+\.bin$/)
    assert.equal(
      existsSync(path.join(process.env.STORAGE_PATH, record.localPath.replace(/^static\//, ''))),
      true,
    )
  } finally {
    globalThis.fetch = originalFetch
    globalThis.setTimeout = originalSetTimeout
    if (originalDbPath === undefined) delete process.env.DB_PATH
    else process.env.DB_PATH = originalDbPath
    if (originalStoragePath === undefined) delete process.env.STORAGE_PATH
    else process.env.STORAGE_PATH = originalStoragePath
    if (originalComfyVideoBaseUrl === undefined) delete process.env.COMFYUI_VIDEO_BASE_URL
    else process.env.COMFYUI_VIDEO_BASE_URL = originalComfyVideoBaseUrl
    if (originalComfyBaseUrl === undefined) delete process.env.COMFYUI_BASE_URL
    else process.env.COMFYUI_BASE_URL = originalComfyBaseUrl
    try {
      await rm(tempRoot, { recursive: true, force: true })
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EBUSY') throw error
    }
  }
})
