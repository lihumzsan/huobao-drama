import assert from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

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

test('uses default ComfyUI image-to-video generation when no video config exists', async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'huobao-video-default-comfyui-'))
  const originalDbPath = process.env.DB_PATH
  const originalStoragePath = process.env.STORAGE_PATH
  const originalComfyVideoBaseUrl = process.env.COMFYUI_VIDEO_BASE_URL
  const originalComfyBaseUrl = process.env.COMFYUI_BASE_URL
  const originalFetch = globalThis.fetch
  const originalSetTimeout = globalThis.setTimeout
  const promptText = 'A calm character slowly turns toward the camera.'
  let uploadPosted = false
  let promptPosted = false

  process.env.DB_PATH = path.join(tempRoot, 'huobao.db')
  process.env.STORAGE_PATH = path.join(tempRoot, 'static-root')
  delete process.env.COMFYUI_VIDEO_BASE_URL
  delete process.env.COMFYUI_BASE_URL
  globalThis.setTimeout = ((handler: TimerHandler, timeout?: number, ...args: any[]) => {
    return originalSetTimeout(handler, Math.min(Number(timeout) || 0, 1), ...args)
  }) as typeof setTimeout

  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = input instanceof Request ? input.url : String(input)

    if (url === 'http://127.0.0.1:8878/upload/image') {
      uploadPosted = true
      assert.equal(init?.method, 'POST')
      return new Response(JSON.stringify({ name: 'huobao-ref.png' }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }

    if (url === 'http://127.0.0.1:8878/prompt') {
      promptPosted = true
      const body = JSON.parse(String(init?.body || '{}'))
      assert.equal(init?.method, 'POST')
      assert.match(body.client_id, /^huobao-drama-video-/)
      assert.ok(containsValue(body.prompt, promptText), 'ComfyUI workflow should include the video prompt')
      assert.ok(containsValue(body.prompt, 'huobao-ref.png'), 'ComfyUI workflow should include uploaded reference image')
      assert.ok(containsValue(body.prompt, '16:9'), 'ComfyUI workflow should default to 16:9 landscape video')
      return new Response(JSON.stringify({ prompt_id: 'prompt-video-1' }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }

    if (url === 'http://127.0.0.1:8878/history/prompt-video-1') {
      return new Response(JSON.stringify({
        promptVideo: {
          status: { status_str: 'success', completed: true },
          outputs: {
            '375': {
              videos: [
                { filename: 'clip.mp4', subfolder: 'video', type: 'output' },
              ],
            },
          },
        },
      }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }

    if (url === 'http://127.0.0.1:8878/view?filename=clip.mp4&subfolder=video&type=output') {
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

    const id = await generateVideo({
      prompt: promptText,
      imageUrl: 'data:image/png;base64,iVBORw0KGgo=',
      duration: 4,
    })
    const record = await waitForVideoRecord(db, schema, id)

    assert.equal(uploadPosted, true)
    assert.equal(promptPosted, true)
    assert.equal(record.provider, 'comfyui')
    assert.equal(record.model, 'basevideo/Seedance2.0_Bernini_01_480p_10s')
    assert.equal(record.aspectRatio, '16:9')
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
