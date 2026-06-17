import assert from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

test('refreshes completed ComfyUI video records from persisted task ids', async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'huobao-video-refresh-comfyui-'))
  const originalDbPath = process.env.DB_PATH
  const originalStoragePath = process.env.STORAGE_PATH
  const originalComfyVideoBaseUrl = process.env.COMFYUI_VIDEO_BASE_URL
  const originalComfyBaseUrl = process.env.COMFYUI_BASE_URL
  const originalFetch = globalThis.fetch

  process.env.DB_PATH = path.join(tempRoot, 'huobao.db')
  process.env.STORAGE_PATH = path.join(tempRoot, 'static-root')
  process.env.COMFYUI_VIDEO_BASE_URL = 'http://127.0.0.1:8878'
  delete process.env.COMFYUI_BASE_URL

  globalThis.fetch = (async (input: string | URL | Request) => {
    const url = input instanceof Request ? input.url : String(input)

    if (url === 'http://127.0.0.1:8878/history/prompt-video-1') {
      return new Response(JSON.stringify({
        promptVideo: {
          status: { status_str: 'success', completed: true },
          outputs: {
            '375': {
              images: [
                { filename: 'huobao-bernini-4_00001_.mp4', subfolder: 'video', type: 'output' },
              ],
              animated: [true],
            },
          },
        },
      }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }

    if (url === 'http://127.0.0.1:8878/view?filename=huobao-bernini-4_00001_.mp4&subfolder=video&type=output') {
      return new Response(Buffer.from([1, 2, 3, 4]), {
        status: 200,
        headers: { 'content-type': 'video/mp4' },
      })
    }

    throw new Error(`Unexpected fetch: ${url}`)
  }) as typeof fetch

  try {
    const { db, schema } = await import('../src/db/index.js')
    const videoGeneration = await import('../src/services/video-generation.js')
    assert.equal(typeof videoGeneration.refreshVideoGenerationStatus, 'function')

    const ts = new Date().toISOString()
    const storyboardRes = db.insert(schema.storyboards).values({
      episodeId: 1,
      storyboardNumber: 1,
      description: 'Opening shot',
      duration: 4,
      createdAt: ts,
      updatedAt: ts,
    }).run()

    const videoRes = db.insert(schema.videoGenerations).values({
      storyboardId: Number(storyboardRes.lastInsertRowid),
      dramaId: 1,
      provider: 'comfyui',
      prompt: 'Opening shot prompt',
      model: 'basevideo/Seedance2.0_Bernini_01_480p_10s',
      referenceMode: 'single',
      status: 'processing',
      taskId: 'prompt-video-1',
      duration: 4,
      createdAt: ts,
      updatedAt: ts,
    }).run()

    const refreshed = await videoGeneration.refreshVideoGenerationStatus(Number(videoRes.lastInsertRowid))
    assert.equal(refreshed.status, 'completed')
    assert.match(refreshed.localPath, /^static\/videos\/.+\.bin$/)
    assert.equal(existsSync(path.join(process.env.STORAGE_PATH, refreshed.localPath.replace(/^static\//, ''))), true)

    const [storyboard] = db.select().from(schema.storyboards).all()
    assert.equal(storyboard.videoUrl, refreshed.localPath)
    assert.equal(storyboard.duration, 4)
  } finally {
    globalThis.fetch = originalFetch
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
