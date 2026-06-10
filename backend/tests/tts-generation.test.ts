import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

function containsValue(value: unknown, expected: string): boolean {
  if (value === expected) return true
  if (Array.isArray(value)) return value.some(item => containsValue(item, expected))
  if (value && typeof value === 'object') {
    return Object.values(value).some(item => containsValue(item, expected))
  }
  return false
}

test('falls back to ComfyUI audio and injects prompt text when no audio config exists', async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'huobao-tts-'))
  const originalDbPath = process.env.DB_PATH
  const originalStoragePath = process.env.STORAGE_PATH
  const originalFetch = globalThis.fetch
  const promptText = 'Generate this exact line as voice.'
  let promptPosted = false

  process.env.DB_PATH = path.join(tempRoot, 'huobao.db')
  process.env.STORAGE_PATH = path.join(tempRoot, 'static-root')

  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = input instanceof Request ? input.url : String(input)

    if (url === 'http://127.0.0.1:8188/prompt') {
      promptPosted = true
      const body = JSON.parse(String(init?.body || '{}'))
      assert.equal(init?.method, 'POST')
      assert.match(body.client_id, /^huobao-drama-audio-/)
      assert.ok(containsValue(body.prompt, promptText), 'ComfyUI workflow should include generated speech text')
      return new Response(JSON.stringify({ prompt_id: 'prompt-1' }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }

    if (url === 'http://127.0.0.1:8188/history/prompt-1') {
      return new Response(JSON.stringify({
        prompt1: {
          status: { status_str: 'success', completed: true },
          outputs: {
            '9': {
              audio: [
                { filename: 'voice.mp3', subfolder: 'audio', type: 'output' },
              ],
            },
          },
        },
      }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }

    if (url === 'http://127.0.0.1:8188/view?filename=voice.mp3&subfolder=audio&type=output') {
      return new Response(Buffer.from([1, 2, 3, 4]), {
        status: 200,
        headers: { 'content-type': 'audio/mpeg' },
      })
    }

    throw new Error(`Unexpected fetch: ${url}`)
  }) as typeof fetch

  try {
    const { generateTTS } = await import('../src/services/tts-generation.js')

    const result = await generateTTS({ text: promptText, voice: 'alloy' })

    assert.equal(promptPosted, true)
    assert.match(result, /^static\/audio\/.+\.mp3$/)
    assert.equal(existsSync(path.join(process.env.STORAGE_PATH, result.replace(/^static\//, ''))), true)
  } finally {
    globalThis.fetch = originalFetch
    if (originalDbPath === undefined) delete process.env.DB_PATH
    else process.env.DB_PATH = originalDbPath
    if (originalStoragePath === undefined) delete process.env.STORAGE_PATH
    else process.env.STORAGE_PATH = originalStoragePath
    try {
      await rm(tempRoot, { recursive: true, force: true })
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EBUSY') throw error
    }
  }
})
