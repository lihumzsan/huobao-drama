import assert from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

async function readJson(resp: Response): Promise<any> {
  const text = await resp.text()
  try {
    return JSON.parse(text)
  } catch {
    return { message: text }
  }
}

test('audio upload saves supported audio under voice uploads', async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'huobao-upload-'))
  const originalStoragePath = process.env.STORAGE_PATH
  process.env.STORAGE_PATH = tempRoot

  try {
    const { default: upload } = await import('../src/routes/upload.js')
    const form = new FormData()
    form.append('file', new File([new Uint8Array([1, 2, 3])], 'voice.mp3', { type: 'audio/mpeg' }))

    const resp = await upload.request('/audio', { method: 'POST', body: form })
    const json = await readJson(resp)

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
    const json = await readJson(resp)

    assert.equal(resp.status, 400)
    assert.match(json.message, /unsupported audio type/)
  } finally {
    if (originalStoragePath === undefined) delete process.env.STORAGE_PATH
    else process.env.STORAGE_PATH = originalStoragePath
    await rm(tempRoot, { recursive: true, force: true })
  }
})
