import assert from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { chmod, mkdtemp, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

async function waitForImageRecord(db: any, schema: any, id: number) {
  for (let i = 0; i < 50; i++) {
    const [record] = db.select().from(schema.imageGenerations).all()
      .filter((row: any) => row.id === id)
    if (record && record.status !== 'processing') return record
    await new Promise(resolve => setTimeout(resolve, 100))
  }
  const [record] = db.select().from(schema.imageGenerations).all()
    .filter((row: any) => row.id === id)
  return record
}

test('uses local Codex image generation when no image config exists', async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'huobao-image-default-codex-'))
  const originalDbPath = process.env.DB_PATH
  const originalStoragePath = process.env.STORAGE_PATH
  const originalCodexBin = process.env.CODEX_BIN

  process.env.DB_PATH = path.join(tempRoot, 'huobao.db')
  process.env.STORAGE_PATH = path.join(tempRoot, 'storage')

  const fakeScript = path.join(tempRoot, 'fake-codex.mjs')
  const fakeBin = process.platform === 'win32'
    ? path.join(tempRoot, 'fake-codex.cmd')
    : path.join(tempRoot, 'fake-codex')
  const png1x1 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII='

  await writeFile(fakeScript, `
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'

const args = process.argv.slice(2)
const outputPath = args[args.indexOf('--output-last-message') + 1]
let stdin = ''
for await (const chunk of process.stdin) stdin += chunk
const promptArg = args[args.length - 1] || ''
const prompt = promptArg === '-' ? stdin : promptArg
const outputMatch = prompt.match(/OUTPUT_PATH:\\s*(.+)/)
if (!outputMatch) throw new Error('missing OUTPUT_PATH')
const imagePath = outputMatch[1].trim()
await mkdir(path.dirname(imagePath), { recursive: true })
await writeFile(imagePath, Buffer.from('${png1x1}', 'base64'))
await writeFile(outputPath, JSON.stringify({
  ok: true,
  local_path: imagePath,
  prompt_used: 'fake codex prompt',
  error: '',
}), 'utf8')
`, 'utf8')

  if (process.platform === 'win32') {
    await writeFile(fakeBin, `@echo off\r\n"${process.execPath}" "${fakeScript}" %*\r\n`, 'utf8')
  } else {
    await writeFile(fakeBin, `#!/usr/bin/env sh\nexec "${process.execPath}" "${fakeScript}" "$@"\n`, 'utf8')
    await chmod(fakeBin, 0o755)
  }

  process.env.CODEX_BIN = fakeBin

  try {
    const { db, schema } = await import('../src/db/index.js')
    const { generateImage } = await import('../src/services/image-generation.js')

    const id = await generateImage({
      prompt: 'fallback codex image',
      size: '16x16',
    })
    const record = await waitForImageRecord(db, schema, id)

    assert.equal(record.provider, 'codex')
    assert.equal(record.model, 'gpt-5.5')
    assert.equal(record.status, 'completed')
    assert.match(record.localPath, /^static\/images\/.+\.png$/)
    assert.equal(
      existsSync(path.join(process.env.STORAGE_PATH, record.localPath.replace(/^static\//, ''))),
      true,
    )
  } finally {
    if (originalDbPath === undefined) delete process.env.DB_PATH
    else process.env.DB_PATH = originalDbPath
    if (originalStoragePath === undefined) delete process.env.STORAGE_PATH
    else process.env.STORAGE_PATH = originalStoragePath
    if (originalCodexBin === undefined) delete process.env.CODEX_BIN
    else process.env.CODEX_BIN = originalCodexBin
    try {
      await rm(tempRoot, { recursive: true, force: true })
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EBUSY') throw error
    }
  }
})
