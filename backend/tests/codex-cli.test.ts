import assert from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { chmod, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { buildCodexExecCommand, createCodexCliErrorMessage, runCodexCliJson } from '../src/services/codex-cli.js'

const baseOptions = {
  schemaPath: 'schema.json',
  outputPath: 'result.json',
}

test('uses CODEX_BIN when explicitly configured', () => {
  const previous = process.env.CODEX_BIN
  process.env.CODEX_BIN = 'C:\\custom\\codex.exe'

  try {
    const command = buildCodexExecCommand(baseOptions)
    assert.equal(command.bin, 'C:\\custom\\codex.exe')
  } finally {
    if (previous === undefined) {
      delete process.env.CODEX_BIN
    } else {
      process.env.CODEX_BIN = previous
    }
  }
})

test('uses the installed Windows Codex CLI before the WindowsApps alias', (t) => {
  const localAppData = process.env.LOCALAPPDATA
  if (process.platform !== 'win32' || !localAppData) {
    t.skip('Windows-only Codex Desktop path')
    return
  }

  const expected = path.join(localAppData, 'OpenAI', 'Codex', 'bin', 'codex.exe')
  if (!existsSync(expected)) {
    t.skip('Codex Desktop CLI is not installed in LOCALAPPDATA')
    return
  }

  const previous = process.env.CODEX_BIN
  delete process.env.CODEX_BIN

  try {
    const command = buildCodexExecCommand(baseOptions)
    assert.equal(command.bin, expected)
  } finally {
    if (previous !== undefined) process.env.CODEX_BIN = previous
  }
})

test('defaults Codex exec to xhigh reasoning and fast service tier', () => {
  const command = buildCodexExecCommand(baseOptions)
  const configValues = command.args.flatMap((arg, index) => arg === '--config' ? [command.args[index + 1]] : [])

  assert.ok(configValues.includes('model_reasoning_effort="xhigh"'))
  assert.ok(configValues.includes('service_tier="fast"'))
})

test('can build a workspace-write Codex exec command with attached images', () => {
  const command = buildCodexExecCommand({
    ...baseOptions,
    sandbox: 'workspace-write',
    images: ['C:\\refs\\scene.png', 'C:\\refs\\character.png'],
  })

  assert.equal(command.args[command.args.indexOf('--sandbox') + 1], 'workspace-write')
  assert.equal(command.args.filter(arg => arg === '--image').length, 2)
  assert.deepEqual(command.args.slice(command.args.indexOf('--image'), command.args.indexOf('--image') + 4), [
    '--image',
    'C:\\refs\\scene.png',
    '--image',
    'C:\\refs\\character.png',
  ])
})

test('can enable Codex CLI features for local image generation', () => {
  const command = buildCodexExecCommand({
    ...baseOptions,
    enabledFeatures: ['image_generation'],
  })

  assert.deepEqual(command.args.slice(command.args.indexOf('--enable'), command.args.indexOf('--enable') + 2), [
    '--enable',
    'image_generation',
  ])
})

test('serializes Codex CLI executions in one backend process', async () => {
  const tempDir = await mkdtemp(path.join(tmpdir(), 'huobao-codex-test-'))
  const fakeScript = path.join(tempDir, 'fake-codex.mjs')
  const fakeBin = process.platform === 'win32'
    ? path.join(tempDir, 'fake-codex.cmd')
    : path.join(tempDir, 'fake-codex')
  const logPath = path.join(tempDir, 'events.log')
  const previousLog = process.env.FAKE_CODEX_LOG
  const previousWait = process.env.FAKE_CODEX_WAIT_MS

  try {
    await writeFile(fakeScript, `
import { appendFile, writeFile } from 'node:fs/promises'

const args = process.argv.slice(2)
const outputPath = args[args.indexOf('--output-last-message') + 1]
await appendFile(process.env.FAKE_CODEX_LOG, 'start:' + process.pid + '\\n')
await new Promise(resolve => setTimeout(resolve, Number(process.env.FAKE_CODEX_WAIT_MS || '100')))
await writeFile(outputPath, JSON.stringify({ ok: true }), 'utf8')
await appendFile(process.env.FAKE_CODEX_LOG, 'end:' + process.pid + '\\n')
`, 'utf8')

    if (process.platform === 'win32') {
      await writeFile(fakeBin, `@echo off\r\n"${process.execPath}" "${fakeScript}" %*\r\n`, 'utf8')
    } else {
      await writeFile(fakeBin, `#!/usr/bin/env sh\nexec "${process.execPath}" "${fakeScript}" "$@"\n`, 'utf8')
      await chmod(fakeBin, 0o755)
    }

    process.env.FAKE_CODEX_LOG = logPath
    process.env.FAKE_CODEX_WAIT_MS = '120'

    const options = {
      cwd: tempDir,
      codexBin: fakeBin,
      schema: {
        type: 'object',
        properties: { ok: { type: 'boolean' } },
        required: ['ok'],
        additionalProperties: false,
      },
      prompt: 'Return ok true',
      timeoutMs: 5_000,
    }

    const [first, second] = await Promise.all([
      runCodexCliJson<{ ok: boolean }>(options),
      runCodexCliJson<{ ok: boolean }>(options),
    ])

    assert.deepEqual(first, { ok: true })
    assert.deepEqual(second, { ok: true })
    const events = (await readFile(logPath, 'utf8')).trim().split(/\r?\n/).map(line => line.split(':')[0])
    assert.deepEqual(events, ['start', 'end', 'start', 'end'])
  } finally {
    if (previousLog === undefined) delete process.env.FAKE_CODEX_LOG
    else process.env.FAKE_CODEX_LOG = previousLog
    if (previousWait === undefined) delete process.env.FAKE_CODEX_WAIT_MS
    else process.env.FAKE_CODEX_WAIT_MS = previousWait
    await rm(tempDir, { recursive: true, force: true })
  }
})

test('keeps the tail of long Codex CLI error output', () => {
  const message = createCodexCliErrorMessage({
    code: 1,
    stdout: '',
    stderr: `early detail\n${'x'.repeat(1500)}\nfinal schema failure`,
  })

  assert.match(message, /early detail/)
  assert.match(message, /final schema failure/)
})
