import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import test from 'node:test'
import sharp from 'sharp'
import {
  CodexImageAdapter,
  buildCodexImagePrompt,
} from '../src/services/adapters/codex-image.js'

test('builds a text-to-image prompt without reference fusion instructions', () => {
  const prompt = buildCodexImagePrompt({
    prompt: '古风庭院，雨夜，电影感',
    frameType: null,
    size: '1024x576',
    referenceCount: 0,
    outputPath: 'C:\\out\\image.png',
  })

  assert.match(prompt, /文生图/)
  assert.match(prompt, /1024x576/)
  assert.match(prompt, /C:\\out\\image\.png/)
  assert.doesNotMatch(prompt, /参考图/)
  assert.doesNotMatch(prompt, /融合/)
})

test('builds a multi-reference fusion prompt with numbered image references', () => {
  const prompt = buildCodexImagePrompt({
    prompt: '角色走进赛博茶馆，保持角色一致',
    frameType: 'first_frame',
    size: '1920x1080',
    referenceCount: 3,
    outputPath: '/tmp/out.png',
  })

  assert.match(prompt, /多图参考生图/)
  assert.match(prompt, /图片1/)
  assert.match(prompt, /图片2/)
  assert.match(prompt, /图片3/)
  assert.match(prompt, /融合/)
  assert.match(prompt, /不要把参考图拼贴/)
})

test('allows panel composition for grid frame types', () => {
  const prompt = buildCodexImagePrompt({
    prompt: '2x2 grid layout, four storyboard frames',
    frameType: 'grid_first_frame_2x2',
    size: '1920x1080',
    referenceCount: 2,
    outputPath: '/tmp/grid.png',
  })

  assert.match(prompt, /宫格/)
  assert.doesNotMatch(prompt, /不要把参考图拼贴/)
})

test('runs Codex image generation through an injectable runner and returns a local static path', async () => {
  const tempDir = await mkdtemp(path.join(tmpdir(), 'huobao-codex-image-test-'))
  const previousStoragePath = process.env.STORAGE_PATH
  process.env.STORAGE_PATH = tempDir

  try {
    const adapter = new CodexImageAdapter(async (options) => {
      assert.equal(options.sandbox, 'danger-full-access')
      assert.deepEqual(options.enabledFeatures, ['image_generation'])
      assert.equal(options.images?.length, 2)
      const outputMatch = options.prompt.match(/OUTPUT_PATH: (.+)/)
      assert.ok(outputMatch, 'prompt should expose OUTPUT_PATH for Codex')
      await sharp({
        create: {
          width: 640,
          height: 640,
          channels: 3,
          background: '#336699',
        },
      }).png().toFile(outputMatch[1].trim())
      return {
        ok: true,
        local_path: outputMatch[1].trim(),
        prompt_used: 'fake prompt',
        error: '',
      }
    })

    const ref = await sharp({
      create: {
        width: 16,
        height: 16,
        channels: 3,
        background: '#ffffff',
      },
    }).png().toBuffer()
    const referenceImages = [
      `data:image/png;base64,${ref.toString('base64')}`,
      `data:image/png;base64,${ref.toString('base64')}`,
    ]

    const result = await adapter.generateLocal({
      provider: 'codex',
      baseUrl: 'local-codex://image-cli',
      apiKey: '',
      model: 'gpt-5.5',
    }, {
      id: 42,
      prompt: 'test image',
      size: '320x180',
      frameType: 'first_frame',
      referenceImages: JSON.stringify(referenceImages),
    })

    assert.match(result.localPath, /^static\/images\/.+\.png$/)
    const metadata = await sharp(path.join(tempDir, result.localPath.replace(/^static\//, ''))).metadata()
    assert.equal(metadata.width, 320)
    assert.equal(metadata.height, 180)
  } finally {
    if (previousStoragePath === undefined) {
      delete process.env.STORAGE_PATH
    } else {
      process.env.STORAGE_PATH = previousStoragePath
    }
    await rm(tempDir, { recursive: true, force: true })
  }
})
