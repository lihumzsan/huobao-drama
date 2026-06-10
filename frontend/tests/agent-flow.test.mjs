import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import test from 'node:test'

test('script agent flow awaits saves before starting agents', async () => {
  const source = await readFile(path.resolve('app/pages/drama/[id]/episode/[episodeNumber].vue'), 'utf8')

  assert.match(
    source,
    /async function saveRaw\(\)\s*{[\s\S]*await episodeAPI\.update\(epId\.value, \{ content: localRaw\.value \}\)/,
  )
  assert.match(
    source,
    /async function saveScr\(\)\s*{[\s\S]*await episodeAPI\.update\(epId\.value, \{ script_content: localScript\.value \}\)/,
  )
  assert.match(
    source,
    /async function doRewrite\(\)\s*{[\s\S]*await saveRaw\(\)[\s\S]*await runAgent\('script_rewriter'/,
  )
  assert.match(
    source,
    /async function doExtract\(\)\s*{[\s\S]*await saveScr\(\)[\s\S]*await runAgent\('extractor'/,
  )
})

test('agent composable waits for completion refresh before clearing running state', async () => {
  const source = await readFile(path.resolve('app/composables/useAgent.ts'), 'utf8')

  assert.match(source, /await onDone\?\.\(\)/)
})

test('voice generation stays callable so backend can fall back to ComfyUI audio', async () => {
  const source = await readFile(path.resolve('app/pages/drama/[id]/episode/[episodeNumber].vue'), 'utf8')

  assert.match(source, /const defaultComfyUiAudioConfig = Object\.freeze\(\{[\s\S]*name: '默认 ComfyUI 音频'[\s\S]*provider: 'comfyui'/)
  assert.match(source, /const effectiveAudioConfig = computed/)
  assert.match(source, /const lockedAudioProvider = computed\(\(\) => effectiveAudioConfig\.value\?\.provider \|\| 'comfyui'\)/)
  assert.doesNotMatch(source, /canGenerateAudio/)
  assert.doesNotMatch(source, /ensureAudioConfigReady/)
  assert.doesNotMatch(source, /:disabled="!canGenerateAudio"/)
  assert.match(
    source,
    /async function batchGenSamples\(\) \{[\s\S]*Promise\.allSettled\(pending\.map\(c => characterAPI\.voiceSample/,
  )
  assert.match(
    source,
    /async function genSample\(id\) \{[\s\S]*await characterAPI\.voiceSample/,
  )
  assert.match(
    source,
    /async function genShotTTS\(sb\) \{[\s\S]*await storyboardAPI\.generateTTS/,
  )
  assert.match(
    source,
    /async function batchShotTTS\(\) \{[\s\S]*Promise\.allSettled\(pending\.map\(sb => storyboardAPI\.generateTTS/,
  )
})

test('voice assignment character cards show character and voice images', async () => {
  const source = await readFile(path.resolve('app/pages/drama/[id]/episode/[episodeNumber].vue'), 'utf8')

  assert.match(source, /class="voice-character-profile"[\s\S]*人物形象/)
  assert.match(source, /class="voice-persona-card"[\s\S]*声音形象/)
  assert.match(source, /function voicePersonaText\(char\)/)
  assert.match(source, /function characterSourceText\(char\)[\s\S]*rawContent\.value[\s\S]*scriptContent\.value/)
})
