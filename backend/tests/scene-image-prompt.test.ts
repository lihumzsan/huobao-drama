import assert from 'node:assert/strict'
import test from 'node:test'

import { buildSceneImagePrompt, buildSceneStoragePrompt } from '../src/services/scene-image-prompt.js'

test('builds scene image prompts as pure background assets without character action', () => {
  const prompt = buildSceneImagePrompt({
    location: 'Qingshan hospital consultation room',
    time: 'autumn night',
    prompt: 'Doctor sits across from a young patient, tense conversation at a desk.',
  })

  assert.match(prompt, /pure background scene/i)
  assert.match(prompt, /Qingshan hospital consultation room/i)
  assert.match(prompt, /autumn night/i)
  assert.match(prompt, /no people/i)
  assert.match(prompt, /no characters/i)
  assert.match(prompt, /no portraits/i)
  assert.doesNotMatch(prompt, /Doctor sits/i)
  assert.doesNotMatch(prompt, /patient/i)
  assert.doesNotMatch(prompt, /conversation/i)
})

test('normalizes scene storage prompts to background-only image generation prompts', () => {
  const prompt = buildSceneStoragePrompt({
    location: 'Qingshan hospital corridor',
    time: 'autumn night',
    prompt: 'A stern doctor hands documents to a man in the hallway.',
  })

  assert.match(prompt, /pure background scene/i)
  assert.match(prompt, /Qingshan hospital corridor/i)
  assert.match(prompt, /empty environment/i)
  assert.match(prompt, /no people/i)
  assert.doesNotMatch(prompt, /doctor/i)
  assert.doesNotMatch(prompt, /man/i)
  assert.doesNotMatch(prompt, /hands documents/i)
})
