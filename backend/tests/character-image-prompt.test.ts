import assert from 'node:assert/strict'
import test from 'node:test'

import { buildCharacterImagePrompt } from '../src/services/character-image-prompt.js'

test('builds a character sheet prompt with portrait and three-view layout', () => {
  const prompt = buildCharacterImagePrompt({
    name: 'Chen Ji',
    role: 'young male lead',
    appearance: 'black messy hair, black blazer, black shirt, calm expression',
    description: 'quiet intern at Qingshan hospital',
    personality: 'reserved and observant',
  })

  assert.match(prompt, /character model sheet/i)
  assert.match(prompt, /left panel/i)
  assert.match(prompt, /front-facing close-up portrait/i)
  assert.match(prompt, /right panel/i)
  assert.match(prompt, /full-body front view/i)
  assert.match(prompt, /full-body side view/i)
  assert.match(prompt, /full-body back view/i)
  assert.match(prompt, /same person/i)
  assert.match(prompt, /same outfit/i)
  assert.match(prompt, /white studio background/i)
  assert.match(prompt, /no text/i)
  assert.match(prompt, /no watermark/i)
  assert.doesNotMatch(prompt, /only a headshot/i)
})

test('spells out the four-region reference board layout strongly enough for image generators', () => {
  const prompt = buildCharacterImagePrompt({
    name: 'Chen Ji',
    appearance: 'young man, black blazer, black shirt',
  })

  assert.match(prompt, /four distinct visual regions/i)
  assert.match(prompt, /left 35%/i)
  assert.match(prompt, /right 65%/i)
  assert.match(prompt, /Region 1/i)
  assert.match(prompt, /Region 2/i)
  assert.match(prompt, /Region 3/i)
  assert.match(prompt, /Region 4/i)
  assert.match(prompt, /must not be a single portrait/i)
})
