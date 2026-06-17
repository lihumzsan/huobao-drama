import assert from 'node:assert/strict'
import test from 'node:test'
import {
  assertStoryboardContextReady,
  assertStoryboardSequence,
  assertStoryboardVideoPromptTiming,
  buildVideoGenerationConstraints,
} from '../src/services/storyboard-validation.js'

test('rejects storyboard numbers that skip or repeat shots', () => {
  assert.throws(
    () => assertStoryboardSequence([
      { shot_number: 1 },
      { shot_number: 3 },
    ]),
    /must be continuous.*missing #2/i,
  )

  assert.throws(
    () => assertStoryboardSequence([
      { shot_number: 1 },
      { shot_number: 1 },
    ]),
    /duplicate #1/i,
  )
})

test('rejects video prompt time ranges that exceed storyboard duration', () => {
  assert.throws(
    () => assertStoryboardVideoPromptTiming([
      {
        shot_number: 4,
        duration: 6,
        video_prompt: '0-3秒：陈迹坐下。<n>3-9秒：医生继续问话。',
      },
    ]),
    /shot #4.*exceeds duration 6s/i,
  )
})

test('accepts Chinese and English prompt time ranges within duration', () => {
  assert.doesNotThrow(() => assertStoryboardVideoPromptTiming([
    {
      shot_number: 1,
      duration: 7,
      video_prompt: '0-3s: doctor looks down.<n>3-7秒：陈迹平静回答。',
    },
  ]))
})

test('requires storyboard context to include linked characters and scenes', () => {
  assert.throws(
    () => assertStoryboardContextReady({ characters: [], scenes: [{ id: 1 }] }),
    /请先完成角色与场景提取/,
  )

  assert.throws(
    () => assertStoryboardContextReady({ characters: [{ id: 1 }], scenes: [] }),
    /请先完成角色与场景提取/,
  )
})

test('builds structured video constraints from the selected model', () => {
  const constraints = buildVideoGenerationConstraints({
    provider: 'comfyui',
    model: 'basevideo/Seedance2.0_Bernini_01_480p_10s',
  })

  assert.equal(constraints.provider, 'comfyui')
  assert.equal(constraints.model, 'basevideo/Seedance2.0_Bernini_01_480p_10s')
  assert.equal(constraints.max_duration_seconds, 10)
  assert.deepEqual(constraints.required_storyboard_duration_range_seconds, { min: 1, max: 10 })
  assert.ok(constraints.prompt_rules.some(rule => rule.includes('duration')))
})
