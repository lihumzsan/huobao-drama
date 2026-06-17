import assert from 'node:assert/strict'
import test from 'node:test'
import { validateLocalCodexPayload } from '../src/services/local-codex-agent.js'
import { VolcEngineVideoAdapter } from '../src/services/adapters/volcengine-video.js'
import type { AIConfig, VideoGenerationRecord } from '../src/services/adapters/types.js'

function storyboardPayload(duration: number) {
  return {
    storyboards: [
      {
        shot_number: 1,
        title: 'Shot',
        shot_type: 'medium',
        angle: 'eye level',
        movement: 'static',
        location: 'room',
        time: 'night',
        action: 'A character reacts to new information.',
        dialogue: '',
        description: 'A focused reaction shot.',
        result: 'The character decides what to do next.',
        atmosphere: 'tense',
        image_prompt: 'cinematic medium shot in a room at night',
        video_prompt: '0-3s: character listens. 3-6s: character reacts.',
        bgm_prompt: 'low tension',
        sound_effect: 'room tone',
        duration,
        scene_id: null,
        character_ids: [],
      },
    ],
  }
}

test('local Codex storyboard payload rejects shots longer than ten seconds', () => {
  assert.throws(() => {
    validateLocalCodexPayload('storyboard_breaker', storyboardPayload(12))
  })
})

test('VolcEngine video request caps generated video duration at ten seconds', () => {
  const adapter = new VolcEngineVideoAdapter()
  const config: AIConfig = {
    provider: 'volcengine',
    baseUrl: 'https://example.test',
    apiKey: 'test-key',
    model: 'test-model',
  }
  const record: VideoGenerationRecord = {
    id: 1,
    prompt: 'test prompt',
    duration: 12,
    aspectRatio: '16:9',
  }

  const request = adapter.buildGenerateRequest(config, record)

  assert.equal(request.body.duration, 10)
})
