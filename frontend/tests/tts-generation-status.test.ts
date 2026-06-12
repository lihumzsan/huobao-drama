import assert from 'node:assert/strict'
import test from 'node:test'

import {
  ttsActionLabel,
  formatTTSGenerationJob,
  isRecoverableTTSFetchError,
  isPendingTTSJob,
  ttsStatusLabel,
} from '../app/utils/ttsGenerationStatus.ts'

test('marks triggered TTS jobs as visibly generating', () => {
  const startedAt = Date.parse('2026-06-11T10:00:00.000Z')
  const now = Date.parse('2026-06-11T10:02:08.000Z')
  const job = { status: 'processing' as const, startedAt }

  assert.equal(isPendingTTSJob(job), true)
  assert.equal(ttsStatusLabel({ hasAudio: false, job }), '生成中')
  assert.equal(formatTTSGenerationJob(job, now), '已触发，生成中 2分8秒')
})

test('surfaces TTS failures with an inline error state', () => {
  const startedAt = Date.parse('2026-06-11T10:00:00.000Z')
  const job = { status: 'failed' as const, startedAt, error: 'ComfyUI queue failed' }

  assert.equal(isPendingTTSJob(job), false)
  assert.equal(ttsStatusLabel({ hasAudio: false, job }), '失败')
  assert.equal(formatTTSGenerationJob(job, startedAt), '生成失败：ComfyUI queue failed')
})

test('generated audio takes precedence over stale pending state', () => {
  const startedAt = Date.parse('2026-06-11T10:00:00.000Z')

  assert.equal(ttsStatusLabel({
    hasAudio: true,
    job: { status: 'processing', startedAt },
  }), '已生成')
})

test('treats browser fetch failures as recoverable until backend state is checked', () => {
  assert.equal(isRecoverableTTSFetchError(new TypeError('Failed to fetch')), true)
  assert.equal(isRecoverableTTSFetchError(new Error('fetch failed')), true)
  assert.equal(isRecoverableTTSFetchError(new Error('ComfyUI generation failed')), false)
})

test('labels existing TTS audio actions as regenerate', () => {
  const startedAt = Date.parse('2026-06-11T10:00:00.000Z')

  assert.equal(ttsActionLabel({ hasAudio: false, job: null }), '生成配音')
  assert.equal(ttsActionLabel({ hasAudio: true, job: null }), '重新生成')
  assert.equal(ttsActionLabel({
    hasAudio: true,
    job: { status: 'processing', startedAt },
  }), '生成中')
})
