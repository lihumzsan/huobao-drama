import assert from 'node:assert/strict'
import test from 'node:test'

import {
  formatImageGenerationJob,
  shouldPollImageGenerationJob,
} from '../app/utils/imageGenerationStatus'

test('keeps long local image jobs visibly running beyond one minute', () => {
  const startedAt = Date.parse('2026-06-10T14:31:39.000Z')
  const now = Date.parse('2026-06-10T14:37:39.000Z')

  const label = formatImageGenerationJob({
    generationId: 12,
    status: 'processing',
    startedAt,
    lastCheckedAt: now,
  }, now)

  assert.match(label, /任务 #12/)
  assert.match(label, /运行中/)
  assert.match(label, /6分/)
  assert.equal(shouldPollImageGenerationJob({ status: 'processing', startedAt }, now), true)
})

test('stops polling and surfaces terminal states', () => {
  const startedAt = Date.parse('2026-06-10T14:31:39.000Z')
  const now = Date.parse('2026-06-10T14:37:39.000Z')

  assert.equal(shouldPollImageGenerationJob({ status: 'completed', startedAt }, now), false)
  assert.equal(shouldPollImageGenerationJob({ status: 'failed', startedAt }, now), false)
  assert.equal(
    formatImageGenerationJob({ status: 'failed', startedAt, error: 'provider timeout' }, now),
    '生成失败：provider timeout',
  )
})
