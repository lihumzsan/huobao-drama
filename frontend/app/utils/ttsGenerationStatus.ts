export type TTSGenerationJobStatus = 'pending' | 'processing' | 'completed' | 'failed' | string

export type TTSGenerationJob = {
  status?: TTSGenerationJobStatus | null
  startedAt: number
  error?: string | null
}

function elapsedLabel(startedAt: number, now: number) {
  const elapsedMs = Math.max(0, now - startedAt)
  const totalSeconds = Math.floor(elapsedMs / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  if (minutes > 0) return `${minutes}分${seconds ? `${seconds}秒` : ''}`
  return `${Math.max(1, seconds)}秒`
}

export function isPendingTTSJob(job: Pick<TTSGenerationJob, 'status'> | null | undefined) {
  return job?.status === 'pending' || job?.status === 'processing'
}

export function ttsStatusLabel(params: { hasAudio: boolean; job?: Pick<TTSGenerationJob, 'status'> | null }) {
  if (params.hasAudio) return '已生成'
  if (params.job?.status === 'failed') return '失败'
  if (isPendingTTSJob(params.job)) return '生成中'
  return '待生成'
}

export function ttsActionLabel(params: { hasAudio: boolean; job?: Pick<TTSGenerationJob, 'status'> | null }) {
  if (isPendingTTSJob(params.job)) return '生成中'
  return params.hasAudio ? '重新生成' : '生成配音'
}

export function formatTTSGenerationJob(job: TTSGenerationJob | null | undefined, now = Date.now()) {
  if (!job) return ''
  if (job.status === 'failed') {
    return job.error ? `生成失败：${job.error}` : '生成失败'
  }
  if (job.status === 'completed') return '已生成'
  return `已触发，生成中 ${elapsedLabel(job.startedAt, now)}`
}

export function isRecoverableTTSFetchError(error: unknown) {
  const message = error instanceof Error ? error.message : String(error || '')
  return /failed to fetch|fetch failed|network\s*error/i.test(message)
}
