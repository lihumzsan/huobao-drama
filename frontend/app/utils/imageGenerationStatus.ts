export type ImageGenerationJobStatus = 'pending' | 'processing' | 'completed' | 'failed' | string

export type ImageGenerationJob = {
  generationId?: number | null
  status?: ImageGenerationJobStatus | null
  startedAt: number
  lastCheckedAt?: number | null
  error?: string | null
}

const MAX_POLL_MS = 15 * 60 * 1000

function elapsedLabel(startedAt: number, now: number) {
  const elapsedMs = Math.max(0, now - startedAt)
  const totalSeconds = Math.floor(elapsedMs / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  if (minutes > 0) return `${minutes}分${seconds ? `${seconds}秒` : ''}`
  return `${Math.max(1, seconds)}秒`
}

function statusLabel(status?: ImageGenerationJobStatus | null) {
  if (status === 'pending') return '排队中'
  if (status === 'processing') return '运行中'
  if (status === 'completed') return '已生成'
  if (status === 'failed') return '生成失败'
  return '已提交'
}

export function formatImageGenerationJob(job: ImageGenerationJob | null | undefined, now = Date.now()) {
  if (!job) return ''
  if (job.status === 'failed') {
    return job.error ? `生成失败：${job.error}` : '生成失败'
  }
  if (job.status === 'completed') return '已生成'

  const id = job.generationId ? `任务 #${job.generationId}` : '任务已提交'
  return `${id} · ${statusLabel(job.status)} · ${elapsedLabel(job.startedAt, now)}`
}

export function shouldPollImageGenerationJob(job: Pick<ImageGenerationJob, 'status' | 'startedAt'> | null | undefined, now = Date.now()) {
  if (!job) return false
  if (job.status === 'completed' || job.status === 'failed') return false
  return now - job.startedAt < MAX_POLL_MS
}
