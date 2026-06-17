import {
  MAX_STORYBOARD_DURATION_SECONDS,
  MIN_STORYBOARD_DURATION_SECONDS,
  requireStoryboardDuration,
} from './duration.js'

type StoryboardNumberInput = {
  shot_number: number
}

type StoryboardTimingInput = {
  shot_number: number
  duration?: number | null
  video_prompt?: string | null
}

type StoryboardContextInput = {
  characters?: unknown[]
  scenes?: unknown[]
}

type VideoConfigInput = {
  provider?: string | null
  model?: string | null
  settings?: string | Record<string, unknown> | null
}

export type VideoGenerationConstraints = {
  provider: string
  model: string
  max_duration_seconds: number
  required_storyboard_duration_range_seconds: { min: number; max: number }
  supports_first_last_frame: boolean
  supports_audio_driven_video: boolean
  prompt_rules: string[]
}

export function assertStoryboardContextReady(context: StoryboardContextInput) {
  if (!context.characters?.length || !context.scenes?.length) {
    throw new Error('请先完成角色与场景提取，再进行分镜拆解。')
  }
}

export function assertStoryboardSequence(storyboards: StoryboardNumberInput[]) {
  const seen = new Set<number>()
  for (const storyboard of storyboards) {
    const shotNumber = Number(storyboard.shot_number)
    if (!Number.isInteger(shotNumber) || shotNumber <= 0) {
      throw new Error(`Storyboard shot_number must be a positive integer, got ${storyboard.shot_number}`)
    }
    if (seen.has(shotNumber)) {
      throw new Error(`Storyboard shot numbers must be unique; duplicate #${shotNumber}`)
    }
    seen.add(shotNumber)
  }

  for (let expected = 1; expected <= storyboards.length; expected += 1) {
    if (!seen.has(expected)) {
      throw new Error(`Storyboard shot numbers must be continuous from #1 to #${storyboards.length}; missing #${expected}`)
    }
  }
}

export function assertStoryboardVideoPromptTiming(storyboards: StoryboardTimingInput[]) {
  for (const storyboard of storyboards) {
    const duration = requireStoryboardDuration(storyboard.duration, `shot #${storyboard.shot_number} duration`)
    const ranges = extractPromptTimeRanges(storyboard.video_prompt || '')
    if (!ranges.length) {
      throw new Error(`Storyboard shot #${storyboard.shot_number} video_prompt must include time ranges such as 0-3s.`)
    }

    for (const range of ranges) {
      if (range.start < 0 || range.end <= range.start) {
        throw new Error(`Storyboard shot #${storyboard.shot_number} has invalid video_prompt time range ${range.start}-${range.end}s.`)
      }
      if (range.end > duration) {
        throw new Error(`Storyboard shot #${storyboard.shot_number} video_prompt range ${range.start}-${range.end}s exceeds duration ${duration}s.`)
      }
    }
  }
}

export function buildVideoGenerationConstraints(config: VideoConfigInput | null | undefined): VideoGenerationConstraints {
  const provider = String(config?.provider || 'unknown').toLowerCase()
  const model = normalizeModelName(config?.model)
  const settings = parseSettings(config?.settings)
  const configuredMax = Number(settings.max_duration_seconds ?? settings.maxDurationSeconds ?? settings.duration)
  const inferredMax = inferMaxDurationSeconds(model)
  const maxDuration = clampDuration(Number.isFinite(configuredMax) && configuredMax > 0 ? configuredMax : inferredMax)

  return {
    provider,
    model,
    max_duration_seconds: maxDuration,
    required_storyboard_duration_range_seconds: {
      min: MIN_STORYBOARD_DURATION_SECONDS,
      max: Math.min(MAX_STORYBOARD_DURATION_SECONDS, maxDuration),
    },
    supports_first_last_frame: supportsFirstLastFrame(provider),
    supports_audio_driven_video: supportsAudioDrivenVideo(provider, model),
    prompt_rules: [
      `Each storyboard duration must be between ${MIN_STORYBOARD_DURATION_SECONDS} and ${Math.min(MAX_STORYBOARD_DURATION_SECONDS, maxDuration)} seconds.`,
      'Every video_prompt must use explicit time ranges that fit inside the storyboard duration.',
      'Do not write time ranges that exceed duration.',
      supportsFirstLastFrame(provider)
        ? 'The video pipeline may use first and last frame references.'
        : 'Prefer a single clear opening frame when the video pipeline does not use first-last references.',
      supportsAudioDrivenVideo(provider, model)
        ? 'If dialogue audio exists, keep mouth movement and action timing compatible with the audio.'
        : 'Do not rely on audio-driven lip sync for this video model.',
    ],
  }
}

function extractPromptTimeRanges(prompt: string) {
  const ranges: Array<{ start: number; end: number }> = []
  const rangePattern = /(\d+(?:\.\d+)?)\s*[-~至到—－]\s*(\d+(?:\.\d+)?)\s*(?:s|秒|sec|seconds)?/gi
  for (const match of prompt.matchAll(rangePattern)) {
    ranges.push({ start: Number(match[1]), end: Number(match[2]) })
  }
  return ranges
}

function normalizeModelName(model: string | null | undefined) {
  const raw = String(model || '').trim()
  if (!raw) return ''
  try {
    const parsed = JSON.parse(raw)
    if (Array.isArray(parsed)) return String(parsed[0] || '').trim()
    if (typeof parsed === 'string') return parsed.trim()
  } catch {
    // Keep raw model names that are not JSON-encoded arrays.
  }
  return raw
}

function parseSettings(settings: VideoConfigInput['settings']): Record<string, unknown> {
  if (!settings) return {}
  if (typeof settings === 'object') return settings
  try {
    const parsed = JSON.parse(settings)
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

function inferMaxDurationSeconds(model: string) {
  const match = model.match(/(?:^|[_\-\s])(\d{1,2})s(?:[_\-\s.]|$)/i)
  if (match) return Number(match[1])
  return MAX_STORYBOARD_DURATION_SECONDS
}

function clampDuration(value: number) {
  const rounded = Math.round(value)
  return Math.min(MAX_STORYBOARD_DURATION_SECONDS, Math.max(MIN_STORYBOARD_DURATION_SECONDS, rounded))
}

function supportsFirstLastFrame(provider: string) {
  return ['comfyui', 'volcengine', 'minimax', 'vidu'].includes(provider)
}

function supportsAudioDrivenVideo(provider: string, model: string) {
  return provider === 'comfyui' || /lip|audio|voice|sound/i.test(model)
}
