export const MIN_STORYBOARD_DURATION_SECONDS = 1
export const MAX_STORYBOARD_DURATION_SECONDS = 10
export const DEFAULT_STORYBOARD_DURATION_SECONDS = 6
export const DEFAULT_VIDEO_DURATION_SECONDS = 5

export function normalizeVideoDuration(
  duration: number | null | undefined,
  options: { min?: number; max?: number; fallback?: number } = {},
) {
  const min = options.min ?? MIN_STORYBOARD_DURATION_SECONDS
  const max = options.max ?? MAX_STORYBOARD_DURATION_SECONDS
  const fallback = options.fallback ?? DEFAULT_VIDEO_DURATION_SECONDS
  const parsed = Math.round(Number(duration ?? fallback))
  const value = Number.isFinite(parsed) ? parsed : fallback
  return Math.min(max, Math.max(min, value))
}

export function requireStoryboardDuration(duration: number | null | undefined, label = 'duration') {
  const parsed = Math.round(Number(duration ?? DEFAULT_STORYBOARD_DURATION_SECONDS))
  if (!Number.isFinite(parsed)) {
    throw new Error(`${label} must be a number between ${MIN_STORYBOARD_DURATION_SECONDS} and ${MAX_STORYBOARD_DURATION_SECONDS} seconds`)
  }
  if (parsed < MIN_STORYBOARD_DURATION_SECONDS || parsed > MAX_STORYBOARD_DURATION_SECONDS) {
    throw new Error(`${label} must be between ${MIN_STORYBOARD_DURATION_SECONDS} and ${MAX_STORYBOARD_DURATION_SECONDS} seconds`)
  }
  return parsed
}

export function assertStoryboardDurationRhythm<T extends { duration?: number | null }>(storyboards: T[]) {
  const durations = storyboards.map((storyboard, index) => requireStoryboardDuration(storyboard.duration, `storyboards[${index}].duration`))
  if (durations.length > 3 && new Set(durations).size === 1) {
    throw new Error('Storyboard durations must follow the plot rhythm; do not assign the same duration to every shot.')
  }
  return durations
}
