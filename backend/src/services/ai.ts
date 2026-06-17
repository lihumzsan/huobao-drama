/**
 * AI 服务抽象层 — 从数据库配置中获取 provider 和 API key
 */
import { db, schema } from '../db/index.js'
import { eq } from 'drizzle-orm'
import { logTaskProgress, logTaskWarn } from '../utils/task-logger.js'
import { joinProviderUrl } from './adapters/url.js'
import { LOCAL_CODEX_MODEL } from './local-codex-agent.js'

export type ServiceType = 'text' | 'image' | 'video' | 'audio'

export interface AIConfig {
  provider: string
  baseUrl: string
  apiKey: string
  model: string
}

export function getTextProviderBaseUrl(config: AIConfig) {
  const provider = config.provider.toLowerCase()

  if (provider === 'openai' || provider === 'openrouter' || provider === 'chatfire') {
    return joinProviderUrl(config.baseUrl, '/v1', '')
  }

  if (provider === 'volcengine') {
    return joinProviderUrl(config.baseUrl, '/api/v3', '')
  }

  if (provider === 'ali') {
    return joinProviderUrl(config.baseUrl, '/api/v1', '')
  }

  return config.baseUrl
}

export function getActiveConfig(serviceType: ServiceType): AIConfig | null {
  const rows = db.select().from(schema.aiServiceConfigs)
    .where(eq(schema.aiServiceConfigs.serviceType, serviceType))
    .all()
    .filter(r => r.isActive)
    .sort((a, b) => (b.priority || 0) - (a.priority || 0)) // 高优先级优先

  const active = rows[0]
  if (!active) {
    logTaskWarn('AIConfig', 'active-config-missing', { serviceType })
    return null
  }

  const models = active.model ? JSON.parse(active.model) : []
  logTaskProgress('AIConfig', 'active-config-selected', {
    serviceType,
    configId: active.id,
    provider: active.provider,
    model: models[0] || '',
    priority: active.priority,
  })
  return {
    provider: active.provider || '',
    baseUrl: active.baseUrl,
    apiKey: active.apiKey,
    model: models[0] || '',
  }
}

export function getTextConfig(): AIConfig {
  const config = getActiveConfig('text')
  if (!config) throw new Error('No active text AI config')
  return config
}

export function getDefaultCodexImageConfig(): AIConfig {
  return {
    provider: 'codex',
    baseUrl: 'local-codex://image-cli',
    apiKey: '',
    model: LOCAL_CODEX_MODEL,
  }
}

export function getImageConfigById(id?: number | null): AIConfig {
  if (id) {
    const config = getConfigById(id)
    if (config) return config
  }
  const active = getActiveConfig('image')
  if (active) return active

  const fallback = getDefaultCodexImageConfig()
  logTaskWarn('AIConfig', 'image-config-default-codex', {
    provider: fallback.provider,
    baseUrl: fallback.baseUrl,
    model: fallback.model,
  })
  return fallback
}

export function getDefaultComfyUiAudioConfig(): AIConfig {
  return {
    provider: 'comfyui',
    baseUrl: process.env.COMFYUI_AUDIO_BASE_URL || process.env.COMFYUI_BASE_URL || 'http://127.0.0.1:8878',
    apiKey: process.env.COMFYUI_AUDIO_API_KEY || '',
    model: process.env.COMFYUI_AUDIO_MODEL || '',
  }
}

export function getDefaultComfyUiVideoConfig(): AIConfig {
  return {
    provider: 'comfyui',
    baseUrl: process.env.COMFYUI_VIDEO_BASE_URL || process.env.COMFYUI_BASE_URL || 'http://127.0.0.1:8878',
    apiKey: process.env.COMFYUI_VIDEO_API_KEY || '',
    model: process.env.COMFYUI_VIDEO_MODEL || 'basevideo/Seedance2.0_Bernini_01_480p_10s',
  }
}

export function getVideoConfigById(id?: number | null): AIConfig {
  if (id) {
    const config = getConfigById(id)
    if (config) return config
  }
  const active = getActiveConfig('video')
  if (active) return active

  const fallback = getDefaultComfyUiVideoConfig()
  logTaskWarn('AIConfig', 'video-config-default-comfyui', {
    provider: fallback.provider,
    baseUrl: fallback.baseUrl,
    model: fallback.model,
  })
  return fallback
}

export function getAudioConfig(): AIConfig {
  const config = getActiveConfig('audio')
  if (!config) {
    const fallback = getDefaultComfyUiAudioConfig()
    logTaskWarn('AIConfig', 'audio-config-default-comfyui', {
      provider: fallback.provider,
      baseUrl: fallback.baseUrl,
      model: fallback.model,
    })
    return fallback
  }
  return config
}

export function getAudioConfigById(id?: number | null): AIConfig {
  if (id) {
    const config = getConfigById(id)
    if (config) return config
  }
  return getAudioConfig()
}

export function getConfigById(id: number): AIConfig | null {
  const [row] = db.select().from(schema.aiServiceConfigs)
    .where(eq(schema.aiServiceConfigs.id, id)).all()
  if (!row || !row.isActive) {
    logTaskWarn('AIConfig', 'config-by-id-missing', { configId: id })
    return null
  }
  const models = row.model ? JSON.parse(row.model) : []
  logTaskProgress('AIConfig', 'config-by-id-selected', {
    configId: id,
    provider: row.provider,
    model: models[0] || '',
    serviceType: row.serviceType,
  })
  return {
    provider: row.provider || '',
    baseUrl: row.baseUrl,
    apiKey: row.apiKey,
    model: models[0] || '',
  }
}
