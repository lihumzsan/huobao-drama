import { existsSync } from 'node:fs'
import { copyFile, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { runCodexCliJson, type CodexRunOptions } from '../codex-cli.js'
import { normalizeImageFile, parseDataUrl, reserveStaticFile } from '../../utils/storage.js'
import type {
  AIConfig,
  ImageGenResponse,
  ImageGenerationRecord,
  ImageLocalGenerationResponse,
  ImagePollResponse,
  ImageProviderAdapter,
  ProviderRequest,
} from './types'

type CodexImageRunResult = {
  ok: boolean
  local_path: string
  prompt_used: string
  error: string
}

export type CodexImageRunner = <T = unknown>(options: CodexRunOptions<T>) => Promise<T>

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const PROJECT_ROOT = path.resolve(__dirname, '../../../..')
const MAX_REFERENCE_IMAGES = 6

const CODEX_IMAGE_SCHEMA = {
  type: 'object',
  properties: {
    ok: { type: 'boolean' },
    local_path: { type: 'string' },
    prompt_used: { type: 'string' },
    error: { type: 'string' },
  },
  required: ['ok', 'local_path', 'prompt_used', 'error'],
  additionalProperties: false,
}

export class CodexImageAdapter implements ImageProviderAdapter {
  provider = 'codex'

  constructor(private readonly runCodex: CodexImageRunner = runCodexCliJson) {}

  async generateLocal(config: AIConfig, record: ImageGenerationRecord): Promise<ImageLocalGenerationResponse> {
    const target = reserveStaticFile('images', '.png')
    const refs = parseReferenceImages(record.referenceImages).slice(0, MAX_REFERENCE_IMAGES)
    const tempDir = await mkdtemp(path.join(tmpdir(), 'huobao-codex-image-'))

    try {
      const referencePaths = await Promise.all(refs.map((ref, index) => writeReferenceImage(tempDir, ref, index)))
      const prompt = buildCodexImagePrompt({
        prompt: record.prompt || 'Generate an image',
        frameType: record.frameType,
        size: record.size || '1920x1080',
        referenceCount: referencePaths.length,
        outputPath: target.absolutePath,
      })

      const result = await this.runCodex<CodexImageRunResult>({
        cwd: PROJECT_ROOT,
        prompt,
        schema: CODEX_IMAGE_SCHEMA,
        model: record.model || config.model || 'gpt-5.5',
        reasoningEffort: 'xhigh',
        serviceTier: 'fast',
        sandbox: 'danger-full-access',
        enabledFeatures: ['image_generation'],
        images: referencePaths,
        timeoutMs: 15 * 60 * 1000,
      })

      if (!result.ok) {
        throw new Error(result.error || 'Codex image generation failed')
      }

      const producedPath = result.local_path || target.absolutePath
      if (producedPath !== target.absolutePath) {
        if (!existsSync(producedPath)) throw new Error(`Codex image output not found: ${producedPath}`)
        await copyFile(producedPath, target.absolutePath)
      }

      if (!existsSync(target.absolutePath)) {
        throw new Error('Codex did not write the requested image file')
      }

      await normalizeImageFile(target.absolutePath, record.size)
      return {
        localPath: target.localPath,
        promptUsed: result.prompt_used || prompt,
      }
    } finally {
      await rm(tempDir, { recursive: true, force: true })
    }
  }

  buildGenerateRequest(): ProviderRequest {
    throw new Error('Codex image generation runs locally and does not build HTTP requests')
  }

  parseGenerateResponse(): ImageGenResponse {
    throw new Error('Codex image generation runs locally and does not parse HTTP responses')
  }

  buildPollRequest(): ProviderRequest {
    throw new Error('Codex image generation is synchronous in the local adapter')
  }

  parsePollResponse(): ImagePollResponse {
    return { status: 'failed', error: 'Codex image generation is synchronous in the local adapter' }
  }

  extractImageUrl(): string | null {
    return null
  }

  extractImageBase64(): { data: string; mimeType: string } | null {
    return null
  }
}

export function buildCodexImagePrompt(params: {
  prompt: string
  frameType?: string | null
  size?: string | null
  referenceCount: number
  outputPath: string
}): string {
  const isGrid = String(params.frameType || '').startsWith('grid_')
  const mode = params.referenceCount > 0
    ? (isGrid ? '多图参考宫格生图' : '多图参考生图')
    : '文生图'
  const references = Array.from({ length: params.referenceCount }, (_, index) => `图片${index + 1}`).join('、')

  return [
    '你是火宝短剧的本机 Codex 生图执行器。',
    `任务类型：${mode}`,
    `输出尺寸：${params.size || '1920x1080'}`,
    `OUTPUT_PATH: ${params.outputPath}`,
    '必须创建 exactly one raster PNG image file at OUTPUT_PATH。',
    '必须使用当前 Codex 环境中可用的图像生成能力；如果当前环境没有真实生图能力，不要用脚本绘制占位图，不要生成 SVG/HTML/纯色图。',
    params.referenceCount > 0 ? `已附加 ${params.referenceCount} 张参考图：${references}。` : '',
    params.referenceCount > 0 && !isGrid
      ? '请融合参考图中的人物、场景、服装、空间关系和整体风格，生成一个新的统一画面。不要把参考图拼贴成 collage，不要出现分割线。'
      : '',
    params.referenceCount > 0 && isGrid
      ? '这是宫格图片任务，可以按提示词生成多 panel/grid layout；每个格子要清晰可拆分，格子数量和布局必须遵守提示词。'
      : '',
    '画面要求：电影感，高质量，主体明确，无文字，无水印，无 logo。',
    `用户提示词：${params.prompt}`,
    '最终只返回 JSON：{"ok":true,"local_path":"<OUTPUT_PATH>","prompt_used":"<实际使用的提示词>","error":""}。如果失败，返回 ok=false 并填写 error。',
  ].filter(Boolean).join('\n')
}

function parseReferenceImages(raw: string | null | undefined): string[] {
  if (!raw) return []
  try {
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed.map(item => String(item || '')).filter(Boolean) : []
  } catch {
    return []
  }
}

async function writeReferenceImage(tempDir: string, source: string, index: number): Promise<string> {
  const parsed = parseDataUrl(source)
  if (parsed) {
    const filePath = path.join(tempDir, `reference-${index + 1}${mimeTypeToExt(parsed.mimeType)}`)
    await writeFile(filePath, Buffer.from(parsed.data, 'base64'))
    return filePath
  }

  const resp = await fetch(source)
  if (!resp.ok) throw new Error(`Reference image fetch failed: ${resp.status}`)
  const mimeType = resp.headers.get('content-type')?.split(';')[0]?.trim() || 'image/png'
  const filePath = path.join(tempDir, `reference-${index + 1}${mimeTypeToExt(mimeType)}`)
  await writeFile(filePath, Buffer.from(await resp.arrayBuffer()))
  return filePath
}

function mimeTypeToExt(mimeType: string): string {
  const map: Record<string, string> = {
    'image/png': '.png',
    'image/jpeg': '.jpg',
    'image/jpg': '.jpg',
    'image/webp': '.webp',
  }
  return map[mimeType] || '.png'
}
