import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

import { getAbsolutePath, parseDataUrl } from '../../utils/storage.js'
import { normalizeVideoDuration } from '../duration.js'
import type {
  AIConfig,
  ProviderRequest,
  VideoGenerationRecord,
  VideoGenResponse,
  VideoPollResponse,
  VideoProviderAdapter,
} from './types'

type WorkflowGraph = Record<string, WorkflowNode>

type WorkflowNode = {
  class_type: string
  inputs: Record<string, unknown>
  _meta?: {
    title?: string
    id?: string
  }
}

type ComfyVideoRef = {
  filename: string
  subfolder?: string
  type?: string
}

type ResolveVideoWorkflowParams = {
  workflowKey: string
  prompt: string
  imageFilenames: string[]
  audioFilename?: string | null
  audioFilenames?: string[]
  duration?: number | null
  fps?: number | null
  frameCount?: number | null
  aspectRatio?: string | null
  resolution?: string | null
  motionLevel?: number | null
  seed?: number | null
  outputPrefix?: string | null
}

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const PROJECT_ROOT = path.resolve(__dirname, '../../../..')
const WORKFLOW_ROOT = path.join(PROJECT_ROOT, 'configs', 'comfyui', 'workflows')

const DEFAULT_VIDEO_WORKFLOW = 'basevideo/Seedance2.0_Bernini_01_480p_10s'
const DEFAULT_AUDIO_LIPSYNC_VIDEO_WORKFLOW = 'basevideo/Seedance2.0_Bernini_01+02+04_480p_10s_图生视频出音效对口型'
const DEFAULT_VIDEO_FPS = 24
const DEFAULT_VIDEO_ASPECT_RATIO = '16:9'
const DEFAULT_VIDEO_RESOLUTION = '480p'
const DEFAULT_VIDEO_OUTPUT_PREFIX = 'video/huobao-bernini'
const DEFAULT_AUDIO_LIPSYNC_OUTPUT_PREFIX = 'video/huobao-bernini-lipsync'
const MOTION_LORA_NAME_FRAGMENT = '260412_rank_64_fp16'
const SEED_CONTROL_VALUES = new Set(['fixed', 'randomize', 'increment', 'decrement'])
const UI_ONLY_INPUT_TYPE_SUFFIXES = ['UPLOAD', '_UI']
const AUDIO_UPLOAD_TIMEOUT_MS = 600_000

const MOTION_LABELS: Record<1 | 2 | 3, string> = {
  1: 'calm',
  2: 'standard',
  3: 'intense',
}

const MOTION_PROMPT_GUIDANCE: Record<1 | 2 | 3, string> = {
  1: 'preserve the reference composition, subjects, spatial layout, and scene identity; use small controlled movement only, such as subtle expression changes, breathing, hand movement, paper movement, or a very slow camera push; avoid jump cuts, large reframing, new subjects, or scene changes.',
  2: 'preserve the reference composition and scene identity while allowing clear but continuous subject action and moderate camera movement that follows the storyboard.',
  3: 'preserve core subject and scene identity while allowing large action, stronger camera movement, and faster visual change only when the storyboard asks for it.',
}

const INTENSE_MOTION_PATTERN = /激烈|奔跑|高速|冲刺|跳跃|打斗|搏斗|爆炸|撞|追逐|翻滚|急速|剧烈|sprint|running|run\b|jump|fight|intense|explosion|chase|fast|crash/i
const CALM_MOTION_PATTERN = /平静|静止|坐着|坐下|站立|凝视|微笑|呼吸|缓慢|轻轻|安静|calm|still|sitting|sit\b|slowly|gentle|peaceful/i

export class ComfyUiVideoAdapter implements VideoProviderAdapter {
  provider = 'comfyui'

  async buildGenerateRequest(config: AIConfig, record: VideoGenerationRecord): Promise<ProviderRequest> {
    const references = collectVideoReferences(record)
    if (!references.length) {
      throw new Error('ComfyUI Bernini video workflow requires at least one reference image')
    }
    const audioSources = collectVideoAudioReferences(record)

    const imageFilenames = await Promise.all(
      references.slice(0, 1).map((source, index) => uploadComfyUiImage(config.baseUrl, source, index)),
    )
    const audioFilenames = await Promise.all(
      audioSources.slice(0, 1).map((source, index) => uploadComfyUiAudio(config.baseUrl, source, index)),
    )
    const hasAudio = audioFilenames.length > 0
    const workflowKey = selectComfyUiVideoWorkflow(record.model || config.model, hasAudio)
    const duration = normalizeVideoDuration(record.duration)
    const fps = normalizeComfyUiVideoFps(record.fps)
    const motionLevel = normalizeComfyUiVideoMotionLevel(record.motionLevel)
      || inferComfyUiVideoMotionLevel([
        record.prompt,
        record.cameraMotion,
        record.style,
      ].filter(Boolean).join('\n'))

    const workflow = resolveComfyUiVideoWorkflow({
      workflowKey,
      prompt: record.prompt || 'Generate a short cinematic video.',
      imageFilenames,
      audioFilenames,
      duration,
      fps,
      frameCount: record.frameCount,
      aspectRatio: record.aspectRatio || DEFAULT_VIDEO_ASPECT_RATIO,
      resolution: record.resolution || DEFAULT_VIDEO_RESOLUTION,
      motionLevel,
      seed: record.seed,
      outputPrefix: hasAudio
        ? `${DEFAULT_AUDIO_LIPSYNC_OUTPUT_PREFIX}-${record.id}`
        : `${DEFAULT_VIDEO_OUTPUT_PREFIX}-${record.id}`,
    })

    return {
      url: buildComfyUiUrl(config.baseUrl, '/prompt'),
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: {
        prompt: workflow,
        client_id: `huobao-drama-video-${record.id}`,
      },
    }
  }

  parseGenerateResponse(result: any): VideoGenResponse {
    const taskId = result?.prompt_id || result?.promptId || result?.id
    if (!taskId) {
      const error = result?.error?.message || result?.error || 'No ComfyUI prompt_id in video response'
      throw new Error(String(error))
    }
    return { isAsync: true, taskId: String(taskId) }
  }

  buildPollRequest(config: AIConfig, taskId: string): ProviderRequest {
    return {
      url: buildComfyUiUrl(config.baseUrl, `/history/${encodeURIComponent(taskId)}`),
      method: 'GET',
      headers: {},
      body: undefined,
    }
  }

  parsePollResponse(result: any, config?: AIConfig): VideoPollResponse {
    return parseComfyUiVideoHistoryResponse(result, config?.baseUrl || '')
  }

  extractVideoUrl(result: any): string | null {
    return result?.videoUrl || result?.video_url || result?.data?.video_url || null
  }
}

export function selectComfyUiVideoWorkflow(model?: string | null, hasAudio = false): string {
  if (hasAudio) return DEFAULT_AUDIO_LIPSYNC_VIDEO_WORKFLOW
  const value = stripJsonSuffix(String(model || '').trim())
  if (value === DEFAULT_AUDIO_LIPSYNC_VIDEO_WORKFLOW) return DEFAULT_VIDEO_WORKFLOW
  if (value.startsWith('basevideo/')) return value
  return DEFAULT_VIDEO_WORKFLOW
}

export function resolveComfyUiVideoWorkflow(params: ResolveVideoWorkflowParams): WorkflowGraph {
  const duration = normalizeVideoDuration(params.duration)
  const fps = normalizeComfyUiVideoFps(params.fps)
  const frameCount = normalizeComfyUiVideoFrameCount(params.frameCount, duration, fps)
  const motionLevel = normalizeComfyUiVideoMotionLevel(params.motionLevel)
    || inferComfyUiVideoMotionLevel(params.prompt)
  const dimensions = resolveVideoDimensions(params.aspectRatio, params.resolution)
  const graph = readWorkflowGraph(params.workflowKey)
  const audioFilenames = params.audioFilenames?.length
    ? params.audioFilenames
    : (params.audioFilename ? [params.audioFilename] : [])

  applyImageInjection(graph, params.imageFilenames)
  if (audioFilenames.length) applyAudioInjection(graph, audioFilenames)
  applyDimensionInjection(graph, dimensions)
  applyFrameCountInjection(graph, frameCount)
  applyFpsInjection(graph, fps)
  applyPromptEnhancerDuration(graph, duration)
  applyMotionLoraInjection(graph, motionLevel)
  applyRunningHubCodexPromptInjection(graph, params.prompt, duration, motionLevel)
  applySafeVideoOutputPrefix(graph, params.outputPrefix || `${DEFAULT_VIDEO_OUTPUT_PREFIX}-${Date.now()}`)
  pruneUnreachableFromOutputs(graph)
  assignSeeds(graph, params.seed)
  return graph
}

export function inferComfyUiVideoMotionLevel(text: string): 1 | 2 | 3 {
  const value = String(text || '')
  if (INTENSE_MOTION_PATTERN.test(value)) return 3
  if (CALM_MOTION_PATTERN.test(value)) return 1
  return 2
}

export function normalizeComfyUiVideoMotionLevel(value: number | null | undefined): 1 | 2 | 3 | null {
  const parsed = Math.round(Number(value))
  if (!Number.isFinite(parsed)) return null
  if (parsed <= 1) return 1
  if (parsed >= 3) return 3
  return 2
}

export function normalizeComfyUiVideoFps(value: number | null | undefined): number {
  const parsed = Math.round(Number(value ?? DEFAULT_VIDEO_FPS))
  if (!Number.isFinite(parsed)) return DEFAULT_VIDEO_FPS
  return Math.min(60, Math.max(1, parsed))
}

export function normalizeComfyUiVideoFrameCount(
  frameCount: number | null | undefined,
  duration: number,
  fps: number,
): number {
  const explicit = Math.round(Number(frameCount))
  if (Number.isFinite(explicit) && explicit > 0) return Math.min(1000, explicit)
  return Math.min(1000, Math.max(1, Math.round(duration * fps) + 1))
}

export function parseComfyUiVideoHistoryResponse(result: any, baseUrl: string): VideoPollResponse {
  const entries = Object.values(result || {}) as any[]
  if (!entries.length) return { status: 'processing' }

  const history = entries[0]
  const statusText = String(history?.status?.status_str || history?.status?.status || '').toLowerCase()
  if (statusText.includes('error') || statusText.includes('failed')) {
    const message = history?.status?.messages?.flat?.()?.join('\n') || 'ComfyUI video generation failed'
    return { status: 'failed', error: String(message) }
  }

  const videoUrl = findFirstOutputVideoUrl(history?.outputs)
  if (videoUrl) return { status: 'completed', videoUrl }

  const video = findFirstOutputVideo(history?.outputs)
  if (video) {
    return {
      status: 'completed',
      videoUrl: buildComfyUiVideoViewUrl(baseUrl, video),
    }
  }

  if (history?.status?.completed === false) return { status: 'processing' }
  return { status: 'processing' }
}

export function buildComfyUiVideoViewUrl(baseUrl: string, video: ComfyVideoRef): string {
  const url = new URL(buildComfyUiUrl(baseUrl, '/view'))
  url.searchParams.set('filename', video.filename)
  url.searchParams.set('subfolder', video.subfolder || '')
  url.searchParams.set('type', video.type || 'output')
  return url.toString()
}

function readWorkflowGraph(workflowKey: string): WorkflowGraph {
  const filePath = resolveWorkflowPath(workflowKey)
  const parsed = JSON.parse(fs.readFileSync(filePath, 'utf8')) as any
  if (isApiWorkflowGraph(parsed)) return normalizeApiWorkflowGraph(parsed)
  if (Array.isArray(parsed?.nodes) && Array.isArray(parsed?.links)) {
    return convertUiWorkflowToApiGraph(parsed)
  }
  throw new Error(`Unsupported ComfyUI video workflow format: ${workflowKey}`)
}

function resolveWorkflowPath(workflowKey: string): string {
  const safeKey = stripJsonSuffix(workflowKey).replace(/\\/g, '/').replace(/^\//, '')
  if (safeKey.includes('..')) throw new Error(`Unsafe ComfyUI video workflow key: ${workflowKey}`)
  const filePath = path.resolve(WORKFLOW_ROOT, `${safeKey}.json`)
  if (!filePath.startsWith(WORKFLOW_ROOT + path.sep)) {
    throw new Error(`ComfyUI video workflow path escaped root: ${workflowKey}`)
  }
  if (!fs.existsSync(filePath)) throw new Error(`ComfyUI video workflow not found: ${workflowKey}`)
  return filePath
}

function isApiWorkflowGraph(value: unknown): value is WorkflowGraph {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const entries = Object.entries(value as Record<string, unknown>)
  return entries.length > 0 && entries.every(([, node]) => {
    return !!node
      && typeof node === 'object'
      && typeof (node as any).class_type === 'string'
      && !!(node as any).inputs
      && typeof (node as any).inputs === 'object'
  })
}

function normalizeApiWorkflowGraph(raw: WorkflowGraph): WorkflowGraph {
  const graph: WorkflowGraph = {}
  for (const [nodeId, node] of Object.entries(raw)) {
    graph[String(nodeId)] = {
      class_type: node.class_type,
      inputs: clone(node.inputs || {}),
      _meta: { ...(node._meta || {}), id: String(nodeId) },
    }
  }
  return graph
}

function convertUiWorkflowToApiGraph(raw: any): WorkflowGraph {
  const linkMap = new Map<number, [string, number]>()
  for (const link of raw.links || []) {
    if (!Array.isArray(link) || link.length < 3) continue
    const linkId = Number(link[0])
    const sourceNodeId = String(link[1])
    const sourceSlot = Number(link[2])
    if (Number.isFinite(linkId) && sourceNodeId && Number.isFinite(sourceSlot)) {
      linkMap.set(linkId, [sourceNodeId, sourceSlot])
    }
  }

  const graph: WorkflowGraph = {}
  for (const rawNode of raw.nodes || []) {
    const nodeId = String(rawNode?.id || '')
    const classType = String(rawNode?.type || '')
    if (!nodeId || !classType) continue

    const inputs: Record<string, unknown> = {}
    const rawWidgetValues = rawNode.widgets_values
    const widgetValues = Array.isArray(rawWidgetValues) ? rawWidgetValues : []
    const widgetObject = rawWidgetValues
      && typeof rawWidgetValues === 'object'
      && !Array.isArray(rawWidgetValues)
      ? rawWidgetValues as Record<string, unknown>
      : null
    let widgetIndex = 0

    for (const inputDef of rawNode.inputs || []) {
      const inputName = String(inputDef?.name || inputDef?.label || '')
      if (!inputName) continue

      const linkId = readUiLinkId(inputDef?.link)
      const linked = linkId !== null ? linkMap.get(linkId) : null
      const hasWidget = !!inputDef?.widget
      if (linked) {
        inputs[inputName] = linked
      } else if (hasWidget && !shouldSkipUiOnlyInput(inputDef)) {
        const widgetValue = widgetObject && Object.prototype.hasOwnProperty.call(widgetObject, inputName)
          ? widgetObject[inputName]
          : widgetValues[widgetIndex]
        if (widgetValue !== undefined) inputs[inputName] = clone(widgetValue)
      }

      if (hasWidget) {
        widgetIndex += 1
        if (typeof widgetValues[widgetIndex] === 'string' && SEED_CONTROL_VALUES.has(widgetValues[widgetIndex])) {
          widgetIndex += 1
        }
      }
    }

    graph[nodeId] = {
      class_type: classType,
      inputs,
      _meta: {
        id: nodeId,
        ...(rawNode.title ? { title: String(rawNode.title) } : {}),
      },
    }
  }

  return graph
}

function applyImageInjection(graph: WorkflowGraph, imageFilenames: string[]): void {
  const loadNodes = Object.entries(graph)
    .filter(([, node]) => normalizeNodeType(node.class_type) === 'loadimage')
    .sort(([left], [right]) => Number(left) - Number(right))
  const fallback = imageFilenames[imageFilenames.length - 1]

  loadNodes.forEach(([, node], index) => {
    const filename = imageFilenames[index] || fallback
    if (filename) node.inputs.image = filename
    else delete node.inputs.image
    delete node.inputs.upload
    delete node.inputs.imageUI
    delete node.inputs.imageui
  })
}

function applyAudioInjection(graph: WorkflowGraph, audioFilenames: string[]): void {
  const loadNodes = Object.entries(graph)
    .filter(([, node]) => normalizeNodeType(node.class_type) === 'loadaudio')
    .sort(([left], [right]) => Number(left) - Number(right))
  const fallback = audioFilenames[audioFilenames.length - 1]

  loadNodes.forEach(([, node], index) => {
    const filename = audioFilenames[index] || fallback
    if (filename) node.inputs.audio = filename
    else delete node.inputs.audio
    delete node.inputs.upload
    delete node.inputs.audioUI
    delete node.inputs.audioui
  })
}

function applyDimensionInjection(
  graph: WorkflowGraph,
  dimensions: { width: number; height: number; aspectRatio: string },
): void {
  const longestSide = Math.max(dimensions.width, dimensions.height)
  for (const node of Object.values(graph)) {
    assignNumericInput(graph, node, 'width', dimensions.width)
    assignNumericInput(graph, node, 'height', dimensions.height)
    assignNumericInput(graph, node, 'scale_to_length', longestSide)
    assignNumericInput(graph, node, 'ref_max_size', longestSide)
    assignStringInputIfPresent(graph, node, 'aspect_ratio', dimensions.aspectRatio)
    assignNumericInput(graph, node, 'proportional_width', parseAspectRatio(dimensions.aspectRatio).width)
    assignNumericInput(graph, node, 'proportional_height', parseAspectRatio(dimensions.aspectRatio).height)
  }
}

function applyFrameCountInjection(graph: WorkflowGraph, frameCount: number): void {
  for (const node of Object.values(graph)) {
    assignNumericInput(graph, node, 'length', frameCount)
    assignNumericInput(graph, node, 'end_frame', frameCount)
  }
}

function applyFpsInjection(graph: WorkflowGraph, fps: number): void {
  for (const node of Object.values(graph)) {
    assignNumericInput(graph, node, 'fps', fps)
    assignNumericInput(graph, node, 'frame_rate', fps)
  }
}

function applyPromptEnhancerDuration(graph: WorkflowGraph, duration: number): void {
  for (const node of Object.values(graph)) {
    if (normalizeNodeType(node.class_type) !== 'berninipromptenhancer') continue
    assignNumericInput(graph, node, 'video_frames', duration)
  }
}

function applyMotionLoraInjection(graph: WorkflowGraph, motionLevel: 1 | 2 | 3): void {
  for (const node of Object.values(graph)) {
    if (normalizeNodeType(node.class_type) !== 'loraloadermodelonly') continue
    if (!String(node.inputs.lora_name || '').includes(MOTION_LORA_NAME_FRAGMENT)) continue
    node.inputs.strength_model = motionLevel
  }
}

function applyRunningHubCodexPromptInjection(
  graph: WorkflowGraph,
  prompt: string,
  duration: number,
  motionLevel: 1 | 2 | 3,
): void {
  const motionLabel = MOTION_LABELS[motionLevel]
  const motionGuidance = MOTION_PROMPT_GUIDANCE[motionLevel]
  const videoRole = [
    'You are Huobao Drama AI video prompt expert.',
    `Target duration: ${duration} seconds.`,
    `motion level: ${motionLabel}.`,
    'Use the reference image and storyboard intent to write only one concise image-to-video prompt.',
    motionGuidance,
    'Describe subject, background, action progression, visual mood, camera movement, and final shot.',
    'Do not output commentary, JSON, lists, or abstract concepts.',
  ].join(' ')
  const videoTask = [
    `Storyboard video prompt: ${prompt}`,
    `Write a ${duration} seconds video prompt.`,
    `motion level: ${motionLabel}.`,
    `motion guidance: ${motionGuidance}`,
    'Write one coherent paragraph based on this storyboard and reference image; do not invent unrelated scene content.',
  ].join('\n')
  const foleyRole = [
    'You are Huobao Drama AI Foley and sound effects prompt expert.',
    `Target duration: ${duration} seconds.`,
    `motion level: ${motionLabel}.`,
    'Use the generated video frames and storyboard intent to write one concise sound effects prompt for video Foley generation.',
    'Focus on natural ambience, footsteps, cloth, props, room tone, impacts, and movement cues that match the visible action.',
    'Do not request speech, dialogue, narration, lyrics, music, or unrelated off-screen events.',
  ].join(' ')
  const foleyTask = [
    `Storyboard video prompt: ${prompt}`,
    `Write one Foley sound effects prompt for a ${duration} seconds generated video.`,
    `motion level: ${motionLabel}.`,
    `motion guidance: ${motionGuidance}`,
    'Describe only audible ambience and physical sound effects that should synchronize with the visible action.',
  ].join('\n')

  for (const node of Object.values(graph)) {
    if (normalizeNodeType(node.class_type) !== 'rhcodexnode') continue
    const title = String(node._meta?.title || '')
    const useFoleyPrompt = /foley|音效/i.test(title)
    assignStringInput(graph, node, 'role', useFoleyPrompt ? foleyRole : videoRole)
    assignStringInput(graph, node, 'prompt', useFoleyPrompt ? foleyTask : videoTask)
    assignNumericInput(graph, node, 'timeout_seconds', 240)
    assignStringInputIfPresent(graph, node, 'sandbox_mode', 'read-only')
    const codexPath = String(process.env.CODEX_BIN || '').trim()
    if (codexPath) assignStringInputIfPresent(graph, node, 'codex_path', codexPath)
  }
}

function applySafeVideoOutputPrefix(graph: WorkflowGraph, outputPrefix: string): void {
  const safePrefix = sanitizeVideoFilenamePrefix(outputPrefix)
  for (const node of Object.values(graph)) {
    if (!isVideoOutputNode(node)) continue
    if (!Object.prototype.hasOwnProperty.call(node.inputs, 'filename_prefix')) continue
    node.inputs.filename_prefix = safePrefix
  }
}

function pruneUnreachableFromOutputs(graph: WorkflowGraph): void {
  const outputNodeIds = Object.entries(graph)
    .filter(([, node]) => isVideoOutputNode(node))
    .map(([nodeId]) => nodeId)
  if (!outputNodeIds.length) return

  const reachable = new Set<string>()
  const visit = (nodeId: string) => {
    if (reachable.has(nodeId)) return
    const node = graph[nodeId]
    if (!node) return
    reachable.add(nodeId)
    for (const input of Object.values(node.inputs)) {
      if (isConnectionValue(input)) visit(String(input[0]))
    }
  }
  outputNodeIds.forEach(visit)

  for (const nodeId of Object.keys(graph)) {
    if (!reachable.has(nodeId)) delete graph[nodeId]
  }
}

function isVideoOutputNode(node: WorkflowNode): boolean {
  const classType = normalizeNodeType(node.class_type)
  if (classType === 'savevideo') return true
  if (classType !== 'vhsvideocombine') return false

  if (node.inputs.save_output === true) return true
  const title = String(node._meta?.title || '')
  return title.includes('最终视频')
}

function assignSeeds(graph: WorkflowGraph, seed?: number | null): void {
  const explicit = Math.round(Number(seed))
  const hasExplicit = Number.isFinite(explicit)
  for (const node of Object.values(graph)) {
    for (const field of ['seed', 'noise_seed']) {
      if (!Object.prototype.hasOwnProperty.call(node.inputs, field) || isConnectionValue(node.inputs[field])) continue
      node.inputs[field] = hasExplicit ? explicit : Math.floor(Math.random() * 2_147_483_648)
    }
  }
}

function assignNumericInput(graph: WorkflowGraph, node: WorkflowNode, inputName: string, value: number): void {
  if (!Object.prototype.hasOwnProperty.call(node.inputs, inputName)) return
  const current = node.inputs[inputName]
  if (isConnectionValue(current)) {
    const source = graph[String(current[0])]
    if (source) setNumericValueOnNode(source, value)
    return
  }
  node.inputs[inputName] = value
}

function assignStringInput(graph: WorkflowGraph, node: WorkflowNode, inputName: string, value: string): void {
  const current = node.inputs[inputName]
  if (isConnectionValue(current)) {
    const source = graph[String(current[0])]
    if (source) setStringValueOnNode(source, value)
    return
  }
  node.inputs[inputName] = value
}

function assignStringInputIfPresent(graph: WorkflowGraph, node: WorkflowNode, inputName: string, value: string): void {
  if (!Object.prototype.hasOwnProperty.call(node.inputs, inputName)) return
  assignStringInput(graph, node, inputName, value)
}

function setNumericValueOnNode(node: WorkflowNode, value: number): void {
  for (const field of ['value', 'width', 'height', 'length', 'end_frame', 'scale_to_length', 'fps', 'frame_rate']) {
    if (Object.prototype.hasOwnProperty.call(node.inputs, field) && !isConnectionValue(node.inputs[field])) {
      node.inputs[field] = value
      return
    }
  }
}

function setStringValueOnNode(node: WorkflowNode, value: string): void {
  for (const field of ['prompt', 'text', 'value', 'string', 'input_string']) {
    if (Object.prototype.hasOwnProperty.call(node.inputs, field) && !isConnectionValue(node.inputs[field])) {
      node.inputs[field] = value
      return
    }
  }
}

function resolveVideoDimensions(
  aspectRatio?: string | null,
  resolution?: string | null,
): { width: number; height: number; aspectRatio: string } {
  const ratio = normalizeAspectRatio(aspectRatio)
  const parsed = parseAspectRatio(ratio)
  const shortSide = parseResolutionShortSide(resolution)

  if (parsed.width === parsed.height) {
    return { width: shortSide, height: shortSide, aspectRatio: ratio }
  }

  if (parsed.width > parsed.height) {
    return {
      width: roundToMultiple((shortSide * parsed.width) / parsed.height, 16),
      height: shortSide,
      aspectRatio: ratio,
    }
  }

  return {
    width: shortSide,
    height: roundToMultiple((shortSide * parsed.height) / parsed.width, 16),
    aspectRatio: ratio,
  }
}

function normalizeAspectRatio(value?: string | null): string {
  const raw = String(value || '').trim()
  const match = raw.match(/^(\d+)\s*[:x]\s*(\d+)$/i)
  if (!match) return DEFAULT_VIDEO_ASPECT_RATIO
  const width = Math.max(1, Number(match[1]))
  const height = Math.max(1, Number(match[2]))
  const divisor = gcd(width, height)
  return `${Math.round(width / divisor)}:${Math.round(height / divisor)}`
}

function parseAspectRatio(value: string): { width: number; height: number } {
  const [width, height] = normalizeAspectRatio(value).split(':').map(Number)
  return { width, height }
}

function parseResolutionShortSide(value?: string | null): number {
  const raw = String(value || DEFAULT_VIDEO_RESOLUTION).trim().toLowerCase()
  const match = raw.match(/(\d+)/)
  const parsed = match ? Number(match[1]) : 480
  if (!Number.isFinite(parsed) || parsed <= 0) return 480
  return Math.min(1080, Math.max(128, Math.round(parsed)))
}

function roundToMultiple(value: number, multiple: number): number {
  return Math.max(multiple, Math.round(value / multiple) * multiple)
}

function gcd(left: number, right: number): number {
  let a = Math.abs(Math.round(left))
  let b = Math.abs(Math.round(right))
  while (b) {
    const next = a % b
    a = b
    b = next
  }
  return a || 1
}

function sanitizeVideoFilenamePrefix(value: string): string {
  const normalized = value.trim().replace(/\\/g, '/')
  const base = normalized || DEFAULT_VIDEO_OUTPUT_PREFIX
  const safe = base
    .split('/')
    .map(segment => segment
      .replace(/[<>:"\\|?*\u0000-\u001F]/g, '-')
      .replace(/[. ]+$/g, '')
      .trim())
    .filter(Boolean)
    .join('/')
  return safe || DEFAULT_VIDEO_OUTPUT_PREFIX
}

function collectVideoReferences(record: VideoGenerationRecord): string[] {
  if (record.referenceMode === 'single' && record.imageUrl) return [record.imageUrl]
  if (record.referenceMode === 'first_last') {
    return [record.firstFrameUrl, record.lastFrameUrl].filter((item): item is string => !!item)
  }
  if (record.referenceMode === 'multiple' && record.referenceImageUrls) {
    return parseReferenceImages(record.referenceImageUrls)
  }
  return [record.imageUrl, record.firstFrameUrl].filter((item): item is string => !!item)
}

function collectVideoAudioReferences(record: VideoGenerationRecord): string[] {
  return [record.audioUrl].map(item => String(item || '').trim()).filter(Boolean)
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

async function uploadComfyUiImage(baseUrl: string, source: string, index: number): Promise<string> {
  const { buffer, mimeType } = await loadImageBuffer(source)
  const filename = `huobao-video-ref-${Date.now()}-${index}${mimeTypeToExt(mimeType)}`
  const imageData = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer
  const form = new FormData()
  form.append('image', new Blob([imageData], { type: mimeType }), filename)
  form.append('type', 'input')
  form.append('overwrite', 'true')

  const resp = await fetch(buildComfyUiUrl(baseUrl, '/upload/image'), {
    method: 'POST',
    body: form,
  })
  if (!resp.ok) throw new Error(`ComfyUI video image upload failed: ${resp.status} ${await resp.text()}`)
  const result = await resp.json() as any
  return String(result?.name || filename)
}

async function uploadComfyUiAudio(baseUrl: string, source: string, index: number): Promise<string> {
  const { buffer, mimeType, filename } = await loadAudioBuffer(source, index)
  const audioData = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer
  const form = new FormData()
  form.append('image', new Blob([audioData], { type: mimeType }), filename)
  form.append('type', 'input')
  form.append('overwrite', 'true')

  const resp = await fetch(buildComfyUiUrl(baseUrl, '/upload/image'), {
    method: 'POST',
    body: form,
    signal: AbortSignal.timeout(AUDIO_UPLOAD_TIMEOUT_MS),
  })
  if (!resp.ok) throw new Error(`ComfyUI video audio upload failed: ${resp.status} ${await resp.text()}`)
  const result = await resp.json() as any
  return String(result?.name || filename)
}

async function loadImageBuffer(source: string): Promise<{ buffer: Buffer; mimeType: string }> {
  const parsed = parseDataUrl(source)
  if (parsed) {
    return {
      buffer: Buffer.from(parsed.data, 'base64'),
      mimeType: parsed.mimeType,
    }
  }

  const resp = await fetch(source)
  if (!resp.ok) throw new Error(`ComfyUI video reference image fetch failed: ${resp.status}`)
  const mimeType = resp.headers.get('content-type')?.split(';')[0]?.trim() || 'image/png'
  return {
    buffer: Buffer.from(await resp.arrayBuffer()),
    mimeType,
  }
}

async function loadAudioBuffer(source: string, index: number): Promise<{ buffer: Buffer; mimeType: string; filename: string }> {
  const parsed = parseDataUrl(source)
  if (parsed) {
    const ext = mimeTypeToAudioExt(parsed.mimeType)
    return {
      buffer: Buffer.from(parsed.data, 'base64'),
      mimeType: parsed.mimeType,
      filename: `huobao-video-audio-${Date.now()}-${index}${ext}`,
    }
  }

  if (/^https?:\/\//i.test(source)) {
    const resp = await fetch(source, { signal: AbortSignal.timeout(AUDIO_UPLOAD_TIMEOUT_MS) })
    if (!resp.ok) throw new Error(`ComfyUI video audio fetch failed: ${resp.status}`)
    const mimeType = resp.headers.get('content-type')?.split(';')[0]?.trim() || 'audio/mpeg'
    const filename = path.basename(new URL(source).pathname) || `huobao-video-audio-${Date.now()}-${index}${mimeTypeToAudioExt(mimeType)}`
    return {
      buffer: Buffer.from(await resp.arrayBuffer()),
      mimeType,
      filename: sanitizeUploadFilename(filename, `huobao-video-audio-${Date.now()}-${index}${mimeTypeToAudioExt(mimeType)}`),
    }
  }

  const normalized = source.startsWith('/static/') ? source.slice(1) : source
  const filePath = normalized.startsWith('static/')
    ? getAbsolutePath(normalized)
    : path.resolve(normalized)
  const ext = path.extname(filePath).toLowerCase()
  return {
    buffer: fs.readFileSync(filePath),
    mimeType: audioExtToMimeType(ext),
    filename: sanitizeUploadFilename(path.basename(filePath), `huobao-video-audio-${Date.now()}-${index}${ext || '.mp3'}`),
  }
}

function findFirstOutputVideo(outputs: any): ComfyVideoRef | null {
  if (!outputs || typeof outputs !== 'object') return null
  const outputEntries = Object.entries(outputs).sort(([left], [right]) => Number(left) - Number(right))
  for (const [, output] of outputEntries) {
    for (const field of ['videos', 'gifs', 'images', 'filenames']) {
      const videos = (output as any)?.[field]
      if (!Array.isArray(videos) || !videos.length) continue
      const video = videos
        .map(item => normalizeOutputVideoRef(item))
        .find(item => !!item && (field !== 'images' || isVideoOutputFilename(item.filename)))
      if (video?.filename) {
        return {
          filename: String(video.filename),
          subfolder: video.subfolder ? String(video.subfolder) : '',
          type: video.type ? String(video.type) : 'output',
        }
      }
    }
  }
  return null
}

function normalizeOutputVideoRef(value: any): ComfyVideoRef | null {
  if (typeof value === 'string' && isVideoOutputFilename(value)) {
    return { filename: value, subfolder: '', type: 'output' }
  }
  if (!value?.filename) return null
  return {
    filename: String(value.filename),
    subfolder: value.subfolder ? String(value.subfolder) : '',
    type: value.type ? String(value.type) : 'output',
  }
}

function isVideoOutputFilename(value: unknown): boolean {
  return /\.(mp4|webm|mov|mkv|avi|gif)$/i.test(String(value || ''))
}

function findFirstOutputVideoUrl(outputs: any): string | null {
  if (!outputs || typeof outputs !== 'object') return null
  const outputEntries = Object.entries(outputs).sort(([left], [right]) => Number(left) - Number(right))
  for (const [, output] of outputEntries) {
    const value = (output as any)?.video_url || (output as any)?.videoUrl || (output as any)?.url
    if (typeof value === 'string' && value) return value
  }
  return null
}

function buildComfyUiUrl(baseUrl: string, route: string): string {
  const normalized = normalizeComfyBaseUrl(baseUrl)
  const url = new URL(normalized)
  const basePath = url.pathname.replace(/\/+$/, '')
  const routePath = route.startsWith('/') ? route : `/${route}`
  url.pathname = `${basePath}${routePath}`.replace(/\/{2,}/g, '/')
  return url.toString()
}

function normalizeComfyBaseUrl(baseUrl: string): string {
  const trimmed = String(baseUrl || '').trim().replace(/\/+$/, '')
  return trimmed || 'http://127.0.0.1:8878'
}

function readUiLinkId(value: unknown): number | null {
  if (value === null || value === undefined) return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function shouldSkipUiOnlyInput(inputDef: any): boolean {
  const type = String(inputDef?.type || '').toUpperCase()
  return UI_ONLY_INPUT_TYPE_SUFFIXES.some(suffix => type.endsWith(suffix))
}

function normalizeNodeType(classType: string): string {
  return classType.toLowerCase().replace(/[^a-z0-9]/g, '')
}

function isConnectionValue(value: unknown): value is [string, number] {
  return Array.isArray(value) && value.length >= 2 && (typeof value[0] === 'string' || typeof value[0] === 'number')
}

function stripJsonSuffix(value: string): string {
  return value.replace(/\.json$/i, '')
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function mimeTypeToExt(mimeType: string): string {
  const map: Record<string, string> = {
    'image/png': '.png',
    'image/jpeg': '.jpg',
    'image/jpg': '.jpg',
    'image/webp': '.webp',
    'image/gif': '.gif',
  }
  return map[mimeType] || '.png'
}

function audioExtToMimeType(ext: string): string {
  const map: Record<string, string> = {
    '.aac': 'audio/aac',
    '.flac': 'audio/flac',
    '.m4a': 'audio/mp4',
    '.mp3': 'audio/mpeg',
    '.ogg': 'audio/ogg',
    '.opus': 'audio/opus',
    '.wav': 'audio/wav',
    '.webm': 'audio/webm',
  }
  return map[ext] || 'audio/mpeg'
}

function mimeTypeToAudioExt(mimeType: string): string {
  const map: Record<string, string> = {
    'audio/aac': '.aac',
    'audio/flac': '.flac',
    'audio/mp4': '.m4a',
    'audio/mpeg': '.mp3',
    'audio/ogg': '.ogg',
    'audio/opus': '.opus',
    'audio/wav': '.wav',
    'audio/webm': '.webm',
    'audio/x-wav': '.wav',
  }
  return map[mimeType] || '.mp3'
}

function sanitizeUploadFilename(value: string, fallback: string): string {
  const safe = String(value || '')
    .replace(/[<>:"\\|?*\u0000-\u001F]/g, '-')
    .replace(/[. ]+$/g, '')
    .trim()
  return safe || fallback
}
