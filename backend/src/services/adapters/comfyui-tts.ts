import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

import { getAbsolutePath, parseDataUrl } from '../../utils/storage.js'
import type { AIConfig, TTSGeneratedAudio, TTSProviderAdapter } from './types'

type WorkflowGraph = Record<string, WorkflowNode>

type WorkflowNode = {
  class_type: string
  inputs: Record<string, unknown>
  _meta?: {
    title?: string
    id?: string
  }
}

type ComfyAudioRef = {
  filename: string
  subfolder?: string
  type?: string
}

type ComfyAudioPollResponse = {
  status: 'pending' | 'processing' | 'completed' | 'failed'
  audioUrl?: string
  error?: string
}

type ResolveAudioWorkflowParams = {
  workflowKey: string
  text: string
  referenceAudioFilename?: string
  modelPath?: string | null
}

type SelectAudioWorkflowParams = {
  text: string
  voice?: string | null
  model?: string | null
}

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const PROJECT_ROOT = path.resolve(__dirname, '../../../..')
const WORKFLOW_ROOT = path.join(PROJECT_ROOT, 'configs', 'comfyui', 'workflows')

const PURE_TEXT_WORKFLOW = 'baseaudio/音色/s2-se'
const FISHS2_SINGLE_CLONE_WORKFLOW = 'baseaudio/单人/s2-one'
const LONGCAT_SINGLE_CLONE_WORKFLOW = 'baseaudio/单人/LongCat-one'
const AUDIO_EXTENSIONS = new Set(['.aac', '.flac', '.m4a', '.mp3', '.ogg', '.opus', '.wav', '.webm'])
const OUTPUT_NODE_TYPES = new Set(['saveaudio', 'saveaudiomp3', 'saveaudioopus', 'previewaudio'])
const SEED_CONTROL_VALUES = new Set(['fixed', 'randomize', 'increment', 'decrement'])
const UI_ONLY_INPUT_TYPE_SUFFIXES = ['UPLOAD', '_UI']
const POLL_INTERVAL_MS = 1500
const GENERATION_TIMEOUT_MS = 600_000

export class ComfyUiTTSAdapter implements TTSProviderAdapter {
  readonly provider = 'comfyui'

  async generateAudio(config: AIConfig, params: any): Promise<TTSGeneratedAudio> {
    const voice = String(params.voice || '').trim()
    const model = params.model || config.model || ''
    const workflowKey = selectComfyUiAudioWorkflow({
      text: params.text,
      voice,
      model,
    })
    const referenceAudioFilename = isReferenceAudioSource(voice)
      ? await resolveReferenceAudioFilename(config.baseUrl, voice)
      : undefined
    const workflow = resolveComfyUiAudioWorkflow({
      workflowKey,
      text: params.text || '',
      referenceAudioFilename,
      modelPath: model,
    })

    const promptId = await queueComfyUiPrompt(config.baseUrl, workflow)
    const audioUrl = await waitForComfyUiAudio(config.baseUrl, promptId)
    const resp = await fetch(audioUrl, { signal: AbortSignal.timeout(GENERATION_TIMEOUT_MS) })
    if (!resp.ok) throw new Error(`ComfyUI audio download failed: ${resp.status} ${await resp.text()}`)

    const audioBuffer = Buffer.from(await resp.arrayBuffer())
    return {
      audioBuffer,
      audioLength: 0,
      sampleRate: 0,
      bitrate: 0,
      format: inferAudioFormatFromUrl(audioUrl),
      channel: 0,
    }
  }
}

export function selectComfyUiAudioWorkflow(params: SelectAudioWorkflowParams): string {
  const model = String(params.model || '').trim()
  if (model.startsWith('baseaudio/')) return stripJsonSuffix(model)

  if (!isReferenceAudioSource(params.voice)) return PURE_TEXT_WORKFLOW

  const normalizedModel = model.toLowerCase()
  if (normalizedModel.includes('s2') || normalizedModel.includes('fish')) {
    return FISHS2_SINGLE_CLONE_WORKFLOW
  }
  return LONGCAT_SINGLE_CLONE_WORKFLOW
}

export function resolveComfyUiAudioWorkflow(params: ResolveAudioWorkflowParams): WorkflowGraph {
  const graph = readWorkflowGraph(params.workflowKey)
  applyTargetTextInjection(graph, params.text)
  if (params.referenceAudioFilename) applyReferenceAudioInjection(graph, params.referenceAudioFilename)
  applyModelPathInjection(graph, params.modelPath || '')
  pruneUnreachableFromOutputs(graph)
  assignRandomSeeds(graph)
  return graph
}

export function parseComfyUiAudioHistoryResponse(result: any, baseUrl: string): ComfyAudioPollResponse {
  const entries = Object.values(result || {}) as any[]
  if (!entries.length) return { status: 'processing' }

  const history = entries[0]
  const statusText = String(history?.status?.status_str || history?.status?.status || '').toLowerCase()
  if (statusText.includes('error') || statusText.includes('failed')) {
    const message = history?.status?.messages?.flat?.()?.join('\n') || 'ComfyUI audio generation failed'
    return { status: 'failed', error: String(message) }
  }

  const audio = findFirstOutputAudio(history?.outputs)
  if (audio) {
    return {
      status: 'completed',
      audioUrl: buildComfyUiAudioViewUrl(baseUrl, audio),
    }
  }

  if (history?.status?.completed === false) return { status: 'processing' }
  return { status: 'processing' }
}

export function findFirstOutputAudio(outputs: any): ComfyAudioRef | null {
  if (!outputs || typeof outputs !== 'object') return null
  const outputEntries = Object.entries(outputs).sort(([left], [right]) => Number(left) - Number(right))
  for (const [, output] of outputEntries) {
    for (const key of ['audio', 'audios']) {
      const audios = (output as any)?.[key]
      if (!Array.isArray(audios) || !audios.length) continue
      const audio = audios[0]
      if (audio?.filename) {
        return {
          filename: String(audio.filename),
          subfolder: audio.subfolder ? String(audio.subfolder) : '',
          type: audio.type ? String(audio.type) : 'output',
        }
      }
    }
  }
  return null
}

export function buildComfyUiAudioViewUrl(baseUrl: string, audio: ComfyAudioRef): string {
  const url = new URL(buildComfyUiUrl(baseUrl, '/view'))
  url.searchParams.set('filename', audio.filename)
  url.searchParams.set('subfolder', audio.subfolder || '')
  url.searchParams.set('type', audio.type || 'output')
  return url.toString()
}

function readWorkflowGraph(workflowKey: string): WorkflowGraph {
  const filePath = resolveWorkflowPath(workflowKey)
  const parsed = JSON.parse(fs.readFileSync(filePath, 'utf8')) as any
  if (isApiWorkflowGraph(parsed)) return normalizeApiWorkflowGraph(parsed)
  if (Array.isArray(parsed?.nodes) && Array.isArray(parsed?.links)) {
    return convertUiWorkflowToApiGraph(parsed)
  }
  throw new Error(`Unsupported ComfyUI audio workflow format: ${workflowKey}`)
}

function resolveWorkflowPath(workflowKey: string): string {
  const safeKey = stripJsonSuffix(workflowKey).replace(/\\/g, '/').replace(/^\//, '')
  if (safeKey.includes('..')) throw new Error(`Unsafe ComfyUI audio workflow key: ${workflowKey}`)
  const filePath = path.resolve(WORKFLOW_ROOT, `${safeKey}.json`)
  if (!filePath.startsWith(WORKFLOW_ROOT + path.sep)) {
    throw new Error(`ComfyUI audio workflow path escaped root: ${workflowKey}`)
  }
  if (!fs.existsSync(filePath)) throw new Error(`ComfyUI audio workflow not found: ${workflowKey}`)
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
    const widgetValues = Array.isArray(rawNode.widgets_values) ? rawNode.widgets_values : []
    let widgetIndex = 0

    for (const inputDef of rawNode.inputs || []) {
      const inputName = String(inputDef?.name || inputDef?.label || '')
      if (!inputName) continue

      const linkId = readUiLinkId(inputDef?.link)
      const linked = linkId !== null ? linkMap.get(linkId) : null
      const hasWidget = !!inputDef?.widget
      if (linked) {
        inputs[inputName] = linked
      } else if (hasWidget && !shouldSkipUiOnlyInput(inputDef) && widgetValues[widgetIndex] !== undefined) {
        inputs[inputName] = clone(widgetValues[widgetIndex])
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
      _meta: { id: nodeId },
    }
  }

  return graph
}

function applyTargetTextInjection(graph: WorkflowGraph, text: string): void {
  for (const node of Object.values(graph)) {
    if (!isTtsNode(node)) continue
    if (Object.prototype.hasOwnProperty.call(node.inputs, 'text')) {
      assignStringInput(graph, node, 'text', text)
    }
  }
}

function applyReferenceAudioInjection(graph: WorkflowGraph, referenceAudioFilename: string): void {
  const loadAudioNodes = Object.entries(graph)
    .filter(([, node]) => normalizeNodeType(node.class_type) === 'loadaudio')
    .sort(([left], [right]) => Number(left) - Number(right))
  for (const [, node] of loadAudioNodes) {
    node.inputs.audio = referenceAudioFilename
    delete node.inputs.upload
    delete node.inputs.audioUI
    delete node.inputs.audioui
  }
}

function applyModelPathInjection(graph: WorkflowGraph, modelPath: string): void {
  const model = modelPath.trim()
  if (!model || model.startsWith('baseaudio/') || model.endsWith('.json')) return
  const normalizedModel = model.toLowerCase()
  for (const node of Object.values(graph)) {
    if (!Object.prototype.hasOwnProperty.call(node.inputs, 'model_path')) continue
    const classType = node.class_type.toLowerCase()
    if (classType.includes('fish') && (normalizedModel.includes('s2') || normalizedModel.includes('fish'))) {
      node.inputs.model_path = model
    }
    if (classType.includes('longcat') && normalizedModel.includes('longcat')) {
      node.inputs.model_path = model
    }
  }
}

function pruneUnreachableFromOutputs(graph: WorkflowGraph): void {
  const outputNodeIds = Object.entries(graph)
    .filter(([, node]) => OUTPUT_NODE_TYPES.has(normalizeNodeType(node.class_type)))
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

function assignRandomSeeds(graph: WorkflowGraph): void {
  for (const node of Object.values(graph)) {
    for (const field of ['seed', 'noise_seed']) {
      if (Object.prototype.hasOwnProperty.call(node.inputs, field) && !isConnectionValue(node.inputs[field])) {
        node.inputs[field] = Math.floor(Math.random() * 2_147_483_648)
      }
    }
  }
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

function setStringValueOnNode(node: WorkflowNode, value: string): void {
  for (const field of ['value', 'text', 'prompt', 'string', 'input_string']) {
    if (Object.prototype.hasOwnProperty.call(node.inputs, field) && !isConnectionValue(node.inputs[field])) {
      node.inputs[field] = value
      return
    }
  }
}

function isTtsNode(node: WorkflowNode): boolean {
  const classType = node.class_type.toLowerCase()
  return classType.includes('tts')
}

function isReferenceAudioSource(value: unknown): boolean {
  const voice = String(value || '').trim()
  if (!voice) return false
  if (voice.startsWith('data:audio/')) return true
  if (/^https?:\/\//i.test(voice)) return true
  const ext = path.extname(stripQueryString(voice)).toLowerCase()
  return AUDIO_EXTENSIONS.has(ext)
}

async function resolveReferenceAudioFilename(baseUrl: string, source: string): Promise<string> {
  if (!shouldUploadReferenceAudio(source)) return source
  const { buffer, mimeType, filename } = await loadAudioBuffer(source)
  return uploadComfyUiAudio(baseUrl, buffer, mimeType, filename)
}

function shouldUploadReferenceAudio(source: string): boolean {
  const value = source.trim()
  if (value.startsWith('data:audio/')) return true
  if (/^https?:\/\//i.test(value)) return true
  if (value.startsWith('static/')) return true
  if (path.isAbsolute(value)) return true
  return fs.existsSync(value)
}

async function loadAudioBuffer(source: string): Promise<{ buffer: Buffer; mimeType: string; filename: string }> {
  const parsed = parseDataUrl(source)
  if (parsed) {
    return {
      buffer: Buffer.from(parsed.data, 'base64'),
      mimeType: parsed.mimeType,
      filename: `huobao-ref-${Date.now()}${mimeTypeToAudioExt(parsed.mimeType)}`,
    }
  }

  if (/^https?:\/\//i.test(source)) {
    const resp = await fetch(source, { signal: AbortSignal.timeout(GENERATION_TIMEOUT_MS) })
    if (!resp.ok) throw new Error(`Reference audio fetch failed: ${resp.status}`)
    const mimeType = resp.headers.get('content-type')?.split(';')[0]?.trim() || 'audio/mpeg'
    return {
      buffer: Buffer.from(await resp.arrayBuffer()),
      mimeType,
      filename: path.basename(new URL(source).pathname) || `huobao-ref-${Date.now()}${mimeTypeToAudioExt(mimeType)}`,
    }
  }

  const filePath = source.startsWith('static/') ? getAbsolutePath(source) : path.resolve(source)
  const ext = path.extname(filePath).toLowerCase()
  return {
    buffer: fs.readFileSync(filePath),
    mimeType: audioExtToMimeType(ext),
    filename: path.basename(filePath),
  }
}

async function uploadComfyUiAudio(baseUrl: string, buffer: Buffer, mimeType: string, filename: string): Promise<string> {
  const audioData = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer
  const form = new FormData()
  form.append('image', new Blob([audioData], { type: mimeType }), filename)
  form.append('type', 'input')
  form.append('overwrite', 'true')

  const resp = await fetch(buildComfyUiUrl(baseUrl, '/upload/image'), {
    method: 'POST',
    body: form,
    signal: AbortSignal.timeout(GENERATION_TIMEOUT_MS),
  })
  if (!resp.ok) throw new Error(`ComfyUI audio upload failed: ${resp.status} ${await resp.text()}`)
  const result = await resp.json() as any
  return String(result?.name || filename)
}

async function queueComfyUiPrompt(baseUrl: string, workflow: WorkflowGraph): Promise<string> {
  const resp = await fetch(buildComfyUiUrl(baseUrl, '/prompt'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: workflow,
      client_id: `huobao-drama-audio-${Date.now()}`,
    }),
    signal: AbortSignal.timeout(GENERATION_TIMEOUT_MS),
  })
  if (!resp.ok) throw new Error(`ComfyUI prompt failed: ${resp.status} ${await resp.text()}`)
  const result = await resp.json() as any
  const promptId = result?.prompt_id || result?.promptId || result?.id
  if (!promptId) throw new Error('No ComfyUI prompt_id in audio response')
  return String(promptId)
}

async function waitForComfyUiAudio(baseUrl: string, promptId: string): Promise<string> {
  const deadline = Date.now() + GENERATION_TIMEOUT_MS
  while (Date.now() < deadline) {
    const resp = await fetch(buildComfyUiUrl(baseUrl, `/history/${encodeURIComponent(promptId)}`), {
      method: 'GET',
      signal: AbortSignal.timeout(GENERATION_TIMEOUT_MS),
    })
    if (!resp.ok) throw new Error(`ComfyUI history failed: ${resp.status} ${await resp.text()}`)
    const result = await resp.json() as any
    const parsed = parseComfyUiAudioHistoryResponse(result, baseUrl)
    if (parsed.status === 'completed' && parsed.audioUrl) return parsed.audioUrl
    if (parsed.status === 'failed') throw new Error(parsed.error || 'ComfyUI audio generation failed')
    await sleep(POLL_INTERVAL_MS)
  }
  throw new Error('ComfyUI audio generation timed out')
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
  return trimmed || 'http://127.0.0.1:8188'
}

function readUiLinkId(value: unknown): number | null {
  if (value === null || value === undefined) return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function shouldSkipUiOnlyInput(inputDef: any): boolean {
  const type = String(inputDef?.type || '').toUpperCase()
  return UI_ONLY_INPUT_TYPE_SUFFIXES.some((suffix) => type.endsWith(suffix))
}

function normalizeNodeType(classType: string): string {
  return classType.toLowerCase().replace(/[^a-z0-9]/g, '')
}

function isConnectionValue(value: unknown): value is [string, number] {
  return Array.isArray(value) && value.length >= 2 && (typeof value[0] === 'string' || typeof value[0] === 'number')
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function stripJsonSuffix(value: string): string {
  return value.replace(/\.json$/i, '')
}

function stripQueryString(value: string): string {
  return value.split(/[?#]/, 1)[0]
}

function inferAudioFormatFromUrl(url: string): string {
  try {
    const ext = path.extname(new URL(url).searchParams.get('filename') || '').replace(/^\./, '').toLowerCase()
    return ext || 'mp3'
  } catch {
    return 'mp3'
  }
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

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}
