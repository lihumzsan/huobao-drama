import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

import { parseDataUrl } from '../../utils/storage.js'
import type {
  AIConfig,
  ImageGenResponse,
  ImageGenerationRecord,
  ImagePollResponse,
  ImageProviderAdapter,
  ProviderRequest,
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

type ResolveWorkflowParams = {
  workflowKey: string
  prompt: string
  negativePrompt?: string | null
  width: number
  height: number
  imageFilenames: string[]
}

type ComfyImageRef = {
  filename: string
  subfolder?: string
  type?: string
}

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const PROJECT_ROOT = path.resolve(__dirname, '../../../..')
const WORKFLOW_ROOT = path.join(PROJECT_ROOT, 'configs', 'comfyui', 'workflows')

const DEFAULT_TEXT_TO_IMAGE_WORKFLOW = 'baseimage/图片生成/Flux2Klein文生图'
const SINGLE_IMAGE_EDIT_WORKFLOW = 'baseimage/图片编辑/qwen单图编辑'
const DOUBLE_IMAGE_EDIT_WORKFLOW = 'baseimage/图片编辑/qwen双图编辑'
const TRIPLE_IMAGE_EDIT_WORKFLOW = 'baseimage/图片编辑/qwen三图编辑'
const MULTI_IMAGE_EDIT_WORKFLOW = 'baseimage/图片编辑/Flux2多图编辑'
const MAX_REFERENCE_IMAGES = 5
const SEED_CONTROL_VALUES = new Set(['fixed', 'randomize', 'increment', 'decrement'])
const UI_ONLY_INPUT_TYPE_SUFFIXES = ['UPLOAD', '_UI']
const OUTPUT_NODE_TYPES = new Set(['saveimage'])

export class ComfyUiImageAdapter implements ImageProviderAdapter {
  provider = 'comfyui'

  async buildGenerateRequest(config: AIConfig, record: ImageGenerationRecord): Promise<ProviderRequest> {
    const refs = parseReferenceImages(record.referenceImages).slice(0, MAX_REFERENCE_IMAGES)
    const imageFilenames = await Promise.all(refs.map((ref, index) => uploadComfyUiImage(config.baseUrl, ref, index)))
    const { width, height } = parseSize(record.size)
    const workflowKey = selectComfyUiImageWorkflow(imageFilenames.length)
    const workflow = resolveComfyUiImageWorkflow({
      workflowKey,
      prompt: record.prompt || 'Generate an image',
      width,
      height,
      imageFilenames,
    })

    return {
      url: buildComfyUiUrl(config.baseUrl, '/prompt'),
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: {
        prompt: workflow,
        client_id: `huobao-drama-${record.id}`,
      },
    }
  }

  parseGenerateResponse(result: any): ImageGenResponse {
    const taskId = result?.prompt_id || result?.promptId || result?.id
    if (!taskId) {
      const error = result?.error?.message || result?.error || 'No ComfyUI prompt_id in response'
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

  parsePollResponse(result: any, config?: AIConfig): ImagePollResponse {
    return parseComfyUiHistoryResponse(result, config?.baseUrl || '')
  }

  extractImageUrl(result: any): string | null {
    return result?.imageUrl || result?.image_url || null
  }

  extractImageBase64(): { data: string; mimeType: string } | null {
    return null
  }
}

export function selectComfyUiImageWorkflow(referenceCount: number): string {
  if (referenceCount <= 0) return DEFAULT_TEXT_TO_IMAGE_WORKFLOW
  if (referenceCount === 1) return SINGLE_IMAGE_EDIT_WORKFLOW
  if (referenceCount === 2) return DOUBLE_IMAGE_EDIT_WORKFLOW
  if (referenceCount === 3) return TRIPLE_IMAGE_EDIT_WORKFLOW
  return MULTI_IMAGE_EDIT_WORKFLOW
}

export function resolveComfyUiImageWorkflow(params: ResolveWorkflowParams): WorkflowGraph {
  const graph = readWorkflowGraph(params.workflowKey)
  applyPromptInjection(graph, params.prompt, params.negativePrompt || '')
  applyDimensionInjection(graph, params.width, params.height)
  applyImageInjection(graph, params.imageFilenames)
  pruneUnreachableFromOutputs(graph)
  assignRandomSeeds(graph)
  return graph
}

export function parseComfyUiHistoryResponse(result: any, baseUrl: string): ImagePollResponse {
  const entries = Object.values(result || {}) as any[]
  if (!entries.length) return { status: 'processing' }

  const history = entries[0]
  const statusText = String(history?.status?.status_str || history?.status?.status || '').toLowerCase()
  if (statusText.includes('error') || statusText.includes('failed')) {
    const message = history?.status?.messages?.flat?.()?.join('\n') || 'ComfyUI generation failed'
    return { status: 'failed', error: String(message) }
  }

  const image = findFirstOutputImage(history?.outputs)
  if (image) {
    return {
      status: 'completed',
      imageUrl: buildComfyUiViewUrl(baseUrl, image),
    }
  }

  if (history?.status?.completed === false) return { status: 'processing' }
  return { status: 'processing' }
}

export function buildComfyUiViewUrl(baseUrl: string, image: ComfyImageRef): string {
  const url = new URL(buildComfyUiUrl(baseUrl, '/view'))
  url.searchParams.set('filename', image.filename)
  url.searchParams.set('subfolder', image.subfolder || '')
  url.searchParams.set('type', image.type || 'output')
  return url.toString()
}

function readWorkflowGraph(workflowKey: string): WorkflowGraph {
  const filePath = resolveWorkflowPath(workflowKey)
  const parsed = JSON.parse(fs.readFileSync(filePath, 'utf8')) as any
  if (isApiWorkflowGraph(parsed)) return normalizeApiWorkflowGraph(parsed)
  if (Array.isArray(parsed?.nodes) && Array.isArray(parsed?.links)) {
    return convertUiWorkflowToApiGraph(parsed)
  }
  throw new Error(`Unsupported ComfyUI workflow format: ${workflowKey}`)
}

function resolveWorkflowPath(workflowKey: string): string {
  const safeKey = workflowKey.replace(/\\/g, '/').replace(/^\//, '')
  if (safeKey.includes('..')) throw new Error(`Unsafe ComfyUI workflow key: ${workflowKey}`)
  const filePath = path.resolve(WORKFLOW_ROOT, `${safeKey}.json`)
  if (!filePath.startsWith(WORKFLOW_ROOT + path.sep)) {
    throw new Error(`ComfyUI workflow path escaped root: ${workflowKey}`)
  }
  if (!fs.existsSync(filePath)) throw new Error(`ComfyUI workflow not found: ${workflowKey}`)
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
      _meta: {
        id: nodeId,
        ...(rawNode.title ? { title: String(rawNode.title) } : {}),
      },
    }
  }

  return graph
}

function applyPromptInjection(graph: WorkflowGraph, prompt: string, negativePrompt: string): void {
  for (const node of Object.values(graph)) {
    const classType = node.class_type.toLowerCase()
    for (const inputName of Object.keys(node.inputs)) {
      const field = inputName.toLowerCase()
      if (negativePrompt && (field === 'negative' || field === 'negative_prompt')) {
        assignStringInput(graph, node, inputName, negativePrompt)
      }
      if (field === 'prompt' || field === 'text' || field === 'value') {
        if (classType.includes('cliptextencode')
          || classType.includes('textencodeqwen')
          || classType.includes('jjktext')
          || classType.includes('primitivestring')) {
          assignStringInput(graph, node, inputName, prompt)
        }
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

function applyDimensionInjection(graph: WorkflowGraph, width: number, height: number): void {
  const aspectRatio = formatAspectRatio(width, height)
  const longestSide = Math.max(width, height)
  for (const node of Object.values(graph)) {
    assignNumericInput(graph, node, 'width', width)
    assignNumericInput(graph, node, 'height', height)
    assignNumericInput(graph, node, 'scale_to_length', longestSide)
    assignStringInputIfPresent(graph, node, 'aspect_ratio', aspectRatio)
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

function assignStringInputIfPresent(graph: WorkflowGraph, node: WorkflowNode, inputName: string, value: string): void {
  if (!Object.prototype.hasOwnProperty.call(node.inputs, inputName)) return
  assignStringInput(graph, node, inputName, value)
}

function applyImageInjection(graph: WorkflowGraph, imageFilenames: string[]): void {
  const loadNodes = Object.entries(graph)
    .filter(([, node]) => node.class_type.toLowerCase().includes('loadimage'))
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

function findFirstOutputImage(outputs: any): ComfyImageRef | null {
  if (!outputs || typeof outputs !== 'object') return null
  const outputEntries = Object.entries(outputs).sort(([left], [right]) => Number(left) - Number(right))
  for (const [, output] of outputEntries) {
    const images = (output as any)?.images
    if (!Array.isArray(images) || !images.length) continue
    const image = images[0]
    if (image?.filename) {
      return {
        filename: String(image.filename),
        subfolder: image.subfolder ? String(image.subfolder) : '',
        type: image.type ? String(image.type) : 'output',
      }
    }
  }
  return null
}

async function uploadComfyUiImage(baseUrl: string, source: string, index: number): Promise<string> {
  const { buffer, mimeType } = await loadImageBuffer(source)
  const filename = `huobao-ref-${Date.now()}-${index}${mimeTypeToExt(mimeType)}`
  const imageData = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer
  const form = new FormData()
  form.append('image', new Blob([imageData], { type: mimeType }), filename)
  form.append('type', 'input')
  form.append('overwrite', 'true')

  const resp = await fetch(buildComfyUiUrl(baseUrl, '/upload/image'), {
    method: 'POST',
    body: form,
  })
  if (!resp.ok) throw new Error(`ComfyUI upload failed: ${resp.status} ${await resp.text()}`)
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
  if (!resp.ok) throw new Error(`Reference image fetch failed: ${resp.status}`)
  const mimeType = resp.headers.get('content-type')?.split(';')[0]?.trim() || 'image/png'
  return {
    buffer: Buffer.from(await resp.arrayBuffer()),
    mimeType,
  }
}

function parseReferenceImages(raw: string | null | undefined): string[] {
  if (!raw) return []
  try {
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed.map((item) => String(item || '')).filter(Boolean) : []
  } catch {
    return []
  }
}

function parseSize(size: string | null | undefined): { width: number; height: number } {
  const [width, height] = String(size || '1920x1080').split('x').map(Number)
  return {
    width: Number.isFinite(width) && width > 0 ? Math.round(width) : 1920,
    height: Number.isFinite(height) && height > 0 ? Math.round(height) : 1080,
  }
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
  return UI_ONLY_INPUT_TYPE_SUFFIXES.some((suffix) => type.endsWith(suffix))
}

function setStringValueOnNode(node: WorkflowNode, value: string): void {
  for (const field of ['value', 'text', 'prompt', 'string', 'input_string']) {
    if (Object.prototype.hasOwnProperty.call(node.inputs, field) && !isConnectionValue(node.inputs[field])) {
      node.inputs[field] = value
      return
    }
  }
}

function setNumericValueOnNode(node: WorkflowNode, value: number): void {
  for (const field of ['value', 'number', 'int', 'width', 'height']) {
    if (Object.prototype.hasOwnProperty.call(node.inputs, field) && !isConnectionValue(node.inputs[field])) {
      node.inputs[field] = value
      return
    }
  }
}

function isConnectionValue(value: unknown): value is [string, number] {
  return Array.isArray(value) && value.length >= 2 && (typeof value[0] === 'string' || typeof value[0] === 'number')
}

function formatAspectRatio(width: number, height: number): string {
  const divisor = gcd(width, height)
  return `${Math.round(width / divisor)}:${Math.round(height / divisor)}`
}

function gcd(left: number, right: number): number {
  let a = Math.abs(left)
  let b = Math.abs(right)
  while (b > 0) {
    const next = a % b
    a = b
    b = next
  }
  return a || 1
}

function normalizeNodeType(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '')
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

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}
