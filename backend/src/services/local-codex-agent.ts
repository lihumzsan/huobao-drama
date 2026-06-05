import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { and, eq } from 'drizzle-orm'
import { z } from 'zod'
import { now } from '../utils/response.js'
import { runCodexCliJson } from './codex-cli.js'

export type LocalCodexAgentType =
  | 'script_rewriter'
  | 'extractor'
  | 'storyboard_breaker'
  | 'voice_assigner'
  | 'grid_prompt_generator'

export const LOCAL_CODEX_MODEL = 'gpt-5.5'
export const LOCAL_CODEX_REASONING_EFFORT = 'xhigh'
export const LOCAL_CODEX_PROVIDER = 'codex'
export const LOCAL_CODEX_CONFIG_ID = 'local-codex'

export const LOCAL_CODEX_VIRTUAL_CONFIG = {
  id: LOCAL_CODEX_CONFIG_ID,
  service_type: 'text',
  provider: LOCAL_CODEX_PROVIDER,
  name: '本机 Codex 文本服务',
  base_url: 'local-codex://cli',
  api_key: '',
  model: [LOCAL_CODEX_MODEL],
  priority: 1000,
  is_default: true,
  is_active: true,
  is_virtual: true,
  settings: {
    backend: 'codex_cli',
    reasoning_effort: LOCAL_CODEX_REASONING_EFFORT,
    timeout_seconds: 900,
  },
} as const

const nonEmptyString = z.string().trim().min(1)
const optionalText = z.string().optional().default('')
const optionalNullableId = z.number().int().positive().nullable().optional()

const scriptRewriteSchema = z.object({
  content: nonEmptyString,
  notes: z.string().optional(),
})

const extractedCharacterSchema = z.object({
  name: nonEmptyString,
  role: optionalText,
  description: optionalText,
  appearance: optionalText,
  personality: optionalText,
})

const extractedSceneSchema = z.object({
  location: nonEmptyString,
  time: z.string().optional().default(''),
  prompt: z.string().optional().default(''),
})

const extractorSchema = z.object({
  characters: z.array(extractedCharacterSchema),
  scenes: z.array(extractedSceneSchema),
})

const storyboardSchema = z.object({
  shot_number: z.number().int().positive(),
  title: z.string().optional().default(''),
  shot_type: z.string().optional().default(''),
  angle: z.string().optional().default(''),
  movement: z.string().optional().default(''),
  location: z.string().optional().default(''),
  time: z.string().optional().default(''),
  action: z.string().optional().default(''),
  dialogue: z.string().optional().default(''),
  description: nonEmptyString,
  result: z.string().optional().default(''),
  atmosphere: z.string().optional().default(''),
  image_prompt: z.string().optional().default(''),
  video_prompt: nonEmptyString,
  bgm_prompt: z.string().optional().default(''),
  sound_effect: z.string().optional().default(''),
  duration: z.number().int().positive().max(120).optional().default(10),
  scene_id: optionalNullableId,
  character_ids: z.array(z.number().int().positive()).optional().default([]),
})

const storyboardBreakerSchema = z.object({
  storyboards: z.array(storyboardSchema).min(1),
})

const voiceAssignmentSchema = z.object({
  character_id: z.number().int().positive(),
  voice_id: nonEmptyString,
  reason: z.string().optional().default(''),
})

const voiceAssignerSchema = z.object({
  assignments: z.array(voiceAssignmentSchema),
})

const gridCellPromptSchema = z.object({
  shot_number: z.number().int().positive(),
  frame_type: nonEmptyString,
  prompt: nonEmptyString,
})

const gridPromptGeneratorSchema = z.object({
  grid_prompt: nonEmptyString,
  cell_prompts: z.array(gridCellPromptSchema).min(1),
})

export const LOCAL_CODEX_OUTPUT_SCHEMAS = {
  script_rewriter: scriptRewriteSchema,
  extractor: extractorSchema,
  storyboard_breaker: storyboardBreakerSchema,
  voice_assigner: voiceAssignerSchema,
  grid_prompt_generator: gridPromptGeneratorSchema,
} as const

export const LOCAL_CODEX_JSON_SCHEMAS: Record<LocalCodexAgentType | 'health_check', Record<string, unknown>> = {
  health_check: {
    type: 'object',
    properties: { ok: { type: 'boolean' } },
    required: ['ok'],
    additionalProperties: false,
  },
  script_rewriter: {
    type: 'object',
    properties: {
      content: { type: 'string', minLength: 1 },
      notes: { type: 'string' },
    },
    required: ['content'],
    additionalProperties: false,
  },
  extractor: {
    type: 'object',
    properties: {
      characters: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            name: { type: 'string', minLength: 1 },
            role: { type: 'string' },
            description: { type: 'string' },
            appearance: { type: 'string' },
            personality: { type: 'string' },
          },
          required: ['name'],
          additionalProperties: false,
        },
      },
      scenes: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            location: { type: 'string', minLength: 1 },
            time: { type: 'string' },
            prompt: { type: 'string' },
          },
          required: ['location'],
          additionalProperties: false,
        },
      },
    },
    required: ['characters', 'scenes'],
    additionalProperties: false,
  },
  storyboard_breaker: {
    type: 'object',
    properties: {
      storyboards: {
        type: 'array',
        minItems: 1,
        items: {
          type: 'object',
          properties: {
            shot_number: { type: 'integer', minimum: 1 },
            title: { type: 'string' },
            shot_type: { type: 'string' },
            angle: { type: 'string' },
            movement: { type: 'string' },
            location: { type: 'string' },
            time: { type: 'string' },
            action: { type: 'string' },
            dialogue: { type: 'string' },
            description: { type: 'string', minLength: 1 },
            result: { type: 'string' },
            atmosphere: { type: 'string' },
            image_prompt: { type: 'string' },
            video_prompt: { type: 'string', minLength: 1 },
            bgm_prompt: { type: 'string' },
            sound_effect: { type: 'string' },
            duration: { type: 'integer', minimum: 1, maximum: 120 },
            scene_id: { anyOf: [{ type: 'integer', minimum: 1 }, { type: 'null' }] },
            character_ids: { type: 'array', items: { type: 'integer', minimum: 1 } },
          },
          required: ['shot_number', 'description', 'video_prompt'],
          additionalProperties: false,
        },
      },
    },
    required: ['storyboards'],
    additionalProperties: false,
  },
  voice_assigner: {
    type: 'object',
    properties: {
      assignments: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            character_id: { type: 'integer', minimum: 1 },
            voice_id: { type: 'string', minLength: 1 },
            reason: { type: 'string' },
          },
          required: ['character_id', 'voice_id'],
          additionalProperties: false,
        },
      },
    },
    required: ['assignments'],
    additionalProperties: false,
  },
  grid_prompt_generator: {
    type: 'object',
    properties: {
      grid_prompt: { type: 'string', minLength: 1 },
      cell_prompts: {
        type: 'array',
        minItems: 1,
        items: {
          type: 'object',
          properties: {
            shot_number: { type: 'integer', minimum: 1 },
            frame_type: { type: 'string', minLength: 1 },
            prompt: { type: 'string', minLength: 1 },
          },
          required: ['shot_number', 'frame_type', 'prompt'],
          additionalProperties: false,
        },
      },
    },
    required: ['grid_prompt', 'cell_prompts'],
    additionalProperties: false,
  },
}

export type LocalCodexPayload<T extends LocalCodexAgentType = LocalCodexAgentType> =
  z.infer<(typeof LOCAL_CODEX_OUTPUT_SCHEMAS)[T]>

export function validateLocalCodexPayload<T extends LocalCodexAgentType>(agentType: T, value: unknown): LocalCodexPayload<T> {
  const schema = LOCAL_CODEX_OUTPUT_SCHEMAS[agentType]
  const parsed = schema.safeParse(value)
  if (!parsed.success) {
    throw new Error(`Codex 输出格式不符合 ${agentType}: ${parsed.error.message}`)
  }
  return parsed.data as LocalCodexPayload<T>
}

export interface LocalCodexAgentResult {
  type: 'done'
  text: string
  toolCalls: Array<{ toolName: string; args: Record<string, unknown> }>
  toolResults: Array<{ toolName: string; result: string }>
}

export interface RunLocalCodexAgentOptions {
  agentType: LocalCodexAgentType
  message: string
  episodeId: number
  dramaId: number
}

export interface RunLocalCodexGridPromptOptions {
  episodeId: number
  dramaId: number
  storyboardIds: number[]
  rows: number
  cols: number
  mode: string
  referenceLegend?: string
}

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const projectRoot = path.resolve(__dirname, '../../..')

export function shouldUseLocalCodexText() {
  return true
}

export async function testLocalCodexCli() {
  return runCodexCliJson<{ ok: boolean }>({
    cwd: projectRoot,
    prompt: '返回 JSON：{"ok":true}。不要添加解释。',
    schema: LOCAL_CODEX_JSON_SCHEMAS.health_check,
    timeoutMs: 5 * 60 * 1000,
    validate: value => z.object({ ok: z.boolean() }).parse(value),
  })
}

export async function runLocalCodexAgent(options: RunLocalCodexAgentOptions): Promise<LocalCodexAgentResult> {
  switch (options.agentType) {
    case 'script_rewriter': return runScriptRewriter(options)
    case 'extractor': return runExtractor(options)
    case 'storyboard_breaker': return runStoryboardBreaker(options)
    case 'voice_assigner': return runVoiceAssigner(options)
    case 'grid_prompt_generator': return runGenericGridPromptGenerator(options)
    default: throw new Error(`Unsupported local Codex agent: ${options.agentType}`)
  }
}

export async function runLocalCodexGridPrompt(options: RunLocalCodexGridPromptOptions) {
  const { db, schema } = await import('../db/index.js')
  const shots = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.episodeId, options.episodeId)).all()
    .filter(sb => options.storyboardIds.includes(sb.id))
    .map(sb => ({
      shot_number: sb.storyboardNumber,
      description: sb.description || sb.title || '',
      shot_type: sb.shotType || '',
      dialogue: sb.dialogue || '',
      location: sb.location || '',
      time: sb.time || '',
    }))

  if (!shots.length) throw new Error('No storyboards found for local Codex grid prompt')

  const prompt = buildPrompt('grid_prompt_generator', [
    '你是火宝短剧的图片提示词工程师。根据镜头信息生成宫格图提示词。',
    `行数：${options.rows}`,
    `列数：${options.cols}`,
    `模式：${options.mode}`,
    options.referenceLegend ? `参考图映射：${options.referenceLegend}` : '',
    `必须严格生成 exactly ${options.rows * options.cols} visible panels，不要合并格子，不要缺格。`,
    '提示词尽量使用英文，保留“格1/格2”等格子编号。',
    `镜头信息 JSON：${JSON.stringify(shots)}`,
  ].join('\n'))

  const payload = await runCodexForAgent('grid_prompt_generator', prompt)
  return validateLocalCodexPayload('grid_prompt_generator', payload)
}

async function runScriptRewriter(options: RunLocalCodexAgentOptions): Promise<LocalCodexAgentResult> {
  const { db, schema } = await import('../db/index.js')
  const [episode] = db.select().from(schema.episodes).where(eq(schema.episodes.id, options.episodeId)).all()
  if (!episode) throw new Error(`Episode not found (id=${options.episodeId})`)
  const source = episode.content || episode.scriptContent
  if (!source) throw new Error(`Episode has no content (id=${options.episodeId})`)

  const prompt = buildPrompt('script_rewriter', [
    '你是专业编剧，擅长将小说改编为短剧剧本。',
    `用户要求：${options.message}`,
    '请改写为格式化剧本，格式要求：',
    '- 场景头：## S编号 | 内景/外景 · 地点 | 时间段',
    '- 动作描写：自然段落，不包含镜头语言',
    '- 对白：角色名：（状态/表情）台词内容',
    '- 每个场景 30-60 秒内容',
    `原始内容：\n${source}`,
  ].join('\n'))

  const payload = validateLocalCodexPayload('script_rewriter', await runCodexForAgent('script_rewriter', prompt))
  db.update(schema.episodes)
    .set({ scriptContent: payload.content, updatedAt: now() })
    .where(eq(schema.episodes.id, options.episodeId))
    .run()

  return agentResult('script_rewriter', `剧本已保存，字数 ${payload.content.length}`, payload)
}

async function runExtractor(options: RunLocalCodexAgentOptions): Promise<LocalCodexAgentResult> {
  const { db, schema } = await import('../db/index.js')
  const [episode] = db.select().from(schema.episodes).where(eq(schema.episodes.id, options.episodeId)).all()
  if (!episode) throw new Error(`Episode not found (id=${options.episodeId})`)
  const script = episode.scriptContent || episode.content
  if (!script) throw new Error(`Episode has no script content (id=${options.episodeId})`)

  const existingCharacters = db.select().from(schema.characters)
    .where(eq(schema.characters.dramaId, options.dramaId)).all()
    .filter(c => !c.deletedAt)
  const existingScenes = db.select().from(schema.scenes)
    .where(eq(schema.scenes.dramaId, options.dramaId)).all()
    .filter(s => !s.deletedAt)

  const prompt = buildPrompt('extractor', [
    '你是制片助理。请从当前集剧本中提取角色和场景，并与已有数据去重。',
    `用户要求：${options.message}`,
    '角色按 name 精确匹配；场景按 location + time 精确匹配。',
    '只提取当前集真实出现或明确提及且对叙事有效的角色和场景。',
    `已有角色 JSON：${JSON.stringify(existingCharacters)}`,
    `已有场景 JSON：${JSON.stringify(existingScenes)}`,
    `剧本：\n${script}`,
  ].join('\n'))

  const payload = validateLocalCodexPayload('extractor', await runCodexForAgent('extractor', prompt))
  const charResult = saveExtractedCharacters(db, schema, options.episodeId, options.dramaId, payload.characters)
  const sceneResult = saveExtractedScenes(db, schema, options.episodeId, options.dramaId, payload.scenes)

  return agentResult(
    'extractor',
    `角色保存完成：新增 ${charResult.created}，合并 ${charResult.merged}；场景保存完成：新增 ${sceneResult.created}，复用 ${sceneResult.reused}`,
    { ...payload, character_result: charResult, scene_result: sceneResult },
  )
}

async function runStoryboardBreaker(options: RunLocalCodexAgentOptions): Promise<LocalCodexAgentResult> {
  const { db, schema } = await import('../db/index.js')
  const context = readStoryboardContext(db, schema, options.episodeId, options.dramaId)

  const prompt = buildPrompt('storyboard_breaker', [
    '你是资深影视分镜师。请将剧本拆解为镜头序列，并生成后续图片、视频、配音、音效、合成都可用的完整字段。',
    `用户要求：${options.message}`,
    '每个镜头优先 10-15 秒。scene_id 必须来自 scenes；character_ids 必须来自 characters；无角色镜头使用空数组。',
    'video_prompt 按 3 秒为一段，可使用 <location>、<role>、<voice>、<n> 标签。',
    `上下文 JSON：${JSON.stringify(context)}`,
  ].join('\n'))

  const payload = validateLocalCodexPayload('storyboard_breaker', await runCodexForAgent('storyboard_breaker', prompt))
  const saveResult = saveStoryboards(db, schema, options.episodeId, payload.storyboards)

  return agentResult('storyboard_breaker', `分镜已保存，共 ${saveResult.count} 个镜头，总时长 ${saveResult.total_duration} 秒`, {
    ...payload,
    save_result: saveResult,
  })
}

async function runVoiceAssigner(options: RunLocalCodexAgentOptions): Promise<LocalCodexAgentResult> {
  const { db, schema } = await import('../db/index.js')
  const characters = db.select().from(schema.characters)
    .where(eq(schema.characters.dramaId, options.dramaId)).all()
    .filter(c => !c.deletedAt)
    .map(c => ({
      id: c.id,
      name: c.name,
      role: c.role || '',
      personality: c.personality || '',
      description: c.description || '',
      current_voice: c.voiceStyle || '',
    }))
  const voices = readVoicesForEpisode(db, schema, options.episodeId)

  const prompt = buildPrompt('voice_assigner', [
    '你是配音导演。请为每个角色选择最合适的 voice_id。',
    `用户要求：${options.message}`,
    '只能从 voices 列表中选择 voice_id；每个角色都必须分配。',
    `characters JSON：${JSON.stringify(characters)}`,
    `voices JSON：${JSON.stringify(voices)}`,
  ].join('\n'))

  const payload = validateLocalCodexPayload('voice_assigner', await runCodexForAgent('voice_assigner', prompt))
  const voiceIds = new Set(voices.map((voice: any) => voice.id))
  for (const assignment of payload.assignments) {
    if (!voiceIds.has(assignment.voice_id)) {
      throw new Error(`Codex 输出了不可用音色: ${assignment.voice_id}`)
    }
    const provider = voices.find((voice: any) => voice.id === assignment.voice_id)?.provider || 'minimax'
    db.update(schema.characters)
      .set({ voiceStyle: assignment.voice_id, voiceProvider: provider, voiceSampleUrl: null, updatedAt: now() })
      .where(eq(schema.characters.id, assignment.character_id))
      .run()
  }

  return agentResult('voice_assigner', `音色已分配，共 ${payload.assignments.length} 个角色`, payload)
}

async function runGenericGridPromptGenerator(options: RunLocalCodexAgentOptions): Promise<LocalCodexAgentResult> {
  const { db, schema } = await import('../db/index.js')
  const characters = db.select().from(schema.characters)
    .where(eq(schema.characters.dramaId, options.dramaId)).all()
    .filter(c => !c.deletedAt)
    .map(c => ({ id: c.id, name: c.name, appearance: c.appearance || '', description: c.description || '' }))
  const scenes = db.select().from(schema.scenes)
    .where(eq(schema.scenes.dramaId, options.dramaId)).all()
    .filter(s => !s.deletedAt)
    .map(s => ({ id: s.id, location: s.location, time: s.time || '', prompt: s.prompt || '' }))

  const prompt = buildPrompt('grid_prompt_generator', [
    '你是专业 AI 图像提示词工程师。请根据用户要求生成图片提示词。',
    `用户要求：${options.message}`,
    '如果没有明确宫格行列，默认返回 1 个 cell_prompts 项。',
    `characters JSON：${JSON.stringify(characters)}`,
    `scenes JSON：${JSON.stringify(scenes)}`,
  ].join('\n'))

  const payload = validateLocalCodexPayload('grid_prompt_generator', await runCodexForAgent('grid_prompt_generator', prompt))
  return agentResult('grid_prompt_generator', '图片提示词已生成', payload)
}

async function runCodexForAgent(agentType: LocalCodexAgentType, prompt: string) {
  return runCodexCliJson({
    cwd: projectRoot,
    prompt,
    schema: LOCAL_CODEX_JSON_SCHEMAS[agentType],
    timeoutMs: LOCAL_CODEX_VIRTUAL_CONFIG.settings.timeout_seconds * 1000,
  })
}

function buildPrompt(agentType: LocalCodexAgentType, body: string) {
  return [
    `你正在为火宝短剧执行 ${agentType} 文本处理任务。`,
    '你只负责分析和生成结构化 JSON；不要尝试修改文件、运行数据库命令或保存任何内容。',
    '最终答案必须严格符合传入的 JSON Schema。',
    body,
  ].join('\n\n')
}

function agentResult(agentType: LocalCodexAgentType, text: string, payload: unknown): LocalCodexAgentResult {
  return {
    type: 'done',
    text,
    toolCalls: [{ toolName: 'local_codex_generate', args: { agentType, model: LOCAL_CODEX_MODEL, reasoning_effort: LOCAL_CODEX_REASONING_EFFORT } }],
    toolResults: [{ toolName: 'local_codex_apply', result: JSON.stringify(payload) }],
  }
}

function saveExtractedCharacters(db: any, schema: any, episodeId: number, dramaId: number, characters: z.infer<typeof extractedCharacterSchema>[]) {
  const ts = now()
  const results = { created: 0, merged: 0 }
  for (const char of characters) {
    const existing = db.select().from(schema.characters)
      .where(eq(schema.characters.dramaId, dramaId)).all()
      .filter((row: any) => !row.deletedAt)
      .find((row: any) => row.name === char.name)

    if (existing) {
      db.update(schema.characters).set({
        role: char.role || existing.role,
        description: char.description || existing.description,
        appearance: char.appearance || existing.appearance,
        personality: char.personality || existing.personality,
        updatedAt: ts,
      }).where(eq(schema.characters.id, existing.id)).run()
      linkCharacterToEpisode(db, schema, episodeId, existing.id)
      results.merged++
    } else {
      const res = db.insert(schema.characters).values({
        dramaId,
        name: char.name,
        role: char.role || '',
        description: char.description || '',
        appearance: char.appearance || '',
        personality: char.personality || '',
        createdAt: ts,
        updatedAt: ts,
      }).run()
      linkCharacterToEpisode(db, schema, episodeId, Number(res.lastInsertRowid))
      results.created++
    }
  }
  return results
}

function saveExtractedScenes(db: any, schema: any, episodeId: number, dramaId: number, scenes: z.infer<typeof extractedSceneSchema>[]) {
  const ts = now()
  const results = { created: 0, reused: 0 }
  for (const scene of scenes) {
    const existing = db.select().from(schema.scenes)
      .where(eq(schema.scenes.dramaId, dramaId)).all()
      .filter((row: any) => !row.deletedAt)
      .find((row: any) => row.location === scene.location && row.time === (scene.time || ''))

    if (existing) {
      linkSceneToEpisode(db, schema, episodeId, existing.id)
      results.reused++
    } else {
      const res = db.insert(schema.scenes).values({
        dramaId,
        location: scene.location,
        time: scene.time || '',
        prompt: scene.prompt || scene.location,
        createdAt: ts,
        updatedAt: ts,
      }).run()
      linkSceneToEpisode(db, schema, episodeId, Number(res.lastInsertRowid))
      results.created++
    }
  }
  return results
}

function linkCharacterToEpisode(db: any, schema: any, episodeId: number, characterId: number) {
  const existing = db.select().from(schema.episodeCharacters)
    .where(and(eq(schema.episodeCharacters.episodeId, episodeId), eq(schema.episodeCharacters.characterId, characterId))).all()
  if (!existing.length) {
    db.insert(schema.episodeCharacters).values({ episodeId, characterId, createdAt: now() }).run()
  }
}

function linkSceneToEpisode(db: any, schema: any, episodeId: number, sceneId: number) {
  const existing = db.select().from(schema.episodeScenes)
    .where(and(eq(schema.episodeScenes.episodeId, episodeId), eq(schema.episodeScenes.sceneId, sceneId))).all()
  if (!existing.length) {
    db.insert(schema.episodeScenes).values({ episodeId, sceneId, createdAt: now() }).run()
  }
}

function readStoryboardContext(db: any, schema: any, episodeId: number, dramaId: number) {
  const [episode] = db.select().from(schema.episodes).where(eq(schema.episodes.id, episodeId)).all()
  if (!episode) throw new Error(`Episode not found (id=${episodeId})`)
  const script = episode.scriptContent || episode.content
  if (!script) throw new Error(`Episode has no script (id=${episodeId})`)

  const characterLinks = db.select().from(schema.episodeCharacters)
    .where(eq(schema.episodeCharacters.episodeId, episodeId)).all()
  const sceneLinks = db.select().from(schema.episodeScenes)
    .where(eq(schema.episodeScenes.episodeId, episodeId)).all()
  const linkedCharacterIds = new Set(characterLinks.map((link: any) => link.characterId))
  const linkedSceneIds = new Set(sceneLinks.map((link: any) => link.sceneId))

  const characters = db.select().from(schema.characters)
    .where(eq(schema.characters.dramaId, dramaId)).all()
    .filter((char: any) => !char.deletedAt)
    .filter((char: any) => !linkedCharacterIds.size || linkedCharacterIds.has(char.id))
    .map((char: any) => ({
      id: char.id,
      name: char.name,
      role: char.role || '',
      description: char.description || '',
      appearance: char.appearance || '',
      personality: char.personality || '',
    }))
  const scenes = db.select().from(schema.scenes)
    .where(eq(schema.scenes.dramaId, dramaId)).all()
    .filter((scene: any) => !scene.deletedAt)
    .filter((scene: any) => !linkedSceneIds.size || linkedSceneIds.has(scene.id))
    .map((scene: any) => ({
      id: scene.id,
      location: scene.location,
      time: scene.time || '',
      prompt: scene.prompt || '',
    }))

  return { episode: { id: episode.id, title: episode.title }, script, characters, scenes }
}

function saveStoryboards(db: any, schema: any, episodeId: number, storyboards: z.infer<typeof storyboardSchema>[]) {
  const episodeSceneIds = new Set(db.select().from(schema.episodeScenes)
    .where(eq(schema.episodeScenes.episodeId, episodeId)).all()
    .map((link: any) => link.sceneId))
  const episodeCharacterIds = new Set(db.select().from(schema.episodeCharacters)
    .where(eq(schema.episodeCharacters.episodeId, episodeId)).all()
    .map((link: any) => link.characterId))

  for (const storyboard of storyboards) {
    if (storyboard.scene_id != null && !episodeSceneIds.has(storyboard.scene_id)) {
      throw new Error(`scene_id ${storyboard.scene_id} 不属于当前集`)
    }
    const invalidCharacterIds = (storyboard.character_ids || []).filter(id => !episodeCharacterIds.has(id))
    if (invalidCharacterIds.length) {
      throw new Error(`character_ids 不属于当前集: ${invalidCharacterIds.join(', ')}`)
    }
  }

  const ts = now()
  const existingStoryboardIds = db.select().from(schema.storyboards)
    .where(eq(schema.storyboards.episodeId, episodeId)).all()
    .map((storyboard: any) => storyboard.id)
  for (const storyboardId of existingStoryboardIds) {
    db.delete(schema.storyboardCharacters)
      .where(eq(schema.storyboardCharacters.storyboardId, storyboardId))
      .run()
  }
  db.delete(schema.storyboards).where(eq(schema.storyboards.episodeId, episodeId)).run()

  let totalDuration = 0
  for (const storyboard of storyboards) {
    const res = db.insert(schema.storyboards).values({
      episodeId,
      storyboardNumber: storyboard.shot_number,
      title: storyboard.title,
      shotType: storyboard.shot_type,
      angle: storyboard.angle,
      movement: storyboard.movement,
      location: storyboard.location,
      time: storyboard.time,
      action: storyboard.action,
      dialogue: storyboard.dialogue,
      description: storyboard.description,
      result: storyboard.result,
      atmosphere: storyboard.atmosphere,
      imagePrompt: storyboard.image_prompt,
      videoPrompt: storyboard.video_prompt,
      bgmPrompt: storyboard.bgm_prompt,
      soundEffect: storyboard.sound_effect,
      sceneId: storyboard.scene_id,
      duration: storyboard.duration || 10,
      createdAt: ts,
      updatedAt: ts,
    }).run()
    syncStoryboardCharacters(db, schema, Number(res.lastInsertRowid), storyboard.character_ids || [])
    totalDuration += storyboard.duration || 10
  }

  db.update(schema.episodes)
    .set({ duration: Math.ceil(totalDuration / 60), updatedAt: ts })
    .where(eq(schema.episodes.id, episodeId))
    .run()

  return { count: storyboards.length, total_duration: totalDuration }
}

function syncStoryboardCharacters(db: any, schema: any, storyboardId: number, characterIds: number[]) {
  db.delete(schema.storyboardCharacters)
    .where(eq(schema.storyboardCharacters.storyboardId, storyboardId))
    .run()
  for (const characterId of [...new Set(characterIds.filter(Boolean))]) {
    db.insert(schema.storyboardCharacters).values({ storyboardId, characterId }).run()
  }
}

function readVoicesForEpisode(db: any, schema: any, episodeId: number) {
  const [episode] = db.select().from(schema.episodes).where(eq(schema.episodes.id, episodeId)).all()
  let provider = 'minimax'
  if (episode?.audioConfigId) {
    const [config] = db.select().from(schema.aiServiceConfigs).where(eq(schema.aiServiceConfigs.id, episode.audioConfigId)).all()
    provider = config?.provider || provider
  }

  const rows = db.select().from(schema.aiVoices).where(eq(schema.aiVoices.provider, provider)).all()
  if (rows.length) {
    return rows.map((voice: any) => ({
      id: voice.voiceId,
      name: voice.voiceName,
      description: voice.description || '',
      language: voice.language || '',
      provider,
    }))
  }

  return [
    { id: 'alloy', name: 'Alloy', description: '平衡自然', language: '多语言', provider },
    { id: 'echo', name: 'Echo', description: '低沉稳重', language: '多语言', provider },
    { id: 'fable', name: 'Fable', description: '温暖富有表现力', language: '多语言', provider },
    { id: 'onyx', name: 'Onyx', description: '深沉有力', language: '多语言', provider },
    { id: 'nova', name: 'Nova', description: '温柔甜美', language: '多语言', provider },
    { id: 'shimmer', name: 'Shimmer', description: '明亮活泼', language: '多语言', provider },
  ]
}
