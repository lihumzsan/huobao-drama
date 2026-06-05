import { spawn } from 'node:child_process'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'

export type CodexReasoningEffort = 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh'

export interface CodexExecCommandOptions {
  schemaPath: string
  outputPath: string
  codexBin?: string
  model?: string
  reasoningEffort?: CodexReasoningEffort
}

export interface CodexRunOptions<T> {
  prompt: string
  schema: Record<string, unknown>
  cwd: string
  codexBin?: string
  model?: string
  reasoningEffort?: CodexReasoningEffort
  timeoutMs?: number
  validate?: (value: unknown) => T
}

export interface CodexProcessResult {
  code: number | null
  signal?: NodeJS.Signals | null
  stdout: string
  stderr: string
}

const DEFAULT_CODEX_MODEL = 'gpt-5.5'
const DEFAULT_REASONING_EFFORT: CodexReasoningEffort = 'xhigh'
const DEFAULT_TIMEOUT_MS = 15 * 60 * 1000

export function buildCodexExecCommand(options: CodexExecCommandOptions) {
  const model = options.model || DEFAULT_CODEX_MODEL
  const reasoningEffort = options.reasoningEffort || DEFAULT_REASONING_EFFORT
  return {
    bin: options.codexBin || process.env.CODEX_BIN || 'codex',
    args: [
      'exec',
      '--ephemeral',
      '--sandbox',
      'read-only',
      '--config',
      'approval_policy="never"',
      '--model',
      model,
      '--config',
      `model_reasoning_effort="${reasoningEffort}"`,
      '--output-schema',
      options.schemaPath,
      '--output-last-message',
      options.outputPath,
      '--color',
      'never',
      '-',
    ],
  }
}

export function parseCodexJsonOutput(raw: string) {
  let text = (raw || '').trim()
  if (text.startsWith('```')) {
    text = text
      .replace(/^```(?:json)?\s*/i, '')
      .replace(/\s*```$/i, '')
      .trim()
  }
  return JSON.parse(text)
}

export function createCodexCliErrorMessage(result: CodexProcessResult) {
  const detail = (result.stderr || result.stdout || '').trim().slice(0, 1200)
  const status = result.signal ? `signal ${result.signal}` : `exit ${result.code ?? 'unknown'}`
  const loginHint = /login|logged|auth|unauthorized|token/i.test(detail)
    ? '请先在本机终端运行 codex login，确认当前系统用户已登录 Codex。'
    : '请确认本机已安装 Codex CLI，且 codex 在 PATH 中可用；也可以用 CODEX_BIN 指定可执行文件路径。'
  return `Codex CLI 执行失败（${status}）。${loginHint}${detail ? `\n${detail}` : ''}`
}

export async function runCodexCliJson<T = unknown>(options: CodexRunOptions<T>): Promise<T> {
  const tempDir = await mkdtemp(path.join(tmpdir(), 'huobao-codex-'))
  const schemaPath = path.join(tempDir, 'schema.json')
  const outputPath = path.join(tempDir, 'result.json')

  try {
    await writeFile(schemaPath, JSON.stringify(options.schema), 'utf8')
    const command = buildCodexExecCommand({
      schemaPath,
      outputPath,
      codexBin: options.codexBin,
      model: options.model,
      reasoningEffort: options.reasoningEffort,
    })

    const result = await runProcess(command.bin, command.args, options.prompt, options.cwd, options.timeoutMs || DEFAULT_TIMEOUT_MS)
    if (result.code !== 0) {
      throw new Error(createCodexCliErrorMessage(result))
    }

    const raw = await readOutputOrStdout(outputPath, result.stdout)
    const parsed = parseCodexJsonOutput(raw)
    return options.validate ? options.validate(parsed) : parsed as T
  } finally {
    await rm(tempDir, { recursive: true, force: true })
  }
}

async function readOutputOrStdout(outputPath: string, stdout: string) {
  try {
    const fromFile = await readFile(outputPath, 'utf8')
    if (fromFile.trim()) return fromFile
  } catch {
    // Codex may still print the final answer to stdout if -o cannot be used.
  }
  return stdout
}

function runProcess(bin: string, args: string[], input: string, cwd: string, timeoutMs: number): Promise<CodexProcessResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, {
      cwd,
      shell: process.platform === 'win32',
      stdio: ['pipe', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    const timer = setTimeout(() => {
      child.kill('SIGTERM')
      reject(new Error(`Codex CLI 执行超时（${Math.round(timeoutMs / 1000)} 秒）。`))
    }, timeoutMs)

    child.stdout.on('data', chunk => { stdout += chunk.toString() })
    child.stderr.on('data', chunk => { stderr += chunk.toString() })
    child.on('error', err => {
      clearTimeout(timer)
      reject(new Error(`Codex CLI 启动失败：${err.message}`))
    })
    child.on('close', (code, signal) => {
      clearTimeout(timer)
      resolve({ code, signal, stdout, stderr })
    })
    child.stdin.end(input)
  })
}
