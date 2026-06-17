import assert from 'node:assert/strict'
import test from 'node:test'
import { LOCAL_CODEX_JSON_SCHEMAS, buildPrompt, validateLocalCodexPayload } from '../src/services/local-codex-agent.js'

function visitObjectSchemas(schema: any, visitor: (schema: any, path: string) => void, path = '$') {
  if (!schema || typeof schema !== 'object') return

  if (schema.type === 'object' && schema.properties) {
    visitor(schema, path)
    for (const [key, child] of Object.entries(schema.properties)) {
      visitObjectSchemas(child, visitor, `${path}.${key}`)
    }
  }

  if (schema.type === 'array') {
    visitObjectSchemas(schema.items, visitor, `${path}[]`)
  }

  for (const child of schema.anyOf || []) {
    visitObjectSchemas(child, visitor, `${path}.anyOf`)
  }
}

test('local Codex structured output schemas require every declared property', () => {
  for (const [agentType, schema] of Object.entries(LOCAL_CODEX_JSON_SCHEMAS)) {
    visitObjectSchemas(schema, (objectSchema, path) => {
      const properties = Object.keys(objectSchema.properties)
      const required = objectSchema.required || []
      assert.deepEqual(
        [...required].sort(),
        [...properties].sort(),
        `${agentType} ${path} required must include every property for Codex structured output`,
      )
    })
  }
})

test('local Codex storyboard validation rejects missing required storyboard fields', () => {
  assert.throws(
    () => validateLocalCodexPayload('storyboard_breaker', {
      storyboards: [
        {
          shot_number: 1,
          description: 'A complete description.',
          video_prompt: '0-3s: action.',
          duration: 3,
          scene_id: null,
          character_ids: [],
        },
      ],
    }),
    /title/i,
  )
})

test('local Codex prompt includes active agent prompt and project skill instructions', () => {
  const prompt = buildPrompt('storyboard_breaker', 'BODY', {
    systemPrompt: 'CUSTOM STORYBOARD SYSTEM PROMPT',
    skillInstructions: 'PROJECT STORYBOARD SKILL RULES',
  })

  assert.match(prompt, /CUSTOM STORYBOARD SYSTEM PROMPT/)
  assert.match(prompt, /PROJECT STORYBOARD SKILL RULES/)
  assert.match(prompt, /BODY/)
})
