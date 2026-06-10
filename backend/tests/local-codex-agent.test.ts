import assert from 'node:assert/strict'
import test from 'node:test'
import { LOCAL_CODEX_JSON_SCHEMAS } from '../src/services/local-codex-agent.js'

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
