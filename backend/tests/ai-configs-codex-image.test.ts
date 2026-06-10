import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

async function readJson(resp: Response): Promise<any> {
  const text = await resp.text()
  try {
    return JSON.parse(text)
  } catch {
    return { message: text }
  }
}

test('huobao preset makes Codex the active image config and migrates existing episode locks', async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'huobao-ai-configs-'))
  const originalDbPath = process.env.DB_PATH
  process.env.DB_PATH = path.join(tempRoot, 'huobao.db')

  try {
    const { db, schema } = await import('../src/db/index.js')
    const { default: aiConfigRoutes } = await import('../src/routes/aiConfigs.js')
    const ts = new Date().toISOString()

    const oldImageConfig = db.insert(schema.aiServiceConfigs).values({
      serviceType: 'image',
      provider: 'comfyui',
      name: 'Old ComfyUI image config',
      baseUrl: 'http://127.0.0.1:8188',
      apiKey: '',
      model: JSON.stringify([]),
      priority: 500,
      isActive: true,
      createdAt: ts,
      updatedAt: ts,
    }).run()
    const oldImageConfigId = Number(oldImageConfig.lastInsertRowid)

    const drama = db.insert(schema.dramas).values({
      title: 'Preset Migration Drama',
      style: 'realistic',
      status: 'draft',
      createdAt: ts,
      updatedAt: ts,
    }).run()
    const dramaId = Number(drama.lastInsertRowid)
    const episode = db.insert(schema.episodes).values({
      dramaId,
      episodeNumber: 1,
      title: 'Episode 1',
      imageConfigId: oldImageConfigId,
      createdAt: ts,
      updatedAt: ts,
    }).run()
    const episodeId = Number(episode.lastInsertRowid)

    const resp = await aiConfigRoutes.request('/huobao-preset', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ api_key: 'video-audio-key' }),
    })
    const json = await readJson(resp)

    assert.equal(resp.status, 200)
    assert.equal(json.code, 200)

    const codexImageConfig = db.select().from(schema.aiServiceConfigs).all()
      .find(row => row.serviceType === 'image' && row.provider === 'codex')
    assert.ok(codexImageConfig)
    assert.equal(codexImageConfig.isActive, true)

    const oldImageConfigAfter = db.select().from(schema.aiServiceConfigs).all()
      .find(row => row.id === oldImageConfigId)
    assert.equal(oldImageConfigAfter?.isActive, false)

    const episodeAfter = db.select().from(schema.episodes).all()
      .find(row => row.id === episodeId)
    assert.equal(episodeAfter?.imageConfigId, codexImageConfig.id)

    const imageListResp = await aiConfigRoutes.request('/?service_type=image')
    const imageListJson = await readJson(imageListResp)
    assert.equal(imageListJson.data[0].provider, 'codex')
  } finally {
    if (originalDbPath === undefined) delete process.env.DB_PATH
    else process.env.DB_PATH = originalDbPath
    try {
      await rm(tempRoot, { recursive: true, force: true })
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EBUSY') throw error
    }
  }
})
