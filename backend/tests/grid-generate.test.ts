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

test('grid image generation uses the selected storyboard episode image config', async () => {
  const tempRoot = await mkdtemp(path.join(os.tmpdir(), 'huobao-grid-generate-'))
  const originalDbPath = process.env.DB_PATH
  process.env.DB_PATH = path.join(tempRoot, 'huobao.db')

  try {
    const { db, schema } = await import('../src/db/index.js')
    const { default: gridRoutes } = await import('../src/routes/grid.js')
    const ts = new Date().toISOString()

    db.insert(schema.aiServiceConfigs).values({
      serviceType: 'image',
      provider: 'active-provider',
      name: 'Active global image config',
      baseUrl: 'local://active',
      apiKey: '',
      model: JSON.stringify(['active-model']),
      priority: 100,
      isActive: true,
      createdAt: ts,
      updatedAt: ts,
    }).run()
    const episodeConfig = db.insert(schema.aiServiceConfigs).values({
      serviceType: 'image',
      provider: 'episode-provider',
      name: 'Episode locked image config',
      baseUrl: 'local://episode',
      apiKey: '',
      model: JSON.stringify(['episode-model']),
      priority: 1,
      isActive: true,
      createdAt: ts,
      updatedAt: ts,
    }).run()
    const episodeConfigId = Number(episodeConfig.lastInsertRowid)

    const drama = db.insert(schema.dramas).values({
      title: 'Grid Config Drama',
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
      imageConfigId: episodeConfigId,
      createdAt: ts,
      updatedAt: ts,
    }).run()
    const episodeId = Number(episode.lastInsertRowid)
    const storyboard = db.insert(schema.storyboards).values({
      episodeId,
      storyboardNumber: 1,
      title: 'Shot 1',
      description: 'test shot',
      createdAt: ts,
      updatedAt: ts,
    }).run()
    const storyboardId = Number(storyboard.lastInsertRowid)

    const resp = await gridRoutes.request('/generate', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        storyboard_ids: [storyboardId],
        drama_id: dramaId,
        rows: 1,
        cols: 1,
        mode: 'first_frame',
        custom_prompt: 'single grid frame',
      }),
    })
    const json = await readJson(resp)

    assert.equal(resp.status, 200)
    const [record] = db.select().from(schema.imageGenerations).all()
      .filter(row => row.id === json.data.image_generation_id)
    assert.equal(record.provider, 'episode-provider')
    assert.equal(record.model, 'episode-model')
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
