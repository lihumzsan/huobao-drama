import { Hono } from 'hono'
import path from 'path'
import { success, badRequest } from '../utils/response.js'
import { saveUploadedFile } from '../utils/storage.js'

const app = new Hono()
const AUDIO_MAX_BYTES = 30 * 1024 * 1024
const AUDIO_EXTENSIONS = new Set(['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.opus', '.webm'])

// POST /upload/image
app.post('/image', async (c) => {
  const body = await c.req.parseBody()
  const file = body['file']

  if (!file || !(file instanceof File)) {
    return badRequest(c, 'file is required')
  }

  const buffer = await file.arrayBuffer()
  const path = await saveUploadedFile(buffer, 'uploads', file.name)
  return success(c, { url: `/${path}`, path })
})

// POST /upload/audio
app.post('/audio', async (c) => {
  const body = await c.req.parseBody()
  const file = body['file']

  if (!file || !(file instanceof File)) {
    return badRequest(c, 'file is required')
  }
  if (!isSupportedAudioFile(file)) {
    return badRequest(c, 'unsupported audio type')
  }
  if (file.size > AUDIO_MAX_BYTES) {
    return badRequest(c, 'audio file is too large')
  }

  const buffer = await file.arrayBuffer()
  const savedPath = await saveUploadedFile(buffer, 'voice-uploads', file.name)
  return success(c, { url: `/${savedPath}`, path: savedPath })
})

function isSupportedAudioFile(file: File): boolean {
  const ext = path.extname(file.name || '').toLowerCase()
  const mime = String(file.type || '').toLowerCase()
  return AUDIO_EXTENSIONS.has(ext) && (mime.startsWith('audio/') || mime === 'application/octet-stream' || !mime)
}

export default app
