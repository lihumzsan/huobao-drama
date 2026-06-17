import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildComfyUiVideoViewUrl,
  ComfyUiVideoAdapter,
  inferComfyUiVideoMotionLevel,
  parseComfyUiVideoHistoryResponse,
  resolveComfyUiVideoWorkflow,
  selectComfyUiVideoWorkflow,
} from '../src/services/adapters/comfyui-video.js'

const IMAGE_TO_VIDEO_WORKFLOW = 'basevideo/Seedance2.0_Bernini_01_480p_10s'
const AUDIO_LIPSYNC_WORKFLOW = 'basevideo/Seedance2.0_Bernini_01+02+04_480p_10s_图生视频出音效对口型'

function findNode(graph: Record<string, any>, classType: string) {
  return Object.values(graph).find(node => node.class_type === classType)
}

function findNodes(graph: Record<string, any>, classType: string) {
  return Object.values(graph).filter(node => node.class_type === classType)
}

function containsValue(value: unknown, expected: string): boolean {
  if (value === expected) return true
  if (typeof value === 'string') return value.includes(expected)
  if (Array.isArray(value)) return value.some(item => containsValue(item, expected))
  if (value && typeof value === 'object') {
    return Object.values(value).some(item => containsValue(item, expected))
  }
  return false
}

test('resolves Bernini 480p workflow with image, fps, frame count, and output prefix injected', () => {
  const graph = resolveComfyUiVideoWorkflow({
    workflowKey: IMAGE_TO_VIDEO_WORKFLOW,
    prompt: 'A calm character slowly turns toward the camera.',
    imageFilenames: ['huobao-ref.png'],
    duration: 10,
    fps: 24,
    frameCount: 241,
    aspectRatio: '9:16',
    motionLevel: 1,
  })

  const loadImage = findNode(graph, 'LoadImage')
  assert.ok(loadImage)
  assert.equal(loadImage.inputs.image, 'huobao-ref.png')
  assert.equal(Object.prototype.hasOwnProperty.call(loadImage.inputs, 'upload'), false)

  const frameSource = findNode(graph, 'PrimitiveInt')
  assert.ok(frameSource)
  assert.equal(frameSource.inputs.value, 241)

  const lengthSource = findNode(graph, 'JWInteger')
  assert.ok(lengthSource)
  assert.equal(lengthSource.inputs.value, 848)

  const scale = findNode(graph, 'LayerUtility: ImageScaleByAspectRatio V2')
  assert.ok(scale)
  assert.equal(scale.inputs.aspect_ratio, '9:16')

  const createVideo = findNode(graph, 'CreateVideo')
  assert.ok(createVideo)
  assert.equal(createVideo.inputs.fps, 24)

  const saveVideo = findNode(graph, 'SaveVideo')
  assert.ok(saveVideo)
  assert.match(String(saveVideo.inputs.filename_prefix), /^video\/huobao-bernini-/)
})

test('defaults ComfyUI Bernini video workflow to 16:9 landscape output', () => {
  const graph = resolveComfyUiVideoWorkflow({
    workflowKey: IMAGE_TO_VIDEO_WORKFLOW,
    prompt: 'A quiet consultation room, the doctor reviews papers while the patient sits still.',
    imageFilenames: ['huobao-ref.png'],
    duration: 4,
    fps: 24,
    frameCount: 97,
    motionLevel: 1,
  })

  const scale = findNode(graph, 'LayerUtility: ImageScaleByAspectRatio V2')
  assert.ok(scale)
  assert.equal(scale.inputs.aspect_ratio, '16:9')
  assert.equal(scale.inputs.proportional_width, 16)
  assert.equal(scale.inputs.proportional_height, 9)

  const lengthSource = findNode(graph, 'JWInteger')
  assert.ok(lengthSource)
  assert.equal(lengthSource.inputs.value, 848)
})

test('injects only the 260412 motion LoRA strength from explicit motion level', () => {
  const graph = resolveComfyUiVideoWorkflow({
    workflowKey: IMAGE_TO_VIDEO_WORKFLOW,
    prompt: 'A character sprints and jumps through sparks.',
    imageFilenames: ['huobao-ref.png'],
    duration: 6,
    fps: 24,
    aspectRatio: '9:16',
    motionLevel: 3,
  })

  const loras = findNodes(graph, 'LoraLoaderModelOnly')
  const motionLora = loras.find(node => String(node.inputs.lora_name).includes('260412_rank_64_fp16'))
  const otherLoras = loras.filter(node => node !== motionLora)

  assert.ok(motionLora)
  assert.equal(motionLora.inputs.strength_model, 3)
  for (const lora of otherLoras) {
    assert.ok(Math.abs(Number(lora.inputs.strength_model) - 1) < 0.000001)
  }
})

test('infers motion levels from storyboard text', () => {
  assert.equal(inferComfyUiVideoMotionLevel('平静地坐着，轻轻呼吸，看向窗外。'), 1)
  assert.equal(inferComfyUiVideoMotionLevel('角色高速奔跑、跳跃、激烈打斗并撞破门。'), 3)
  assert.equal(inferComfyUiVideoMotionLevel('The character walks forward and turns around.'), 2)
})

test('adapts RunningHub Codex prompts to duration and motion level', () => {
  const graph = resolveComfyUiVideoWorkflow({
    workflowKey: IMAGE_TO_VIDEO_WORKFLOW,
    prompt: 'The actor calmly raises a cup and smiles.',
    imageFilenames: ['huobao-ref.png'],
    duration: 8,
    fps: 24,
    aspectRatio: '9:16',
    motionLevel: 1,
  })

  assert.equal(containsValue(graph, '8 seconds'), true)
  assert.equal(containsValue(graph, 'motion level: calm'), true)
  assert.equal(containsValue(graph, 'preserve the reference composition'), true)
  assert.equal(containsValue(graph, 'small controlled movement'), true)
  assert.equal(containsValue(graph, 'The actor calmly raises a cup and smiles.'), true)
  assert.equal(containsValue(graph, '5秒内的视频动态'), false)
  assert.equal(containsValue(graph, '10-second video prompt'), false)
  assert.equal(containsValue(graph, 'clear opening, transition, climax, and final frame'), false)
})

test('selects the audio lipsync workflow only when audio is present', () => {
  assert.equal(selectComfyUiVideoWorkflow(IMAGE_TO_VIDEO_WORKFLOW, false), IMAGE_TO_VIDEO_WORKFLOW)
  assert.equal(selectComfyUiVideoWorkflow(IMAGE_TO_VIDEO_WORKFLOW, true), AUDIO_LIPSYNC_WORKFLOW)
})

test('resolves Bernini audio lipsync workflow with voice audio and final VHS output retained', () => {
  const graph = resolveComfyUiVideoWorkflow({
    workflowKey: AUDIO_LIPSYNC_WORKFLOW,
    prompt: 'A doctor speaks calmly to the patient.',
    imageFilenames: ['huobao-ref.png'],
    audioFilename: 'huobao-voice.wav',
    duration: 10,
    fps: 24,
    frameCount: 241,
    aspectRatio: '16:9',
    motionLevel: 1,
    outputPrefix: 'video/huobao-bernini-lipsync-42',
  } as any)

  const loadImage = findNode(graph, 'LoadImage')
  assert.ok(loadImage)
  assert.equal(loadImage.inputs.image, 'huobao-ref.png')

  const loadAudio = findNode(graph, 'LoadAudio')
  assert.ok(loadAudio)
  assert.equal(loadAudio.inputs.audio, 'huobao-voice.wav')
  assert.equal(Object.prototype.hasOwnProperty.call(loadAudio.inputs, 'upload'), false)
  assert.equal(Object.prototype.hasOwnProperty.call(loadAudio.inputs, 'audioUI'), false)

  const finalVideo = Object.values(graph).find(node => {
    return node.class_type === 'VHS_VideoCombine'
      && String(node._meta?.title || '').includes('最终视频')
  })
  assert.ok(finalVideo)
  assert.equal(finalVideo.inputs.save_output, true)
  assert.equal(finalVideo.inputs.filename_prefix, 'video/huobao-bernini-lipsync-42')

  assert.ok(findNode(graph, 'PainterAudioCut'))
  assert.ok(findNode(graph, 'PainterAV2V'))
  assert.ok(findNode(graph, 'SoundFlow_Mixer'))

  const frameSource = findNode(graph, 'PrimitiveInt')
  assert.ok(frameSource)
  assert.equal(frameSource.inputs.value, 241)

  const createVideo = findNode(graph, 'CreateVideo')
  assert.ok(createVideo)
  assert.equal(createVideo.inputs.fps, 24)
})

test('adapts the Foley RunningHub Codex node to sound effect prompting in audio workflow', () => {
  const graph = resolveComfyUiVideoWorkflow({
    workflowKey: AUDIO_LIPSYNC_WORKFLOW,
    prompt: 'A character walks through a quiet hospital corridor and opens a door.',
    imageFilenames: ['huobao-ref.png'],
    audioFilename: 'huobao-voice.wav',
    duration: 6,
    fps: 24,
    frameCount: 145,
    aspectRatio: '16:9',
    motionLevel: 2,
  } as any)

  const foleyNode = Object.values(graph).find(node => {
    return node.class_type === 'RH_CODEX_NODE'
      && String(node._meta?.title || '').includes('Foley')
  })
  assert.ok(foleyNode)
  assert.equal(containsValue(graph, 'sound effects'), true)
  assert.equal(containsValue(foleyNode, 'Foley'), true)
})

test('builds ComfyUI audio lipsync request with uploaded image and audio inputs', async () => {
  const originalFetch = globalThis.fetch
  const adapter = new ComfyUiVideoAdapter()
  let uploadCount = 0

  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = input instanceof Request ? input.url : String(input)

    if (url === 'http://127.0.0.1:8878/upload/image') {
      uploadCount += 1
      assert.equal(init?.method, 'POST')
      return new Response(JSON.stringify({
        name: uploadCount === 1 ? 'huobao-ref.png' : 'huobao-voice.wav',
      }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }

    throw new Error(`Unexpected fetch: ${url}`)
  }) as typeof fetch

  try {
    const request = await adapter.buildGenerateRequest({
      provider: 'comfyui',
      baseUrl: 'http://127.0.0.1:8878',
      apiKey: '',
      model: IMAGE_TO_VIDEO_WORKFLOW,
    }, {
      id: 42,
      model: IMAGE_TO_VIDEO_WORKFLOW,
      prompt: 'A doctor speaks calmly to the patient.',
      referenceMode: 'single',
      imageUrl: 'data:image/png;base64,iVBORw0KGgo=',
      audioUrl: 'data:audio/wav;base64,UklGRg==',
      duration: 10,
      fps: 24,
      frameCount: 241,
      aspectRatio: '16:9',
      motionLevel: 1,
    } as any)

    assert.equal(uploadCount, 2)
    assert.equal(request.url, 'http://127.0.0.1:8878/prompt')
    assert.equal(containsValue(request.body.prompt, 'huobao-ref.png'), true)
    assert.equal(containsValue(request.body.prompt, 'huobao-voice.wav'), true)
    assert.ok(findNode(request.body.prompt, 'LoadAudio'))
    assert.ok(Object.values(request.body.prompt).some((node: any) => {
      return node.class_type === 'VHS_VideoCombine'
        && String(node._meta?.title || '').includes('最终视频')
    }))
  } finally {
    globalThis.fetch = originalFetch
  }
})

test('parses ComfyUI video history responses', () => {
  assert.deepEqual(parseComfyUiVideoHistoryResponse({}, 'http://127.0.0.1:8188'), {
    status: 'processing',
  })

  assert.deepEqual(
    parseComfyUiVideoHistoryResponse({
      promptA: {
        status: { status_str: 'success', completed: true },
        outputs: {
          '375': {
            videos: [
              { filename: 'clip.mp4', subfolder: 'video', type: 'output' },
            ],
          },
        },
      },
    }, 'http://127.0.0.1:8188'),
    {
      status: 'completed',
      videoUrl: 'http://127.0.0.1:8188/view?filename=clip.mp4&subfolder=video&type=output',
    },
  )
})

test('parses ComfyUI SaveVideo image outputs that contain mp4 files', () => {
  assert.deepEqual(
    parseComfyUiVideoHistoryResponse({
      promptA: {
        status: { status_str: 'success', completed: true },
        outputs: {
          '375': {
            images: [
              { filename: 'huobao-bernini-4_00001_.mp4', subfolder: 'video', type: 'output' },
            ],
            animated: [true],
          },
        },
      },
    }, 'http://127.0.0.1:8878'),
    {
      status: 'completed',
      videoUrl: 'http://127.0.0.1:8878/view?filename=huobao-bernini-4_00001_.mp4&subfolder=video&type=output',
    },
  )
})

test('parses ComfyUI VHS filename outputs that contain mp4 files', () => {
  assert.deepEqual(
    parseComfyUiVideoHistoryResponse({
      promptA: {
        status: { status_str: 'success', completed: true },
        outputs: {
          '1503': {
            filenames: [
              { filename: 'Bernini-T10-lipsync_00001.mp4', subfolder: 'video', type: 'output' },
            ],
          },
        },
      },
    }, 'http://127.0.0.1:8878'),
    {
      status: 'completed',
      videoUrl: 'http://127.0.0.1:8878/view?filename=Bernini-T10-lipsync_00001.mp4&subfolder=video&type=output',
    },
  )
})

test('builds ComfyUI video view urls', () => {
  assert.equal(
    buildComfyUiVideoViewUrl('http://127.0.0.1:8188/', {
      filename: 'clip.mp4',
      subfolder: 'video',
      type: 'output',
    }),
    'http://127.0.0.1:8188/view?filename=clip.mp4&subfolder=video&type=output',
  )
})
