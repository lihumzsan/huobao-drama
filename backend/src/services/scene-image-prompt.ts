type SceneImagePromptInput = {
  location?: string | null
  time?: string | null
  prompt?: string | null
}

const PEOPLE_OR_ACTION_PATTERN = /\b(?:doctor|patient|nurse|man|woman|boy|girl|person|people|character|actor|actress|portrait|face|body|standing|sitting|walking|walks|talking|conversation|dialogue|hands|holding|looks|gazes)\b|(?:医生|患者|病人|护士|男人|女人|男子|女子|男孩|女孩|人物|角色|演员|肖像|脸|站|坐|走|递|交谈|对话|说话|凝视|看着|拿着)/i

function cleanText(value: string | null | undefined) {
  return String(value || '').replace(/\s+/g, ' ').trim()
}

function unique(parts: string[]) {
  return parts.filter((value, index, array) => array.indexOf(value) === index)
}

function sanitizeBackgroundNotes(value: string | null | undefined) {
  const parts = cleanText(value)
    .split(/[\n.;。；!?！？]+/)
    .map(cleanText)
    .filter(Boolean)
    .filter(part => !PEOPLE_OR_ACTION_PATTERN.test(part))

  return unique(parts).join('; ')
}

export function buildSceneImagePrompt(scene: SceneImagePromptInput) {
  const location = cleanText(scene.location) || 'unspecified location'
  const time = cleanText(scene.time)
  const backgroundNotes = sanitizeBackgroundNotes(scene.prompt)

  return [
    `Location: ${location}.`,
    time ? `Time and lighting: ${time}.` : 'Time and lighting: cinematic lighting consistent with the episode.',
    backgroundNotes
      ? `Environment notes: ${backgroundNotes}.`
      : 'Environment notes: empty environment, architecture, furniture, props, light, weather, and atmosphere only.',
    'Generate a cinematic pure background scene plate for film and short drama production.',
    'The image must show the location itself as an empty environment: architecture, furniture, props, lighting, weather, atmosphere, depth, and production design.',
    'If source material mentions characters, people, body movement, dialogue, or plot action, omit them and keep only the environment.',
    'Negative constraints: no people, no characters, no actors, no portraits, no faces, no bodies, no hands, no crowds, no dialogue, no text, no captions, no labels, no watermark, no logo.',
    '16:9 horizontal cinematic frame, high quality, realistic lighting, consistent art style.',
  ].join('\n')
}

export function buildSceneStoragePrompt(scene: SceneImagePromptInput) {
  return buildSceneImagePrompt(scene)
}
