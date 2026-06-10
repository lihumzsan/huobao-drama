type CharacterImagePromptInput = {
  name?: string | null
  role?: string | null
  appearance?: string | null
  description?: string | null
  personality?: string | null
}

function cleanText(value: string | null | undefined) {
  return String(value || '').replace(/\s+/g, ' ').trim()
}

function joinDetails(parts: Array<string | null | undefined>) {
  return parts
    .map(cleanText)
    .filter(Boolean)
    .filter((value, index, array) => array.indexOf(value) === index)
    .join('; ')
}

export function buildCharacterImagePrompt(character: CharacterImagePromptInput) {
  const name = cleanText(character.name) || 'unnamed character'
  const details = joinDetails([
    character.role,
    character.appearance,
    character.description,
    character.personality,
  ])

  return [
    `Character: ${name}.`,
    details ? `Character details: ${details}.` : 'Character details: realistic modern drama character.',
    'Generate exactly one photorealistic character model sheet / character reference board for film and short drama production.',
    'Composition: 16:9 horizontal canvas, white studio background, clean even lighting, four distinct visual regions, no text labels, no panel borders.',
    'Layout: left panel / left 35% of the canvas is one large front-facing close-up portrait; right panel / right 65% of the canvas contains three evenly spaced full-body turnaround views.',
    'Region 1: front-facing close-up portrait, head and shoulders, face clearly visible, neutral expression, looking at camera.',
    'Region 2: full-body front view of the same person, neutral standing pose, arms visible, feet visible.',
    'Region 3: full-body side view of the same person, exact same outfit and hairstyle, feet visible.',
    'Region 4: full-body back view of the same person, exact same outfit and hairstyle, feet visible.',
    'Consistency: same person, same outfit, same hairstyle, same age, same facial features, same body proportions across all views.',
    'Pose: neutral standing A-pose or relaxed straight pose in the turnaround views, full body not cropped.',
    'Negative constraints: must not be a single portrait, no cropped full-body views, no text, no captions, no labels, no watermark, no logo, no extra characters, no unrelated collage, no dramatic scenery, no props.',
    'High-end studio reference photography, accurate clothing details, production-ready character reference sheet.',
  ].join('\n')
}
