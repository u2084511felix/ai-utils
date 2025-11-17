import asyncio

from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

openai = AsyncOpenAI()

input = """Ah, noble traveler! Heed my words, and I shall lead thee to the fabled Holeful Bakery!\n\nStep forth upon West 74th Street, marching straight with purpose. When thou dost reach the great crossing at Columbus Avenue, turn left, as if answering the call to adventure!\n\nContinue southward, past bustling merchants and townfolk, until thou dost arrive at Amsterdam Avenue. Here, turn right, for the scent of warm-baked glory draws near!\n\nLo! Just ahead, the crest of Levain Bakery stands proud. Enter, noble traveler, and claim thy rightful reward—a golden, gooey treasure beyond measure!\n\nGo forth, and may thy quest be delicious and true!"""

instructions = """Affect: Deep, commanding, and slightly dramatic, with an archaic and reverent quality that reflects the grandeur of Olde English storytelling.\n\nTone: Noble, heroic, and formal, capturing the essence of medieval knights and epic quests, while reflecting the antiquated charm of Olde English.\n\nEmotion: Excitement, anticipation, and a sense of mystery, combined with the seriousness of fate and duty.\n\nPronunciation: Clear, deliberate, and with a slightly formal cadence. Specific words like \"hast,\" \"thou,\" and \"doth\" should be pronounced slowly and with emphasis to reflect Olde English speech patterns.\n\nPause: Pauses after important Olde English phrases such as \"Lo!\" or \"Hark!\" and between clauses like \"Choose thy path\" to add weight to the decision-making process and allow the listener to reflect on the seriousness of the quest."""

async def main() -> None:

    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="verse",
        input=input,
        instructions=instructions,
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)

if __name__ == "__main__":
    asyncio.run(main())
