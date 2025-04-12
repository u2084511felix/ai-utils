from aiutils.openai_config import Generate
import asyncio


test = Generate()


async def test_ing():

    res = await test.web_search("what are the latest tarrifs china has put on the US?")
    print(res)


asyncio.run(test_ing())
