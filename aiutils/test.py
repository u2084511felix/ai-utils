from openai_config import cclient, Generate 
import asyncio

cclient.set_vendor("google")
test = Generate()

async def test_ing():

    #res = await test.web_search("what are the latest tarrifs china has put on the US?")

    print(cclient.vendor)
    system = "You are a helpful assistant."
    res = await test.generate(system, "write a short haiku about the rule of law over power")
    print(res)


asyncio.run(test_ing())
