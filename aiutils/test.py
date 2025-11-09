from openai_config import cclient, Generate 
import asyncio
from aiutils import TextModels
from pydantic import BaseModel
test = Generate()

cclient.set_vendor("google")
from modules import structured_outputs_generator

async def test_ing():

    #res = await test.web_search("what are the latest tarrifs china has put on the US?")

    print(cclient.vendor)
    system = "You are a helpful assistant."

    class Haiki(BaseModel):
        line1: str
        line2: str
        line3: str
        line4: str
        line5: str

    res = await structured_outputs_generator("write a short haiku about the rule of law over power", Haiki, input_type="pydantic", model=TextModels.gemini_25_flash_lite, vendor="google")

    for i in res.keys():
        print(res[i])
    #res = await test.generate(system, "write a short haiku about the rule of law over power")


asyncio.run(test_ing())
