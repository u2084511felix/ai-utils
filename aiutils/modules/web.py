

from dataclasses import dataclass, field


from pydantic import BaseModel


from ..clients import make_client, Client
from ..exceptions import ClientError
from .gpt_module import GPTModule
from ..models import TextModels



def _clean_pydantic_model(model):
    model_dict = model.dict()
    clean_dict = {k: v for k, v in model_dict.items() if v is not None}
    return clean_dict


class UserLocation(BaseModel):
    type: str = "approximate"
    country: str
    city: str
    region: str
    timezone: str | None = None


"""
NOTE: 
    search_context_size: "low", "medium", "high"
    "new something"
"""


class SearchTools(BaseModel):
    type: str = "web_search_preview"
    search_context_size: str | None = None
    user_location: UserLocation | None = None


class WebSearchResponsesModel(BaseModel):
    model: str = TextModels.gpt_4_1_nano
    tools: list[SearchTools] = field(default_factory=list)
    input: str | None = None


@dataclass
class WebSearch(GPTModule):
    client: Client = make_client()

    async def web_search(self, tool_dict=None, **kwargs) -> str:
        """
        kwargs:
            model: str = f"{TextModels.hipster_mini}"
            tools: List = 
                type: "web_search_preview"
                search_context_size: str | None = None
                user_location: UserLocation | None = None

            input: str = None

        NOTE: search_context_size: "low", "medium", "high"
        """
        web_search_model = WebSearchResponsesModel(**kwargs)
        clean_model = _clean_pydantic_model(web_search_model)

        if tool_dict is not None:
            new_tools = SearchTools()
            tool_dump = new_tools.model_dump()
            new_dict = {}
            for k, v in tool_dump.items():
                if v is None:
                    continue
                else:
                    if k in tool_dict.keys():
                        new_dict[k] = tool_dict[k]

            clean_model["tools"] = []
            clean_model["tools"].append(tool_dump)

        response = await self.client.async_responses_create(**clean_model)
        if response.error:
            raise ClientError from response.error
        return response.content

