from __future__ import annotations
import os
from aiutils import TextModels
from aiutils.openai_config import Generate
from jinja2 import Template
from typing import get_args
from pathlib import Path


class PromptEngine():
    def __init__(self):
        self.prompt = ""
        self.filepath = ""
        self.kwargs = {}

    def set_prompt(self):
        self.prompt = Path(self.filepath).read_text(encoding="utf-8").format(**self.kwargs)

    def set_template_prompt(self):
        self.prompt = Template(Path(self.filepath).read_text(encoding="utf-8")).render(kwargs=self.kwargs)

    def save_prompt(self, path):
        Path(path).write_text(self.prompt)


class Refactor():
    def __init__(self):
        self.prompt = ""
        self.kwargs = {}
        self.prompt_engine = PromptEngine()
        self.context_file_paths = []
        self.target_file_paths = []
        self.verbose = False


    def get_script(self, path):
        with open(path, "r") as f:
            return f.read()

    def undo_diffs():
        generator = Generate()
        generator.undo_last_diffs()


    async def refactor_script(self):
        generator = Generate()
        self.prompt_engine.set_prompt()

        print(f"Generating diff ...")
        await generator.apply_diff(self.prompt_engine.prompt, self.target_file_paths, verbose=self.verbose)
        print(f"Successfully generated ...")
