from newsflash.script import generate_quiplash_prompt
from unittest.mock import Mock, sentinel
from openai import OpenAI


def test_generate_quiplash_prompt_dry() -> None:
    client = Mock(spec=OpenAI)
    generate_quiplash_prompt(
        client,
        headline=sentinel.HEADLINE,
        abstract=sentinel.ABSTRACT,
        model=sentinel.MODEL,
        is_wet=False,
    )
