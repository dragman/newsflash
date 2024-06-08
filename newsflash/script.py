import datetime
import json
import logging
import os
from collections import Counter
from typing import Any, Literal, cast

import click
import requests
from dotenv import load_dotenv
from openai import OpenAI

from newsflash.constants import (
    IGNORE_ABSTRACTS,
    STANDARD_PROMPT,
    SYSTEM_PROMPT,
    THRIPLASH_PROMPT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
SupportedModels = Literal["gpt-4-turbo", "gpt-3.5-turbo"]
MAX_TOKENS = 50
MAX_CHARS = 120

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_SOURCES = ["bbc-news", "bbc-sport"]

JSONType = dict[str, str | int | float | bool | None | dict[str, Any] | list[Any]]


# News API URL for top BBC news stories of the last week
def fetch_articles(
    page: int, from_date: datetime.date, to_date: datetime.date
) -> JSONType:
    sources = ",".join(NEWS_API_SOURCES)
    url = (
        f"https://newsapi.org/v2/everything?sources={sources},&from={from_date}&to={to_date}"
        f"&sortBy=popularity&apiKey={NEWS_API_KEY}&pageSize=100&page={page}"
    )

    # Fetch the news data
    logging.info(f"Sending request: {url}")
    response = requests.get(url)
    news_data = cast(JSONType, response.json())
    return news_data


def get_current_date() -> datetime.date:
    current_date = datetime.datetime.now().date()
    return current_date


def save_articles(run_date: datetime.date, lookback: int, max_pages: int = 1) -> str:
    # Current date and one week ago date
    start_date = run_date - datetime.timedelta(days=lookback)
    all_articles: list[JSONType] = []
    current_date = start_date
    while current_date <= run_date:
        logger.info(f"Collecting articles for {current_date}")
        page = 1
        total_results = 1
        while (page - 1) * 100 < total_results and page <= max_pages:
            news_data = fetch_articles(page, current_date, current_date)
            total_results = cast(int, news_data["totalResults"])
            articles = cast(list[JSONType], news_data["articles"])
            all_articles.extend(articles)
            page += 1
        current_date += datetime.timedelta(days=1)

    logger.info(f"Finished requesting articles, got {len(all_articles)}!")
    headlines_and_abstracts = {}
    seen_abstracts: set[str] = set()

    skipped_abstracts: list[str] = []
    for article in all_articles:
        headline = article["title"]
        if not headline:
            logger.warning("Skipping null headline")
            continue

        abstract = cast(str | None, article["description"])

        if abstract is not None:
            if abstract in IGNORE_ABSTRACTS:
                skipped_abstracts.append(abstract)
                continue

            if abstract in seen_abstracts:
                skipped_abstracts.append(abstract)
                continue

            seen_abstracts.add(abstract)
        else:
            # Some providers have null abstracts, so just use headline.
            abstract = ""

        headlines_and_abstracts[headline] = abstract

    logging.info(
        f"Finished processing stories, skipped abstracts: {Counter(skipped_abstracts)}"
    )

    # Save to a file
    file_name = f"bbc_news_{run_date.isoformat()}.json"
    with open(file_name, "w") as file:
        json.dump(headlines_and_abstracts, file, indent=4)

    logger.info(f"BBC News headlines and abstracts saved to {file_name}")
    return file_name


def generate_quiplash_prompt(
    headline: str,
    abstract: str,
    model: SupportedModels,
    is_wet: bool = False,
    is_thriplash: bool = False,
) -> str | None:
    if is_thriplash:
        gpt_prompt = THRIPLASH_PROMPT
    else:
        gpt_prompt = STANDARD_PROMPT

    gpt_prompt = gpt_prompt.format(headline=headline, abstract=abstract)

    if not is_wet:
        logger.info(f"Not sending prompt, dry mode!\n{gpt_prompt}")
        return '"Testing prompts..."'

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": gpt_prompt},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content


def process_articles(
    file_name: str,
    model: SupportedModels,
    is_wet: bool = False,
    thriplash_pct: float = 0.1,
) -> None:
    # Load your dictionary with headlines and abstracts
    with open(file_name, "r") as file:
        headlines_and_abstracts = json.load(file)

    # Figure out how many thriplash prompts we will ame
    thriplash_threshold = (1 - thriplash_pct) * len(headlines_and_abstracts)

    # Generate prompts for each abstract
    quiplash_prompts: list[dict[str, str]] = []
    for i, (headline, abstract) in enumerate(headlines_and_abstracts.items()):
        is_thriplash = i > thriplash_threshold
        logger.info(f"Sending: {headline}")
        quiplash_prompt = generate_quiplash_prompt(
            headline, abstract, model, is_wet=is_wet, is_thriplash=is_thriplash
        )
        if quiplash_prompt is None:
            logger.warning("Nothing returned from GPT!")
            continue

        quiplash_prompts.append(
            {
                "headline": headline,
                "abstract": abstract,
                "quiplash_prompt": quiplash_prompt.strip("\"'"),
                "is_thriplash": is_thriplash,
            }
        )

    # Save the results to a formatted JSON file
    with open(f"{file_name}_quiplash_prompts.json", "w") as file:
        json.dump(quiplash_prompts, file, indent=4)

    in_thriplash_section = False
    with open(f"{file_name}_quiplash_prompts.txt", "w") as file:
        for quiplash_prompt in quiplash_prompts:
            if not in_thriplash_section:
                if quiplash_prompt["is_thriplash"]:
                    file.write("=== THRIPLASHES ===\n\n")
                    in_thriplash_section = True
            prompt = quiplash_prompt["quiplash_prompt"]
            if len(prompt) > MAX_CHARS:
                logging.warning("Skipping long prompt: {prompt}")
                continue

            file.write(prompt + "\n\n")


@click.command()
@click.option("--wet/--dry", is_flag=True, default=False)
@click.option("--input-date", "-d", type=click.DateTime())
@click.option(
    "--model",
    "-m",
    type=click.Choice(["gpt-4-turbo", "gpt-3.5-turbo"]),
    default="gpt-3.5-turbo",
    required=True,
)
@click.option("--lookback", "-l", type=int, default=7)
def main(
    wet: bool,
    input_date: datetime.datetime | None,
    model: SupportedModels,
    lookback: int,
) -> None:
    if input_date is None:
        run_date = get_current_date() - datetime.timedelta(days=1)
    else:
        run_date = input_date.date()

    articles_file = save_articles(run_date, lookback=lookback)
    process_articles(articles_file, model, is_wet=wet)
    logging.info("Job's done!")


if __name__ == "__main__":
    main()
