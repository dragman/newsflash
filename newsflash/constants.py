IGNORE_ABSTRACTS = {
    "The latest five minute news bulletin from BBC World Service.",
    "Continuing coverage of the 2024 General Election Campaign, from BBC News",
}

SYSTEM_PROMPT = (
    "You are a helpful assistant that generates 'Quiplash' prompts from news headline|abstract pairs. You may also be explicitly asked to generate "
    "'Thriplash' prompts. A 'Thriplash' prompt is a special round in the game Quiplash where players have to provide three funny responses to a given prompt. "
    "For example: 'Three things you might find in a wizard's pocket.' Please do not enclose prompts in quotation marks or try to answer prompts yourself, "
    "please just generate prompts under 120 characters and try to leave room for creativity from the players.  When shortneing names from headlines prefer surnames."
)

STANDARD_PROMPT = "Turn this news headline|abstract into standard Quiplash prompt: {headline}|{abstract}"

THRIPLASH_PROMPT = (
    "Turn this news headline|abstract into a 'Thriplash' prompt: {headline}|{abstract}"
)
