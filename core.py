"""Input validation and application behavior independent of the web UI."""

from __future__ import annotations

import secrets

from memory import MemoryStore


def effective_prompt(prompt: str, style: str) -> str:
    prompt = prompt.strip()
    style = style.strip()
    if not prompt or len(prompt) > 800:
        raise ValueError("Enter an image description of up to 800 characters.")
    if len(style) > 400:
        raise ValueError("The preferred style must be at most 400 characters.")
    return f"{prompt}, {style}" if style else prompt


def generate_and_remember(
    store: MemoryStore, model, prompt: str, style: str, steps: int, seed: int | None
) -> dict:
    combined = effective_prompt(prompt, style)
    steps = int(steps)
    if not 1 <= steps <= 4:
        raise ValueError("Choose between 1 and 4 generation steps.")
    seed = -1 if seed is None else int(seed)
    if seed == -1:
        seed = secrets.randbelow(2**32)
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be -1 or between 0 and 4294967295.")

    image = model.generate_image(combined, steps=steps, seed=seed)
    record = store.save_generation(
        image,
        prompt=prompt.strip(),
        effective_prompt=combined,
        style=style.strip(),
        model_id=model.image_model_id,
        steps=steps,
        seed=seed,
    )
    store.set_style(style)
    return record


def conversation_context(store: MemoryStore, user_text: str) -> list[dict]:
    profile = store.profile()
    name = profile["display_name"]
    recent_images = store.generations(limit=3)
    image_context = "; ".join(
        f"{item['prompt']} (seed {item['seed']})" for item in recent_images
    ) or "none yet"
    old_matches = store.relevant_older_messages(user_text)
    older_context = "; ".join(
        f"{item['role']}: {item['content'][:300]}" for item in old_matches
    ) or "none"
    system = (
        "You are a friendly assistant for creating images. Reply in the user's "
        "language, keep answers concise, and use past conversation naturally. "
        f"The local user's chosen name is {name}. "
        f"User notes to remember: {profile['notes'][:1500] or 'none'}. "
        f"Images already generated, newest first: {image_context}. "
        f"Older related conversation: {older_context}. "
        "You can discuss or refine an image idea, but image generation happens "
        "only when the user presses the Generate button."
    )
    recent = store.messages(limit=12)
    # Keep the context bounded even when the saved history grows indefinitely.
    budget = 5000
    selected = []
    for message in reversed(recent):
        content = message["content"][:1000]
        if len(content) > budget:
            break
        selected.append({"role": message["role"], "content": content})
        budget -= len(content)
    selected.reverse()
    return [{"role": "system", "content": system}, *selected,
            {"role": "user", "content": user_text}]
