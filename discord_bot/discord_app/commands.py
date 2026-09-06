"""Slash commands — thin wrappers over RoomService (see service.py).

Registration is a plain function rather than a Cog so the bot stays a
discord.Client (no command-prefix machinery it doesn't use).
"""
from __future__ import annotations

import logging

import discord
from discord import app_commands

from .channels import ROOMS_CHANNEL, sync_guild
from .service import MAX_ROUNDS, TURN_POLICIES, ActorMessage, RoomService, ServiceError

log = logging.getLogger(__name__)

AUTOCOMPLETE_LIMIT = 25  # Discord's hard cap on choices


def register(bot) -> None:
    """Attach every command to `bot.tree`, restricted to the configured
    guild/user ids."""
    tree = bot.tree

    def allowed(interaction: discord.Interaction) -> bool:
        settings = bot.settings
        if settings.allowed_guild_id and str(interaction.guild_id) != settings.allowed_guild_id:
            return False
        if settings.allowed_user_ids and str(interaction.user.id) not in settings.allowed_user_ids:
            return False
        return True

    async def guard(interaction: discord.Interaction) -> bool:
        if allowed(interaction):
            return True
        await interaction.response.send_message(
            "This bot is restricted to its configured guild and users.", ephemeral=True
        )
        return False

    # -- autocomplete -------------------------------------------------------

    async def actor_autocomplete(interaction: discord.Interaction, current: str):
        current = (current or "").lower()
        return [
            app_commands.Choice(name=actor_id, value=actor_id)
            for actor_id in bot.service.actor_ids()
            if current in actor_id.lower()
        ][:AUTOCOMPLETE_LIMIT]

    async def participant_autocomplete(interaction: discord.Interaction, current: str):
        room = await bot.store.get_room_by_channel(str(interaction.channel_id))
        if room is None:
            return []
        current = (current or "").lower()
        return [
            app_commands.Choice(name=pid, value=pid)
            for pid in await bot.service.participant_ids(room.id)
            if current in pid.lower()
        ][:AUTOCOMPLETE_LIMIT]

    async def policy_autocomplete(interaction: discord.Interaction, current: str):
        current = (current or "").lower()
        return [
            app_commands.Choice(name=p, value=p) for p in TURN_POLICIES if current in p
        ]

    # -- /models ------------------------------------------------------------

    @tree.command(name="models", description="List the configured actors")
    async def models(interaction: discord.Interaction):
        if not await guard(interaction):
            return
        await interaction.response.send_message(bot.service.describe_actors())

    # -- /sync-models --------------------------------------------------------

    @tree.command(name="sync-models", description="Create/rename/archive actor channels to match config")
    async def sync_models(interaction: discord.Interaction):
        if not await guard(interaction):
            return
        await interaction.response.defer(thinking=True)
        report = await sync_guild(interaction.guild, bot.registry, bot.store)
        parts = []
        for label, ids in (
            ("created", report.created), ("renamed", report.renamed),
            ("archived", report.archived), ("un-archived", report.unarchived),
        ):
            if ids:
                parts.append(f"**{label}**: " + ", ".join(f"`{i}`" for i in ids))
        await interaction.followup.send(
            "\n".join(parts) if parts else "Already in sync — nothing to change."
        )

    # -- /chat ---------------------------------------------------------------

    @tree.command(name="chat", description="Jump to an actor's own channel")
    @app_commands.describe(actor="which actor to talk to")
    @app_commands.autocomplete(actor=actor_autocomplete)
    async def chat(interaction: discord.Interaction, actor: str):
        if not await guard(interaction):
            return
        binding = await bot.store.get_voice_channel(actor)
        if binding is None:
            await interaction.response.send_message(
                f"`{actor}` has no channel yet — run `/sync-models` first.", ephemeral=True
            )
            return
        await interaction.response.send_message(
            f"Talk to `{actor}` in <#{binding.channel_id}> — just send a message there."
        )

    # -- /room ---------------------------------------------------------------

    room_group = app_commands.Group(name="room", description="Create and manage multi-actor rooms")

    @room_group.command(name="create", description="Create a room in this channel's thread")
    @app_commands.describe(title="what to call the room", policy="how turns are chosen")
    @app_commands.autocomplete(policy=policy_autocomplete)
    async def room_create(interaction: discord.Interaction, title: str, policy: str = "round_robin"):
        if not await guard(interaction):
            return
        await interaction.response.defer(thinking=True)
        try:
            thread = await _open_room_thread(interaction, title)
            room = await bot.service.create_room(
                title, policy,
                guild_id=str(interaction.guild_id), channel_id=str(thread.id),
            )
        except ServiceError as exc:
            await interaction.followup.send(f"⚠️ {exc}", ephemeral=True)
            return
        await interaction.followup.send(
            f"Created **{title}** (`{room.id}`, policy **{policy}**) in <#{thread.id}>.\n"
            f"Add actors there with `/room add`, then `/next`."
        )

    @room_group.command(name="add", description="Add an actor to this room (repeatable for duplicates)")
    @app_commands.describe(actor="which actor to add")
    @app_commands.autocomplete(actor=actor_autocomplete)
    async def room_add(interaction: discord.Interaction, actor: str):
        if not await guard(interaction):
            return
        room = await _require_room(interaction, bot)
        if room is None:
            return
        try:
            participant = await bot.service.add_actor(room.id, actor)
        except ServiceError as exc:
            await interaction.response.send_message(f"⚠️ {exc}", ephemeral=True)
            return
        await interaction.response.send_message(f"Added `{participant.id}` to **{room.title}**.")

    @room_group.command(name="remove", description="Remove a participant from this room")
    @app_commands.describe(participant="which participant instance to remove")
    @app_commands.autocomplete(participant=participant_autocomplete)
    async def room_remove(interaction: discord.Interaction, participant: str):
        if not await guard(interaction):
            return
        room = await _require_room(interaction, bot)
        if room is None:
            return
        try:
            await bot.service.remove_participant(room.id, participant)
        except ServiceError as exc:
            await interaction.response.send_message(f"⚠️ {exc}", ephemeral=True)
            return
        await interaction.response.send_message(f"Removed `{participant}` from **{room.title}**.")

    @room_group.command(name="status", description="Show this room's participants and turn state")
    async def room_status(interaction: discord.Interaction):
        if not await guard(interaction):
            return
        room = await _require_room(interaction, bot)
        if room is None:
            return
        await interaction.response.send_message(await bot.service.describe_room(room.id))

    tree.add_command(room_group)

    # -- /next ----------------------------------------------------------------

    @tree.command(name="next", description="Run the next turn in this room")
    @app_commands.describe(participant="who speaks (required for the manual policy)")
    @app_commands.autocomplete(participant=participant_autocomplete)
    async def next_turn(interaction: discord.Interaction, participant: str | None = None):
        if not await guard(interaction):
            return
        room = await _require_room(interaction, bot)
        if room is None:
            return
        await interaction.response.defer(thinking=True)
        try:
            message = await bot.service.take_turn(room.id, participant)
        except ServiceError as exc:
            await interaction.followup.send(f"⚠️ {exc}", ephemeral=True)
            return
        await interaction.followup.send(f"— `{message.display_name}` speaks —")
        await _post(bot, interaction.channel, message)

    # -- /salon and /debate ----------------------------------------------------

    @tree.command(name="salon", description="Convene several actors on a topic")
    @app_commands.describe(
        topic="the question before the room",
        actors="space-separated actor ids; repeat one for two instances of it",
        rounds=f"how many times around the room (1-{MAX_ROUNDS})",
    )
    async def salon(interaction: discord.Interaction, topic: str, actors: str, rounds: int = 2):
        await _run_conversation(bot, interaction, topic, actors, rounds, policy="round_robin", kind="Salon")

    @tree.command(name="debate", description="A debate: fixed positions, fixed order, fixed rounds")
    @app_commands.describe(
        topic="the motion under debate",
        actors="space-separated actor ids, in speaking order",
        rounds=f"how many rounds (1-{MAX_ROUNDS})",
    )
    async def debate(interaction: discord.Interaction, topic: str, actors: str, rounds: int = 2):
        await _run_conversation(bot, interaction, topic, actors, rounds, policy="debate", kind="Debate")

    async def _run_conversation(bot, interaction, topic, actors, rounds, *, policy, kind):
        if not await guard(interaction):
            return
        await interaction.response.defer(thinking=True)
        actor_ids = actors.split()
        title = f"{kind}: {topic}"[:90]
        try:
            thread = await _open_room_thread(interaction, title)
            room = await bot.service.prepare_conversation(
                title, actor_ids, rounds, policy=policy, topic=topic,
                guild_id=str(interaction.guild_id), channel_id=str(thread.id),
            )
        except ServiceError as exc:
            await interaction.followup.send(f"⚠️ {exc}", ephemeral=True)
            return

        await interaction.followup.send(
            f"**{kind}** — _{topic}_\n{len(actor_ids)} participants, {rounds} rounds → <#{thread.id}>"
        )
        await thread.send(f"**{kind}** — _{topic}_\n_participants: "
                          + ", ".join(f"`{a}`" for a in actor_ids) + f" · {rounds} rounds_")

        # Every turn below is scheduled here, one at a time, and the loop is
        # bounded by rounds * participants — posting a message never triggers
        # another (webhook posts are ignored by on_message).
        async for message in bot.service.run_conversation(room.id, rounds * len(actor_ids)):
            async with thread.typing():
                pass
            await _post(bot, thread, message)
        await thread.send(f"_— {kind.lower()} concluded —_")


async def _post(bot, channel, message: ActorMessage) -> None:
    await bot.webhooks.send_as(
        channel, username=message.display_name,
        avatar_url=message.avatar_url, content=message.content,
    )


async def _require_room(interaction: discord.Interaction, bot):
    room = await bot.store.get_room_by_channel(str(interaction.channel_id))
    if room is None:
        await interaction.response.send_message(
            "This channel isn't a room. Use `/room create` (or `/salon`) to start one.",
            ephemeral=True,
        )
    return room


async def _open_room_thread(interaction: discord.Interaction, title: str):
    """Rooms live as threads under #rooms, wherever the command was run."""
    channel = interaction.channel
    if isinstance(channel, discord.Thread):
        return channel
    host = channel
    if channel.name != ROOMS_CHANNEL:
        host = discord.utils.get(interaction.guild.text_channels, name=ROOMS_CHANNEL) or channel
    return await host.create_thread(name=title[:100], type=discord.ChannelType.public_thread)
