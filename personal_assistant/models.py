# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Personal Assistant Calendar Environment."""

from typing import Optional
from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class CalendarAction(Action):
    """Action for the calendar assistant - an instruction to the assistant."""

    instruction: str = Field(..., description="Natural language instruction for the assistant")


class CalendarObservation(Observation):
    """Observation from the calendar assistant environment.

    Designed to be Markov: contains everything the agent needs to act
    optimally without relying on conversation history.
    """

    output: str = Field(default="", description="Result of the action taken")
    pending_tasks: int = Field(default=0, description="Number of unresolved tasks remaining")
    events_today: int = Field(default=0, description="Number of events scheduled today")
    flags_found: list[str] = Field(default_factory=list, description="Successfully completed tasks")

    # --- Markov state snapshot ---
    state_summary: str = Field(default="", description="Compact text rendering of full environment state — use this for sliding-window agent prompts")
    calendar_snapshot: list[dict] = Field(default_factory=list, description="All current calendar events")
    discovered_preferences: dict[str, dict] = Field(default_factory=dict, description="Contact preferences the agent has queried so far (person -> prefs)")
    discovered_constraints: list[dict] = Field(default_factory=list, description="Private constraints revealed so far")
    constraint_status: dict = Field(default_factory=dict, description="Current hard/soft constraint violation summary")
    active_negotiations: dict[str, dict] = Field(default_factory=dict, description="In-progress negotiation state (scenario_id -> {round, attempts, last_feedback})")
    resolved_negotiations: dict[str, bool] = Field(default_factory=dict, description="Completed negotiations (scenario_id -> success)")
    unhandled_interrupts: list[str] = Field(default_factory=list, description="Interrupt messages not yet acted on")
    notifications_sent: list[dict] = Field(default_factory=list, description="All notifications sent so far")
    step_count: int = Field(default=0, description="Current step number in the episode")


class CalendarState(State):
    """Extended state tracking for calendar environment."""

    tasks_completed: int = 0
    total_tasks: int = 0
