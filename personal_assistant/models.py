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
    """Observation from the calendar assistant environment."""

    output: str = Field(default="", description="Result of the action taken")
    pending_tasks: int = Field(default=0, description="Number of unresolved tasks remaining")
    events_today: int = Field(default=0, description="Number of events scheduled today")
    flags_found: list[str] = Field(default_factory=list, description="Successfully completed tasks")


class CalendarState(State):
    """Extended state tracking for calendar environment."""

    tasks_completed: int = 0
    total_tasks: int = 0
