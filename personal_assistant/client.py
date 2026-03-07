# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Personal Assistant Calendar Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import CalendarAction, CalendarObservation, CalendarState


class PersonalAssistantEnv(
    EnvClient[CalendarAction, CalendarObservation, CalendarState]
):
    """Client for the Personal Assistant Calendar Environment."""

    def _step_payload(self, action: CalendarAction) -> Dict:
        return {"instruction": action.instruction}

    def _parse_result(self, payload: Dict) -> StepResult[CalendarObservation]:
        obs_data = payload.get("observation", {})
        observation = CalendarObservation(
            output=obs_data.get("output", ""),
            pending_tasks=obs_data.get("pending_tasks", 0),
            events_today=obs_data.get("events_today", 0),
            flags_found=obs_data.get("flags_found", []),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> CalendarState:
        return CalendarState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            tasks_completed=payload.get("tasks_completed", 0),
            total_tasks=payload.get("total_tasks", 0),
        )
