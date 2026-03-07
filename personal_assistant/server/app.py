# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Personal Assistant Calendar Environment."""

try:
    from openenv.core.env_server.web_interface import create_web_interface_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install with: uv pip install openenv-core"
    ) from e

from models import CalendarAction, CalendarObservation
from .personal_assistant_environment import PersonalAssistantEnvironment

app = create_web_interface_app(
    PersonalAssistantEnvironment,
    CalendarAction,
    CalendarObservation,
    env_name="personal_assistant",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
