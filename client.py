# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sre Decision Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import SreDecisionAction, SreDecisionObservation
except ImportError:
    from models import SreDecisionAction, SreDecisionObservation  # type: ignore


class SreDecisionEnv(
    EnvClient[SreDecisionAction, SreDecisionObservation, State]
):
    """
    Client for the Sre Decision Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with SreDecisionEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.metadata)
        ...
        ...     result = client.step(SreDecisionAction(action_name="inspect_logs", rationale="Initial triage"))
        ...     print(result.observation.logs)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = SreDecisionEnv.from_docker_image("sre_decision_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(SreDecisionAction(action_name="inspect_metrics", rationale="Check CPU"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SreDecisionAction) -> Dict:
        """
        Convert SreDecisionAction to JSON payload for step message.
        """
        import logging
        try:
            from .models import VALID_ACTIONS
        except ImportError:
            from models import VALID_ACTIONS  # type: ignore

        # Validation Layer
        if action.action_name not in VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action.action_name}'. Must be one of: {VALID_ACTIONS}")
            
        payload = {
            "action_name": action.action_name,
            "rationale": action.rationale
        }
        
        logging.debug(f"Outgoing action payload: {payload}")
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[SreDecisionObservation]:
        """
        Parse server response into StepResult[SreDecisionObservation].
        """
        import logging
        logging.debug(f"Incoming observation payload: {payload}")
        
        obs_data = payload.get("observation", {})
        observation = SreDecisionObservation(
            logs=obs_data.get("logs"),
            metrics=obs_data.get("metrics"),
            messages=obs_data.get("messages"),
            observer=obs_data.get("observer"),
            time_step=obs_data.get("time_step", 0),
        )

        result = StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
        # Store metadata dynamically since OpenEnv StepResult omits it
        setattr(result, 'info', payload.get("info", {}))
        return result

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
