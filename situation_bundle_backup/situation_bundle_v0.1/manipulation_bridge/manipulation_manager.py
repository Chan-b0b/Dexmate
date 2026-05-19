#!/usr/bin/env python3
# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Task manager bridge for manipulation, perception (Initialize workflows), and execution.

Flow:
1) Subscribe ``/manipulation/command`` and dispatch by workflow (Initialize / Manipulation / Finish / …).
2) For Initialize flows, after Stand completes, publish ``/perception/pick_check_request`` and handle
   ``/perception/pick_check_result``. Perception always ends with ``/manipulation/state`` **DONE**
   (payload includes ``can_pick``). If ``can_pick`` is false, the next command with the same
   Initialize ``operation_type`` skips Stand and only re-sends the pick check request.
3) Mirror ``/execution/state`` back to ``/manipulation/state`` for the active sequence.

Topics for perception are configurable with environment variables:
  - ``SITUATION_PERCEPTION_REQUEST_TOPIC`` (default: ``/perception/pick_check_request``)
  - ``SITUATION_PERCEPTION_RESULT_TOPIC`` (default: ``/perception/pick_check_result``)
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

LOGGER = logging.getLogger("manipulation_manager")

PERCEPTION_REQUEST_TOPIC = os.environ.get(
    "SITUATION_PERCEPTION_REQUEST_TOPIC", "/perception/pick_check_request"
)
PERCEPTION_RESULT_TOPIC = os.environ.get(
    "SITUATION_PERCEPTION_RESULT_TOPIC", "/perception/pick_check_result"
)

WORKFLOW_STEPS: dict[str, list[str]] = {
    "VegaTask1-040-Finish": ["VegaTask1-041-Place", "VegaTask1-042-Sit"],
    "VegaTask1-020-Manipulation": ["VegaTask1-021-Grab"],
    "VegaTask2-020-Manipulation": ["VegaTask2-021-Grab"],
    "VegaTask2-040-Finish": ["VegaTask2-041-Place", "VegaTask2-042-Sit"],
    "VegaTask3-030-Manipulation": ["VegaTask3-031-Grab"],
    "VegaTask4-020-Finish": ["VegaTask4-021-Place", "VegaTask4-022-Sit"],
}

INITIALIZE_PERCEPTION_WORKFLOWS: dict[str, tuple[str, str]] = {
    "VegaTask1-010-Initialize": ("VegaTask1-011-Stand", "VegaTask1-012-Perception"),
    "VegaTask2-010-Initialize": ("VegaTask2-011-Stand", "VegaTask2-012-Perception"),
    "VegaTask3-020-Initialize": ("VegaTask3-021-Stand", "VegaTask3-022-Perception"),
}

VEGA_TASK_IDS: frozenset[str] = frozenset(
    (
        "VegaTask1-011-Stand",
        "VegaTask1-012-Perception",
        "VegaTask1-021-Grab",
        "VegaTask1-041-Place",
        "VegaTask1-042-Sit",
        "VegaTask2-011-Stand",
        "VegaTask2-012-Perception",
        "VegaTask2-021-Grab",
        "VegaTask2-041-Place",
        "VegaTask2-042-Sit",
        "VegaTask3-021-Stand",
        "VegaTask3-022-Perception",
        "VegaTask3-031-Grab",
        "VegaTask4-021-Place",
        "VegaTask4-022-Sit",
        "G1Task1-1",
        "G1Task2-1",
    )
)

_OPERATION_ALIASES: dict[str, str] = {
    "g1_task_1_1": "G1Task1-1",
    "g1_task_2_1": "G1Task2-1",
}


def resolve_vega_task_id(operation_type: str) -> str | None:
    raw = operation_type.strip()
    if not raw:
        return None
    if raw in VEGA_TASK_IDS:
        return raw
    key = raw.lower()
    if key in _OPERATION_ALIASES:
        return _OPERATION_ALIASES[key]
    compact = key.replace("_", "").replace("-", "")
    if compact == "g1task11":
        return "G1Task1-1"
    if compact == "g1task21":
        return "G1Task2-1"
    return None


class ManipulationManager(Node):
    def __init__(self) -> None:
        super().__init__("manipulation_manager")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self._pub_state = self.create_publisher(String, "/manipulation/state", qos)
        self._pub_exec_cmd = self.create_publisher(String, "/situation/next_action", qos)
        self._pub_perception_req = self.create_publisher(String, PERCEPTION_REQUEST_TOPIC, qos)

        self.create_subscription(String, "/manipulation/command", self._on_command, qos)
        self.create_subscription(String, "/execution/state", self._on_execution_state, qos)
        self.create_subscription(String, PERCEPTION_RESULT_TOPIC, self._on_perception_result, qos)

        self._lock = threading.Lock()
        self._active_sequence_id: int | None = None
        self._active_operation: str | None = None
        self._active_execution_operation: str | None = None
        self._active_workflow_steps: list[str] = []
        self._active_workflow_index = -1
        self._waiting_for_perception = False
        self._active_status = ""
        self._active_message = ""
        # After can_pick=false on a Initialize workflow, next same Initialize skips Stand.
        self._skip_stand_on_next_initialize: str | None = None
        self.create_timer(10.0, self._publish_executing_heartbeat)

        LOGGER.info(
            "Sub /manipulation/command, /execution/state, %s",
            PERCEPTION_RESULT_TOPIC,
        )
        LOGGER.info(
            "Pub /manipulation/state, /situation/next_action, %s",
            PERCEPTION_REQUEST_TOPIC,
        )

    def publish_state(
        self,
        sequence_id: int | None,
        status: str,
        message: str,
        *,
        extra_payload: dict[str, Any] | None = None,
        log_publish: bool = False,
    ) -> None:
        payload = {
            "sequence_id": sequence_id,
            "status": status,
            "message": message,
        }
        if extra_payload:
            payload.update(extra_payload)
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._pub_state.publish(msg)
        if log_publish:
            LOGGER.info("publish /manipulation/state: %s", msg.data)

    def _reset_active(self) -> None:
        with self._lock:
            self._active_sequence_id = None
            self._active_operation = None
            self._active_execution_operation = None
            self._active_workflow_steps = []
            self._active_workflow_index = -1
            self._waiting_for_perception = False
            self._active_status = ""
            self._active_message = ""

    def _set_active_state_message(self, status: str, message: str) -> None:
        with self._lock:
            self._active_status = status
            self._active_message = message

    def _publish_executing_heartbeat(self) -> None:
        with self._lock:
            sid = self._active_sequence_id
            status = self._active_status
            message = self._active_message
        if sid is None or status.lower() != "executing":
            return
        self.publish_state(sid, "executing", message, log_publish=False)

    def _start_workflow(self, operation_type: str, sequence_id: int) -> tuple[bool, str]:
        steps = WORKFLOW_STEPS.get(operation_type)
        if not steps:
            return False, f"unknown workflow operation_type: {operation_type!r}"

        first_step = steps[0]
        with self._lock:
            self._active_workflow_steps = list(steps)
            self._active_workflow_index = 0
            self._active_execution_operation = first_step

        message = f"동작 중... ({first_step})"
        self._set_active_state_message("executing", message)
        self.publish_state(sequence_id, "executing", message)
        return self.execute_operation(first_step, sequence_id)

    def _start_initialize_perception_workflow(
        self, operation_type: str, sequence_id: int
    ) -> tuple[bool, str]:
        steps = INITIALIZE_PERCEPTION_WORKFLOWS.get(operation_type)
        if not steps:
            return False, f"unknown initialize workflow operation_type: {operation_type!r}"

        stand_step, perception_step = steps
        with self._lock:
            self._active_workflow_steps = [stand_step, perception_step]
            self._active_workflow_index = 0
            self._active_execution_operation = stand_step

        message = f"동작 중... ({stand_step})"
        self._set_active_state_message("executing", message)
        self.publish_state(sequence_id, "executing", message)
        return self.execute_operation(stand_step, sequence_id)

    def _start_initialize_perception_only(
        self, operation_type: str, sequence_id: int
    ) -> tuple[bool, str]:
        steps = INITIALIZE_PERCEPTION_WORKFLOWS.get(operation_type)
        if not steps:
            return False, f"unknown initialize workflow operation_type: {operation_type!r}"

        stand_step, perception_step = steps
        with self._lock:
            self._active_workflow_steps = [stand_step, perception_step]
            self._active_workflow_index = 1
            self._active_execution_operation = perception_step
            self._waiting_for_perception = True

        message = f"perception 확인 중... ({perception_step})"
        self._set_active_state_message("executing", message)
        self.publish_state(sequence_id, "executing", message, log_publish=True)
        ok, result_message = self._publish_named_perception_request(perception_step, sequence_id)
        if not ok:
            return False, result_message
        return True, result_message

    def execute_operation(
        self,
        operation_type_in: str,
        sequence_id: int | None = None,
        *,
        extra_payload: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        vega_id = resolve_vega_task_id(operation_type_in)
        if vega_id is None:
            return (
                False,
                f"unknown operation_type {operation_type_in!r}; "
                f"use one of {sorted(VEGA_TASK_IDS)} or an alias",
            )

        payload: dict[str, Any] = {
            "action": vega_id,
            "operation_type": vega_id,
            "vega_task_id": vega_id,
        }
        if sequence_id is not None:
            payload["sequence_id"] = sequence_id
        if extra_payload:
            payload.update(extra_payload)

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._pub_exec_cmd.publish(msg)
        if sequence_id is not None:
            LOGGER.info(
                "publish /situation/next_action sequence_id=%s: %s",
                sequence_id,
                msg.data,
            )
        else:
            LOGGER.info("publish /situation/next_action: %s", msg.data)
        return True, f"command forwarded: {vega_id}"

    def _publish_named_perception_request(
        self,
        operation_type: str,
        sequence_id: int,
        *,
        expected_execution_operation_type: str | None = None,
    ) -> tuple[bool, str]:
        payload = {
            "sequence_id": sequence_id,
            "operation_type": operation_type,
            "vega_task_id": operation_type,
            "request_type": "pick_check",
        }
        if expected_execution_operation_type is not None:
            payload["expected_execution_operation_type"] = expected_execution_operation_type
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._pub_perception_req.publish(msg)
        LOGGER.info(
            "publish %s sequence_id=%s: %s",
            PERCEPTION_REQUEST_TOPIC,
            sequence_id,
            msg.data,
        )
        return True, f"waiting perception result for {operation_type}"

    def _parse_perception_result(
        self, msg_str: str
    ) -> tuple[int | None, str | None, bool, dict[str, Any] | None]:
        raw = (msg_str or "").strip()
        if not raw:
            raise ValueError("empty perception result")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("perception result JSON must be an object")

        seq_val = parsed.get("sequence_id")
        sequence_id: int | None = None
        if seq_val is not None:
            try:
                sequence_id = int(seq_val)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid perception sequence_id") from exc

        op = parsed.get("operation_type")
        operation_type = str(op).strip() if op is not None else None
        if operation_type == "":
            operation_type = None

        can_pick = parsed.get("can_pick")
        if not isinstance(can_pick, bool):
            raise ValueError("perception can_pick must be boolean")

        offset = parsed.get("offset")
        if offset is not None and not isinstance(offset, dict):
            raise ValueError("perception offset must be an object")
        return sequence_id, operation_type, can_pick, offset

    def _parse_execution_message(
        self, msg_str: str
    ) -> tuple[str | None, int | None]:
        raw = (msg_str or "").strip()
        if not raw:
            return None, None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                value = parsed.get("state") or parsed.get("execution_state")
                if value is None:
                    return None, None
                seq_val = parsed.get("sequence_id")
                seq: int | None = None
                if seq_val is not None:
                    try:
                        seq = int(seq_val)
                    except (TypeError, ValueError):
                        seq = None
                return str(value).strip().upper(), seq
            if isinstance(parsed, str):
                return parsed.strip().upper(), None
        except json.JSONDecodeError:
            pass
        return raw.upper(), None

    def handle_command(self, msg_str: str) -> None:
        sequence_id: int | None = None
        try:
            data: Any = json.loads(msg_str)
            if not isinstance(data, dict):
                raise ValueError("command JSON must be an object")

            if "sequence_id" not in data:
                raise ValueError("missing sequence_id")
            try:
                sequence_id = int(data["sequence_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid sequence_id") from exc

            operation_type = data.get("operation_type")
            if not isinstance(operation_type, str) or not operation_type.strip():
                raise ValueError("missing operation_type")
            operation_type = operation_type.strip()

            skip_stand_override = data.get("skip_stand")
            if skip_stand_override is not None and not isinstance(skip_stand_override, bool):
                raise ValueError("skip_stand must be a boolean if present")

            perception_only = False
            with self._lock:
                if self._active_sequence_id is not None:
                    self.publish_state(sequence_id, "failed", "another command is running")
                    return
                if operation_type not in INITIALIZE_PERCEPTION_WORKFLOWS:
                    self._skip_stand_on_next_initialize = None
                elif (
                    self._skip_stand_on_next_initialize is not None
                    and self._skip_stand_on_next_initialize != operation_type
                ):
                    self._skip_stand_on_next_initialize = None
                if skip_stand_override is False:
                    self._skip_stand_on_next_initialize = None

                if operation_type in INITIALIZE_PERCEPTION_WORKFLOWS:
                    if skip_stand_override is True:
                        perception_only = True
                    elif skip_stand_override is False:
                        perception_only = False
                    else:
                        perception_only = (
                            self._skip_stand_on_next_initialize == operation_type
                        )
                    if perception_only:
                        self._skip_stand_on_next_initialize = None

                self._active_sequence_id = sequence_id
                self._active_operation = operation_type
                self._active_execution_operation = None
                self._waiting_for_perception = False

            LOGGER.info(
                "handling command sequence_id=%s operation_type=%s",
                sequence_id,
                operation_type,
            )

            if operation_type in INITIALIZE_PERCEPTION_WORKFLOWS:
                if perception_only:
                    ok, result_message = self._start_initialize_perception_only(
                        operation_type,
                        sequence_id,
                    )
                else:
                    ok, result_message = self._start_initialize_perception_workflow(
                        operation_type,
                        sequence_id,
                    )
            elif operation_type in WORKFLOW_STEPS:
                ok, result_message = self._start_workflow(operation_type, sequence_id)
            else:
                with self._lock:
                    self._active_execution_operation = operation_type
                self._set_active_state_message("executing", "동작 중...")
                self.publish_state(sequence_id, "executing", "동작 중...")
                ok, result_message = self.execute_operation(operation_type, sequence_id)

            if not ok:
                self._reset_active()
                self.publish_state(sequence_id, "failed", result_message)

        except Exception as exc:
            LOGGER.warning("command handling failed: %s", exc)
            self.publish_state(sequence_id, "failed", str(exc))

    def _on_perception_result(self, msg: String) -> None:
        try:
            seq, result_op, can_pick, offset = self._parse_perception_result(msg.data or "")
        except Exception as exc:
            LOGGER.warning("invalid perception result: %s", exc)
            return

        with self._lock:
            active_sid = self._active_sequence_id
            active_op = self._active_operation
            waiting = self._waiting_for_perception

        if not waiting or active_sid is None or active_op not in INITIALIZE_PERCEPTION_WORKFLOWS:
            return
        if seq is not None and seq != active_sid:
            return
        expected_result_op = active_op
        if active_op in INITIALIZE_PERCEPTION_WORKFLOWS:
            expected_result_op = INITIALIZE_PERCEPTION_WORKFLOWS[active_op][1]
        if result_op is not None and result_op != expected_result_op:
            return

        if active_op in INITIALIZE_PERCEPTION_WORKFLOWS:
            perception_step = INITIALIZE_PERCEPTION_WORKFLOWS[active_op][1]
            message = f"perception 결과 수신 ({perception_step})"
            extra_payload: dict[str, Any] = {
                "can_pick": can_pick,
                "operation_type": active_op,
            }
            if offset is not None:
                extra_payload["offset"] = offset

            with self._lock:
                if can_pick:
                    self._skip_stand_on_next_initialize = None
                else:
                    self._skip_stand_on_next_initialize = active_op

            self.publish_state(
                active_sid,
                "DONE",
                message,
                extra_payload=extra_payload,
                log_publish=True,
            )
            self._reset_active()
            return

    def _on_execution_state(self, msg: String) -> None:
        state, exec_seq = self._parse_execution_message(msg.data)
        if state is None:
            return

        with self._lock:
            sid = self._active_sequence_id
            op = self._active_execution_operation or self._active_operation
            waiting = self._waiting_for_perception
            workflow_steps = list(self._active_workflow_steps)
            workflow_index = self._active_workflow_index

        if sid is None or waiting:
            return
        if exec_seq is not None and exec_seq != sid:
            return

        if state == "EXECUTING":
            message = f"동작 중... ({op})"
            self._set_active_state_message("executing", message)
            self.publish_state(sid, "executing", message, log_publish=False)
            return

        if state == "DONE":
            if workflow_steps and 0 <= workflow_index < len(workflow_steps):
                next_index = workflow_index + 1
                if next_index < len(workflow_steps):
                    next_step = workflow_steps[next_index]
                    if (
                        self._active_operation in INITIALIZE_PERCEPTION_WORKFLOWS
                        and next_step == INITIALIZE_PERCEPTION_WORKFLOWS[self._active_operation][1]
                    ):
                        with self._lock:
                            self._active_workflow_index = next_index
                            self._active_execution_operation = next_step
                            self._waiting_for_perception = True
                        message = f"perception 확인 중... ({next_step})"
                        self._set_active_state_message("executing", message)
                        self.publish_state(sid, "executing", message, log_publish=True)
                        ok, result_message = self._publish_named_perception_request(
                            next_step,
                            sid,
                        )
                        if not ok:
                            self._reset_active()
                            self.publish_state(sid, "failed", result_message, log_publish=True)
                        return

                    with self._lock:
                        self._active_workflow_index = next_index
                        self._active_execution_operation = next_step
                    message = f"동작 중... ({next_step})"
                    self._set_active_state_message("executing", message)
                    self.publish_state(sid, "executing", message, log_publish=True)
                    ok, result_message = self.execute_operation(next_step, sid)
                    if not ok:
                        self._reset_active()
                        self.publish_state(sid, "failed", result_message, log_publish=True)
                    return

                self.publish_state(sid, "DONE", "완료", log_publish=True)
                self._reset_active()
                return

            self.publish_state(sid, "success", "완료", log_publish=True)
            self._reset_active()
            return

        if state in ("FAILED", "STOPPED"):
            self.publish_state(
                sid,
                "failed",
                f"execution {state.lower()}",
                log_publish=True,
            )
            self._reset_active()

    def _on_command(self, msg: String) -> None:
        self.handle_command(msg.data or "")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    rclpy.init()
    node = ManipulationManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
