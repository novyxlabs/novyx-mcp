"""
Cloud backend for Novyx MCP.

Thin wrapper around the Novyx Python SDK client.
Preserves exact existing behavior — every method delegates to the SDK.
"""

from __future__ import annotations

import json
import os
from typing import Any

from novyx import Novyx


class CloudBackend:
    """Delegates all operations to the Novyx Cloud API via the Python SDK."""

    def __init__(self) -> None:
        api_key = os.environ.get("NOVYX_API_KEY")
        if not api_key:
            raise ValueError("NOVYX_API_KEY environment variable is required")
        try:
            self._client = Novyx(api_key=api_key, source="mcp")
        except TypeError:
            # Older SDK versions don't accept source=
            self._client = Novyx(api_key=api_key)

    # ------------------------------------------------------------------
    # Core Memory
    # ------------------------------------------------------------------

    def remember(self, observation: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.remember(observation, **kwargs)

    def recall(self, query: str, **kwargs: Any) -> Any:
        """Returns the SDK's SearchResult object."""
        return self._client.recall(query, **kwargs)

    def forget(self, memory_id: str) -> bool:
        return self._client.forget(memory_id)

    def memories(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._client.memories(**kwargs)

    def memory(self, memory_id: str) -> dict[str, Any]:
        return self._client.memory(memory_id)

    def stats(self) -> dict[str, Any]:
        return self._client.stats()

    def draft_memory(self, observation: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.draft_memory(observation, **kwargs)

    def memory_drafts(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.memory_drafts(**kwargs)

    def memory_draft(self, draft_id: str) -> dict[str, Any]:
        return self._client.memory_draft(draft_id)

    def draft_diff(self, draft_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.draft_diff(draft_id, **kwargs)

    def merge_draft(self, draft_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.merge_draft(draft_id, **kwargs)

    def reject_draft(self, draft_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.reject_draft(draft_id, **kwargs)

    def memory_branch(self, branch_id: str) -> dict[str, Any]:
        return self._client.memory_branch(branch_id)

    def merge_branch(self, branch_id: str) -> dict[str, Any]:
        return self._client.merge_branch(branch_id)

    def reject_branch(self, branch_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.reject_branch(branch_id, **kwargs)

    # ------------------------------------------------------------------
    # Rollback & Audit
    # ------------------------------------------------------------------

    def rollback(self, target: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.rollback(target, **kwargs)

    def audit(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._client.audit(**kwargs)

    # ------------------------------------------------------------------
    # Links & Knowledge Graph
    # ------------------------------------------------------------------

    def link(self, source_id: str, target_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.link(source_id, target_id, **kwargs)

    def triple(self, subject: str, predicate: str, object_name: str) -> dict[str, Any]:
        return self._client.triple(subject, predicate, object_name)

    def triples(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.triples(**kwargs)

    # ------------------------------------------------------------------
    # Context Spaces
    # ------------------------------------------------------------------

    def create_space(self, name: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.create_space(name=name, **kwargs)

    def list_spaces(self) -> dict[str, Any]:
        return self._client.list_spaces()

    def get_space(self, space_id: str) -> dict[str, Any]:
        return self._client.get_space(space_id)

    def update_space(self, space_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.update_space(space_id, **kwargs)

    def delete_space(self, space_id: str) -> None:
        self._client.delete_space(space_id)

    def space_memories(self, space_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.space_memories(space_id, **kwargs)

    def share_context(self, tag: str, to_email: str, permission: str = "read") -> dict[str, Any]:
        return self._client.share_context(tag, to_email=to_email, permission=permission)

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------

    def usage(self) -> dict[str, Any]:
        return self._client.usage()

    # ------------------------------------------------------------------
    # Replay (Pro+)
    # ------------------------------------------------------------------

    def replay_timeline(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.replay_timeline(**kwargs)

    def replay_snapshot(self, at: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.replay_snapshot(at, **kwargs)

    def replay_lifecycle(self, memory_id: str) -> dict[str, Any]:
        return self._client.replay_lifecycle(memory_id)

    def replay_diff(self, start: str, end: str) -> dict[str, Any]:
        return self._client.replay_diff(start, end)

    # ------------------------------------------------------------------
    # Cortex (Pro+)
    # ------------------------------------------------------------------

    def cortex_status(self) -> dict[str, Any]:
        return self._client.cortex_status()

    def cortex_run(self) -> dict[str, Any]:
        return self._client.cortex_run()

    def cortex_insights(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.cortex_insights(**kwargs)

    def cortex_config(self) -> dict[str, Any]:
        return self._client.cortex_config()

    def cortex_update_config(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.cortex_update_config(**kwargs)

    # ------------------------------------------------------------------
    # Supersede
    # ------------------------------------------------------------------

    def supersede(self, old_memory_id: str, new_memory_id: str) -> dict[str, Any]:
        return self._client.supersede(old_memory_id, new_memory_id)

    # ------------------------------------------------------------------
    # Links (extended)
    # ------------------------------------------------------------------

    def unlink(self, source_id: str, target_id: str) -> dict[str, Any]:
        return self._client.unlink(source_id, target_id)

    def links(self, memory_id: str, *, relation: str | None = None) -> dict[str, Any]:
        return self._client.links(memory_id, relation=relation)

    def edges(self, *, memory_id: str | None = None, relation: str | None = None,
              direction: str = "both", limit: int = 100) -> dict[str, Any]:
        return self._client.edges(memory_id=memory_id, relation=relation,
                                  direction=direction, limit=limit)

    # ------------------------------------------------------------------
    # Knowledge Graph (extended)
    # ------------------------------------------------------------------

    def delete_triple(self, triple_id: str) -> dict[str, Any]:
        return self._client.delete_triple(triple_id)

    def entities(self, *, limit: int = 100, offset: int = 0,
                 entity_type: str | None = None) -> dict[str, Any]:
        return self._client.entities(limit=limit, offset=offset, entity_type=entity_type)

    def entity(self, entity_id: str) -> dict[str, Any]:
        return self._client.entity(entity_id)

    def delete_entity(self, entity_id: str) -> dict[str, Any]:
        return self._client.delete_entity(entity_id)

    # ------------------------------------------------------------------
    # Rollback (extended)
    # ------------------------------------------------------------------

    def rollback_preview(self, target: str) -> dict[str, Any]:
        return self._client.rollback_preview(target)

    def rollback_history(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._client.rollback_history(limit=limit)

    # ------------------------------------------------------------------
    # Audit (extended)
    # ------------------------------------------------------------------

    def audit_verify(self) -> dict[str, Any]:
        return self._client.audit_verify()

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    def trace_create(self, name: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.trace_create(name, **kwargs)

    def trace_step(self, trace_id: str, step_name: str, **kwargs: Any) -> dict[str, Any]:
        content = kwargs.pop("content", None)
        metadata = kwargs.pop("metadata", None)
        if content is None and kwargs:
            content = {
                key: value
                for key, value in {
                    "input": kwargs.pop("input_data", None),
                    "output": kwargs.pop("output_data", None),
                }.items()
                if value is not None
            }
        if content is None:
            content_text = ""
        elif isinstance(content, str):
            content_text = content
        else:
            content_text = json.dumps(content, default=str)
        return self._client.trace_step(trace_id, step_name, content_text, metadata=metadata)

    def trace_complete(self, trace_id: str) -> dict[str, Any]:
        return self._client.trace_complete(trace_id)

    def trace_verify(self, trace_id: str) -> dict[str, Any]:
        return self._client.trace_verify(trace_id)

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def eval_run(self, *, min_score: float | None = None) -> dict[str, Any]:
        return self._client.eval_run(min_score=min_score)

    def eval_gate(self, min_score: float) -> dict[str, Any]:
        return self._client.eval_gate(min_score)

    def eval_history(self, *, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        return self._client.eval_history(limit=limit, offset=offset)

    def eval_drift(self, *, days: int = 7) -> dict[str, Any]:
        return self._client.eval_drift(days=days)

    # ------------------------------------------------------------------
    # Replay (extended)
    # ------------------------------------------------------------------

    def replay_memory(self, memory_id: str) -> dict[str, Any]:
        return self._client.replay_memory(memory_id)

    def replay_recall(self, query: str, at: str, *, limit: int = 5) -> dict[str, Any]:
        return self._client.replay_recall(query, at, limit=limit)

    def replay_drift(self, from_ts: str, to_ts: str) -> dict[str, Any]:
        return self._client.replay_drift(from_ts, to_ts)

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def context_now(self) -> dict[str, Any]:
        return self._client.context_now()

    def dashboard(self) -> dict[str, Any]:
        return self._client.dashboard()

    # ------------------------------------------------------------------
    # Sharing (extended)
    # ------------------------------------------------------------------

    def accept_shared_context(self, token: str) -> dict[str, Any]:
        return self._client.accept_shared_context(token)

    def shared_contexts(self) -> dict[str, Any]:
        return self._client.shared_contexts()

    def revoke_shared_context(self, token: str) -> dict[str, Any]:
        return self._client.revoke_shared_context(token)

    # ------------------------------------------------------------------
    # Eval Baselines (Pro+)
    # ------------------------------------------------------------------

    def eval_baseline_create(self, query: str, expected_observation: str) -> dict[str, Any]:
        return self._client.eval_baseline_create(query, expected_observation)

    def eval_baselines(self) -> dict[str, Any]:
        return self._client.eval_baselines()

    def eval_baseline_delete(self, baseline_id: str) -> bool:
        return self._client.eval_baseline_delete(baseline_id)

    # ------------------------------------------------------------------
    # Audit Export (Pro+)
    # ------------------------------------------------------------------

    def audit_export(self, format: str = "json") -> dict[str, Any]:
        data = self._client.audit_export(format=format)
        # SDK returns raw bytes; decode for MCP tool consumption
        if isinstance(data, bytes):
            import json as _json
            try:
                return _json.loads(data.decode())
            except Exception:
                return {"raw": data.decode(errors="replace")}
        return data if isinstance(data, dict) else {"raw": str(data)}

    # ------------------------------------------------------------------
    # Explain Action (Inspectability)
    # ------------------------------------------------------------------

    def explain_action(self, action_id: str) -> dict[str, Any]:
        return self._client.explain_action(action_id)

    # ------------------------------------------------------------------
    # Sentinel Intel — Threat Intelligence (Pro+)
    # ------------------------------------------------------------------

    def threat_feed(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.threat_feed(**kwargs)

    def threat_stats(self) -> dict[str, Any]:
        return self._client.threat_stats()

    def threat_record(self, threat_event: dict) -> dict[str, Any]:
        return self._client.threat_record(threat_event)

    def threat_trending(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.threat_trending(**kwargs)

    def threat_match(self, threat_event: dict, **kwargs: Any) -> dict[str, Any]:
        return self._client.threat_match(threat_event, **kwargs)

    def threat_signature(self, signature_id: str) -> dict[str, Any]:
        return self._client.threat_signature(signature_id)

    def threat_mitigate(self, signature_id: str) -> dict[str, Any]:
        return self._client.threat_mitigate(signature_id)

    # ------------------------------------------------------------------
    # Sentinel Intel — Auto Defense (Pro+)
    # ------------------------------------------------------------------

    def defense_list(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.defense_list(**kwargs)

    def defense_deploy(self, signature_id: str, rule_type: str, rule_config: dict | None = None) -> dict[str, Any]:
        return self._client.defense_deploy(signature_id, rule_type, rule_config)

    def defense_remove(self, defense_id: str) -> dict[str, Any]:
        return self._client.defense_remove(defense_id)

    def defense_effectiveness(self, defense_id: str) -> dict[str, Any]:
        return self._client.defense_effectiveness(defense_id)

    def defense_record_block(self, defense_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.defense_record_block(defense_id, **kwargs)

    def defense_stats(self) -> dict[str, Any]:
        return self._client.defense_stats()

    def defense_recommend(self, signature_id: str) -> dict[str, Any]:
        return self._client.defense_recommend(signature_id)

    # ------------------------------------------------------------------
    # Sentinel Intel — Correlation (Pro+)
    # ------------------------------------------------------------------

    def correlate_threat(self, threat_event: dict) -> dict[str, Any]:
        return self._client.correlate_threat(threat_event)

    def detect_campaign(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.detect_campaign(**kwargs)

    def coordinated_attack_check(self, threat_events: list, **kwargs: Any) -> dict[str, Any]:
        return self._client.coordinated_attack_check(threat_events, **kwargs)

    def related_signatures(self, signature_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.related_signatures(signature_id, **kwargs)

    # ------------------------------------------------------------------
    # Streams
    # ------------------------------------------------------------------

    def stream_status(self) -> dict[str, Any]:
        return self._client.stream_status()

    # ------------------------------------------------------------------
    # Runtime v2: Agents (Pro+)
    # ------------------------------------------------------------------

    def create_agent(self, **kwargs: Any) -> dict[str, Any]:
        name = kwargs.pop("name")
        return self._client.create_agent(name, **kwargs)

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        return self._client.get_agent(agent_id)

    def list_agents(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.list_agents(**kwargs)

    def update_agent(self, agent_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.update_agent(agent_id, **kwargs)

    def delete_agent(self, agent_id: str) -> dict[str, Any]:
        return self._client.delete_agent(agent_id)

    # ------------------------------------------------------------------
    # Runtime v2: Missions (Pro+)
    # ------------------------------------------------------------------

    def create_mission(self, **kwargs: Any) -> dict[str, Any]:
        agent_id = kwargs.pop("agent_id")
        goal = kwargs.pop("goal")
        return self._client.create_mission(agent_id, goal, **kwargs)

    def get_mission(self, mission_id: str) -> dict[str, Any]:
        return self._client.get_mission(mission_id)

    def list_missions(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.list_missions(**kwargs)

    def update_mission(self, mission_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.update_mission(mission_id, **kwargs)

    def delete_mission(self, mission_id: str) -> dict[str, Any]:
        return self._client.delete_mission(mission_id)

    def pause_mission(self, mission_id: str) -> dict[str, Any]:
        return self._client.pause_mission(mission_id)

    def resume_mission(self, mission_id: str) -> dict[str, Any]:
        return self._client.resume_mission(mission_id)

    def cancel_mission(self, mission_id: str) -> dict[str, Any]:
        return self._client.cancel_mission(mission_id)

    # ------------------------------------------------------------------
    # Runtime v2: Capabilities (Pro+)
    # ------------------------------------------------------------------

    def create_capability(self, **kwargs: Any) -> dict[str, Any]:
        name = kwargs.pop("name")
        return self._client.create_capability(name, **kwargs)

    def get_capability(self, capability_id: str) -> dict[str, Any]:
        return self._client.get_capability(capability_id)

    def list_capabilities(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.list_capabilities(**kwargs)

    def update_capability(self, capability_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.update_capability(capability_id, **kwargs)

    def delete_capability(self, capability_id: str) -> dict[str, Any]:
        return self._client.delete_capability(capability_id)

    # ------------------------------------------------------------------
    # Runtime v2: Checkpoints (Pro+)
    # ------------------------------------------------------------------

    def create_checkpoint(self, **kwargs: Any) -> dict[str, Any]:
        mission_id = kwargs.pop("mission_id")
        return self._client.create_checkpoint(mission_id, **kwargs)

    def get_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        return self._client.get_checkpoint(checkpoint_id)

    def list_checkpoints(self, mission_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._client.list_checkpoints(mission_id, **kwargs)

    def rollback_to_checkpoint(self, **kwargs: Any) -> dict[str, Any]:
        mission_id = kwargs.pop("mission_id")
        checkpoint_id = kwargs.pop("checkpoint_id")
        return self._client.rollback_to_checkpoint(mission_id, checkpoint_id, **kwargs)

    # ------------------------------------------------------------------
    # Runtime v2: Interventions (Pro+)
    # ------------------------------------------------------------------

    def create_intervention(self, **kwargs: Any) -> dict[str, Any]:
        intervention_type = kwargs.pop("intervention_type")
        return self._client.create_intervention(intervention_type, **kwargs)

    def get_intervention(self, intervention_id: str) -> dict[str, Any]:
        return self._client.get_intervention(intervention_id)

    def list_interventions(self, **kwargs: Any) -> dict[str, Any]:
        return self._client.list_interventions(**kwargs)
