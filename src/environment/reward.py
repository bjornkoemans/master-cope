"""
Reward functions for the AgentOptimizer environment.

Two modes:
  - "simple_duration": R = -wait_minutes per task completion.
  - "shaped": Multi-component reward with configurable weights.

When individual_rewards=True (IC3Net mode), each agent receives a
differentiated reward based on their role in the step (volunteered,
assigned, idle, etc.).  When False, all agents receive the same scalar.

The `self` parameter refers to an AgentOptimizerEnvironment instance.
"""

import numpy as np
from collections import defaultdict

from .entities import Status
from .work_schedule import working_seconds_between


# ── Default configuration ───────────────────────────────────────────
DEFAULT_REWARD_CONFIG = {
    "mode": "shaped",           # "simple_duration" | "shaped"
    "individual_rewards": False, # True = per-agent rewards (IC3Net)

    # --- shaped mode weights (set to 0.0 to disable a component) ---
    "w_wait": 1.0,              # R_wait:   penalty for task waiting time
    "w_fallback": 1.0,          # R_fallback: penalty when nobody volunteered
    "w_coordination": 0.0,      # R_coordination: bonus for selective volunteering
    "w_balance": 0.0,           # R_balance: penalty for queue imbalance within roles
    "w_volunteer": 0.0,         # R_volunteer: immediate reward for volunteering (at assignment)
    "w_assignment": 0.0,        # R_assignment: reward shaping based on estimated wait at assignment

    # --- shaped mode parameters ---
    "wait_scale": "log",        # "log" = -log(1+min), "linear" = -minutes
    "fallback_penalty": -5.0,   # per fallback event
    "coordination_bonus": 1.0,  # max bonus for perfect volunteer count
    "volunteer_bonus": 1.0,     # bonus per successful volunteer event
    "adaptive_fallback": False,  # when true, fallback_penalty is multiplied by mean |R_wait|
    "fallback_multiplier": 2.0, # multiplier for adaptive fallback scaling

    # --- volunteer overshoot controls ---
    "volunteer_winner_only": False,   # True = only assigned agent gets R_volunteer (not all volunteers)
    "w_waste": 0.0,                   # R_waste: penalty for volunteering but NOT getting assigned
    "waste_penalty": -1.0,            # base penalty per wasted volunteer event
    "coordination_penalize_excess": False,  # True = R_coordination becomes negative when n_vol >> required
}


def _get_wait_minutes(task, work_schedule_enabled: bool) -> float:
    """Compute waiting time in minutes for a completed task."""
    if work_schedule_enabled:
        wait_sec = working_seconds_between(
            task.assigned_timestamp,
            task.start_timestamp,
        )
    else:
        wait_sec = (
            task.start_timestamp - task.assigned_timestamp
        ).total_seconds()
    return max(wait_sec / 60.0, 0.0)


def _find_completed_task(self):
    """Find a task that completed at the current timestep."""
    completed_task = self.completed_task
    current_time = self.current_time

    # Check pending cases first (more likely to have recent completions)
    for case in self.pending_cases:
        ct = case.current_task
        if (
            ct is not None
            and ct.status == Status.COMPLETED
            and ct.completion_timestamp == current_time
        ):
            return ct

    # Check last completed case
    if not completed_task and self.completed_cases:
        last_case = self.completed_cases[-1]
        ct = last_case.current_task
        if (
            ct is not None
            and ct.status == Status.COMPLETED
            and ct.completion_timestamp == current_time
        ):
            return ct

    return completed_task


def _find_completed_case(self):
    """Find a case that completed at the current timestep."""
    if self.completed_cases:
        last_case = self.completed_cases[-1]
        if last_case.completion_timestamp == self.current_time:
            return last_case
    return None


def get_reward(self):
    """Compute the reward for the current step.

    Returns:
        dict[int, float] if individual_rewards=True (per-agent rewards),
        float otherwise (shared reward, broadcast by simulator).
    """
    rc = getattr(self, 'reward_config', DEFAULT_REWARD_CONFIG)
    mode = rc.get('mode', 'shaped')
    individual = rc.get('individual_rewards', False)

    # Read and reset per-step flags (set by step())
    fallback_flag = getattr(self, '_fallback_this_step', False)
    self._fallback_this_step = False

    agent_ids = [a.id for a in self.agents]

    # No reward when simulation is done
    if not self.pending_cases and not self.future_cases:
        if individual:
            return {aid: 0.0 for aid in agent_ids}
        return 0.0

    # ── Simple duration mode ────────────────────────────────────────
    if mode == 'simple_duration':
        completed_task = _find_completed_task(self)
        if completed_task:
            wait_min = _get_wait_minutes(
                completed_task,
                getattr(self, 'work_schedule_enabled', False),
            )
            if individual:
                rewards = {aid: 0.0 for aid in agent_ids}
                assigned_id = completed_task.assigned_agent.id if completed_task.assigned_agent else None
                if assigned_id is not None:
                    rewards[assigned_id] = -wait_min
                return rewards
            return -wait_min
        if individual:
            return {aid: 0.0 for aid in agent_ids}
        return 0.0

    # ── Shaped mode ─────────────────────────────────────────────────
    if mode == 'shaped':
        work_sched = getattr(self, 'work_schedule_enabled', False)
        completed_task = _find_completed_task(self)

        if individual:
            rewards = {aid: 0.0 for aid in agent_ids}
        else:
            reward = 0.0

        # ─ R_wait: penalty for task waiting time (at task completion) ─
        w_wait = rc.get('w_wait', 1.0)
        if completed_task and w_wait != 0.0:
            wait_min = _get_wait_minutes(completed_task, work_sched)
            wait_scale = rc.get('wait_scale', 'log')
            if wait_scale == 'log':
                r_wait = -np.log1p(wait_min)
            else:
                r_wait = -wait_min

            # Track running average of wait penalties for adaptive fallback
            if rc.get('adaptive_fallback', False):
                alpha = 0.01
                old_avg = getattr(self, '_running_avg_wait_penalty', abs(r_wait))
                self._running_avg_wait_penalty = (1 - alpha) * old_avg + alpha * abs(r_wait)

            if individual:
                # Only the assigned agent bears the wait penalty
                assigned_id = completed_task.assigned_agent.id if completed_task.assigned_agent else None
                if assigned_id is not None:
                    rewards[assigned_id] += w_wait * r_wait
                else:
                    for aid in agent_ids:
                        rewards[aid] += w_wait * r_wait / len(agent_ids)
            else:
                reward += w_wait * r_wait

        # ─ R_balance: queue imbalance penalty (every step) ─
        w_balance = rc.get('w_balance', 0.0)
        if w_balance != 0.0:
            role_queues = defaultdict(list)
            role_agents = defaultdict(list)
            for agent in self.agents:
                q = agent.case_queue.size()
                if agent.current_case is not None:
                    q += 1
                role_queues[agent.role].append(q)
                role_agents[agent.role].append(agent.id)

            if individual:
                # Per-agent: penalize deviation from role mean
                for role, queues in role_queues.items():
                    if len(queues) > 1:
                        mean_q = np.mean(queues)
                        if mean_q > 0:
                            for i, aid in enumerate(role_agents[role]):
                                deviation = (queues[i] - mean_q) / mean_q
                                rewards[aid] += w_balance * (-abs(deviation))
            else:
                r_balance = 0.0
                for role, queues in role_queues.items():
                    if len(queues) > 1:
                        mean_q = np.mean(queues)
                        if mean_q > 0:
                            cv = np.std(queues) / mean_q
                            r_balance -= cv
                reward += w_balance * r_balance

        # ─ R_fallback: penalty when nobody volunteered (at assignment) ─
        w_fallback = rc.get('w_fallback', 1.0)
        if fallback_flag and w_fallback != 0.0:
            fallback_penalty_base = rc.get('fallback_penalty', -5.0)
            # Apply adaptive scaling if enabled
            if rc.get('adaptive_fallback', False):
                avg_wait = getattr(self, '_running_avg_wait_penalty', 1.0)
                fallback_val = w_fallback * (-abs(avg_wait) * rc.get('fallback_multiplier', 2.0))
            else:
                fallback_val = w_fallback * fallback_penalty_base

            if individual:
                # Penalize agents who COULD have volunteered but didn't
                capable_ids = getattr(self, '_capable_agent_ids_this_step', None)
                if capable_ids:
                    per_agent = fallback_val / len(capable_ids)
                    for aid in capable_ids:
                        rewards[aid] += per_agent
                else:
                    for aid in agent_ids:
                        rewards[aid] += fallback_val / len(agent_ids)
            else:
                reward += fallback_val

        # ─ R_volunteer: immediate reward for volunteering (at assignment) ─
        w_vol = rc.get('w_volunteer', 0.0)
        if w_vol != 0.0 and not fallback_flag:
            vol_ids = getattr(self, '_volunteer_ids_this_step', None)
            if vol_ids:
                vol_bonus = w_vol * rc.get('volunteer_bonus', 1.0)
                if individual:
                    if rc.get('volunteer_winner_only', False):
                        # Only the assigned agent gets the volunteer bonus
                        assigned_id = getattr(self, '_assigned_agent_this_step', None)
                        if assigned_id is not None and assigned_id in vol_ids and assigned_id in rewards:
                            rewards[assigned_id] += vol_bonus
                    else:
                        # Legacy: split bonus among ALL volunteers
                        per_vol = vol_bonus / len(vol_ids)
                        for vid in vol_ids:
                            if vid in rewards:
                                rewards[vid] += per_vol
                else:
                    reward += vol_bonus

        # ─ R_waste: penalty for volunteering but NOT getting assigned ─
        w_waste = rc.get('w_waste', 0.0)
        if w_waste != 0.0 and not fallback_flag:
            vol_ids = getattr(self, '_volunteer_ids_this_step', None)
            assigned_id = getattr(self, '_assigned_agent_this_step', None)
            if vol_ids and assigned_id is not None and individual:
                waste_pen = w_waste * rc.get('waste_penalty', -1.0)
                for vid in vol_ids:
                    if vid != assigned_id and vid in rewards:
                        rewards[vid] += waste_pen

        # ─ R_assignment: estimated wait penalty at assignment time ─
        w_assign = rc.get('w_assignment', 0.0)
        if w_assign != 0.0:
            assigned_agent = getattr(self, '_assigned_agent_this_step', None)
            if assigned_agent is not None:
                agent_obj = self.agents[assigned_agent]
                q_len = agent_obj.case_queue.size()
                if agent_obj.current_case is not None:
                    q_len += 1
                # Estimate wait using global medians
                medians = getattr(self, 'global_activity_medians', {})
                avg_median = np.mean(list(medians.values())) if medians else 5.0
                est_wait_min = q_len * avg_median
                wait_scale = rc.get('wait_scale', 'log')
                if wait_scale == 'log':
                    r_assign = -np.log1p(est_wait_min)
                else:
                    r_assign = -est_wait_min
                if individual:
                    rewards[assigned_agent] += w_assign * r_assign
                else:
                    reward += w_assign * r_assign

        # ─ R_coordination: bonus for selective volunteering (at task completion) ─
        w_coord = rc.get('w_coordination', 0.0)
        if completed_task and w_coord != 0.0:
            n_vol = len(completed_task.volunteer_ids) if completed_task.volunteer_ids else 0
            req = completed_task.agents_required
            if n_vol > 0 and req > 0:
                ratio = min(n_vol, req) / max(n_vol, req)
                if rc.get('coordination_penalize_excess', False) and n_vol > req:
                    # Penalize excess: ratio goes negative when too many volunteer
                    # Perfect (n_vol == req): +1.0, double (n_vol == 2*req): 0.0, more: negative
                    r_coord = rc.get('coordination_bonus', 1.0) * (2.0 * ratio - 1.0)
                else:
                    # Legacy: always positive, approaches 0 with more excess
                    r_coord = rc.get('coordination_bonus', 1.0) * ratio
                if individual:
                    # Only volunteers get the coordination reward/penalty
                    per_vol = w_coord * r_coord / n_vol
                    for vid in completed_task.volunteer_ids:
                        if vid in rewards:
                            rewards[vid] += per_vol
                else:
                    reward += w_coord * r_coord

        if individual:
            return rewards
        return reward

    raise ValueError(f"Unknown reward mode: '{mode}'. Use 'simple_duration' or 'shaped'.")
