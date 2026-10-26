"""PF2 retained-background idle policy; other phase rules remain in the pinned bundle.

This is a new Pi protocol, not a correction to the old per-sample 20% rule.
Global CPU tail mean <=20%; each sample <=40%. No power or causal inference.
"""
from copy import copy
from admission import readiness as original_readiness

VERSION = 'PF2_tick_weighted_mean20_peak40_v1'


def readiness(rows, policy, expected_boot, initial_oom, quiet_started_ns, now_ns, idle):
    if not idle:
        return original_readiness(rows, policy, expected_boot, initial_oom, quiet_started_ns, now_ns, False)
    if (policy.get('cpu_policy_version') != VERSION or
            policy.get('cpu_aggregation') != 'tick_weighted_tail_mean' or
            policy.get('max_idle_cpu_percent') != 20 or
            policy.get('max_idle_cpu_peak_percent') != 40):
        return dict(state='STOP', reason='PF2_idle_policy_not_frozen', remaining_s=0,
                    same_boot_repeats_allowed=True, formal_execution_authorized=False)
    # Preserve all original time, capacity, temperature, paging and freshness checks.
    peak_policy = copy(policy)
    peak_policy['max_idle_cpu_percent'] = policy['max_idle_cpu_peak_percent']
    proof = original_readiness(rows, peak_policy, expected_boot, initial_oom, quiet_started_ns, now_ns, True)
    proof['cpu_policy_version'] = VERSION
    if proof['reason'] == 'background_CPU_busy':
        proof['reason'] = 'background_CPU_peak_busy'
    if proof['state'] != 'READY':
        return proof
    tail = [row for row in rows if row['monotonic_ns'] >= proof['tail_start_ns']]
    invalid = any(not 0 <= row['cpu_busy_percent'] <= 100 for row in tail)
    pairs = list(zip(tail, tail[1:]))
    for before, after in pairs:
        delta_total = after['cpu_total_ticks'] - before['cpu_total_ticks']
        delta_idle = after['cpu_idle_ticks'] - before['cpu_idle_ticks']
        if not (delta_total > 0 and 0 <= delta_idle <= delta_total):
            invalid = True
    if invalid or not pairs:
        proof.update(state='STOP', reason='PF2_CPU_counter_interval_invalid', remaining_s=0)
        return proof
    dt = tail[-1]['cpu_total_ticks'] - tail[0]['cpu_total_ticks']
    di = tail[-1]['cpu_idle_ticks'] - tail[0]['cpu_idle_ticks']
    mean = 100 * (dt - di) / dt
    proof.update(cpu_tail_mean_percent=mean,
                 cpu_tail_peak_percent=max(row['cpu_busy_percent'] for row in tail),
                 cpu_tail_total_ticks=dt, cpu_tail_idle_ticks=di,
                 cpu_mean_gate_percent=20, cpu_peak_gate_percent=40)
    if mean > policy['max_idle_cpu_percent']:
        proof.update(state='WAIT', reason='background_CPU_mean_busy', remaining_s=policy['tail_s'])
    return proof
