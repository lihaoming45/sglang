import argparse
import statistics
import sys
from collections import Counter

import requests

URL = "http://127.0.0.1:30100/server_info"


def pct(n, d):
    return (n / d * 100.0) if d else 0.0


def pick_graph_tier(tokens, graph_tiers):
    """Estimate the smallest observed graph tier that can hold `tokens`."""
    for tier in graph_tiers:
        if tier >= tokens:
            return tier
    return None


parser = argparse.ArgumentParser(
    description=(
        "Analyze DSpark static/compact verify scheduling with one unified output format."
    )
)
parser.add_argument(
    "--mode",
    choices=["auto", "compact", "static"],
    default="auto",
    help="Mode to analyze. auto selects the mode with the most valid records.",
)
parser.add_argument(
    "--url",
    default=URL,
    help=f"server_info URL (default: {URL})",
)
args = parser.parse_args()

try:
    response = requests.get(args.url, timeout=60)
    response.raise_for_status()
    info = response.json()
except Exception as exc:
    print(f"Failed to read {args.url}: {exc}")
    sys.exit(1)

states = info.get("internal_states", [])

# -----------------------------------------------------------------------------
# Detect available modes
# -----------------------------------------------------------------------------
mode_hist = Counter()
for state in states:
    payload = state.get("dspark_info_record")
    if not payload:
        continue
    for record in payload.get("records", []):
        mode = record.get("mode")
        if (
            mode in ("compact", "static")
            and record.get("num_running_reqs", 0) > 0
            and record.get("reqs")
        ):
            mode_hist[mode] += 1

if args.mode == "auto":
    if not mode_hist:
        print("No valid static/compact DSpark records found in /server_info.")
        sys.exit(1)
    selected_mode = mode_hist.most_common(1)[0][0]
else:
    selected_mode = args.mode

# -----------------------------------------------------------------------------
# Collect rank payloads + observed graph tiers
# -----------------------------------------------------------------------------
rank_payloads = []
all_graph_tiers = set()

for rank, state in enumerate(states):
    payload = state.get("dspark_info_record")
    if not payload:
        continue

    gamma = int(payload.get("verify_num_draft_tokens", 6))
    records = [
        record
        for record in payload.get("records", [])
        if record.get("mode") == selected_mode
        and record.get("num_running_reqs", 0) > 0
        and record.get("reqs")
    ]

    if not records:
        continue

    for record in records:
        if "verify_tokens_graph_key" in record:
            all_graph_tiers.add(int(record["verify_tokens_graph_key"]))

    rank_payloads.append((rank, gamma, records))

if not rank_payloads:
    print(f"No valid records found for mode={selected_mode!r}.")
    print(f"Detected valid record modes: {dict(mode_hist)}")
    sys.exit(1)

graph_tiers = sorted(all_graph_tiers)

print("=" * 120)
print("DSpark Verify Scheduling / Graph Tier Analysis")
print("=" * 120)
print(f"Detected valid record modes : {dict(mode_hist)}")
print(f"Selected analysis mode      : {selected_mode.upper()}")
print(f"Observed graph tiers        : {graph_tiers}")
print()

# -----------------------------------------------------------------------------
# Global accumulators
# -----------------------------------------------------------------------------
global_steps = 0
global_reqs = 0
global_actual = 0
global_static = 0
global_graph = 0
global_local = 0
global_dp = 0
global_dp_valid_steps = 0
global_est_static_graph = 0
global_est_static_graph_steps = 0

global_verify_hist = Counter()
global_graph_hist = Counter()
global_static_graph_hist = Counter()
gamma_hist = Counter()
transition_hist = Counter()

bad_vs_gamma = 0
bad_vs_six = 0

down_tier_steps = 0
same_tier_steps = 0
up_tier_steps = 0
unknown_tier_steps = 0
logical_saved_same_graph_steps = 0
definite_graph_reduction_steps = 0

# -----------------------------------------------------------------------------
# Per-rank analysis -- SAME metrics for static and compact
# -----------------------------------------------------------------------------
for rank, gamma, records in rank_payloads:
    gamma_hist[gamma] += 1

    actual_list = []
    static_list = []
    graph_list = []
    local_list = []
    dp_list = []
    est_static_graph_list = []
    verify_hist = Counter()

    rank_req_count = 0
    rank_bad_gamma = 0
    rank_bad_six = 0
    rank_down = 0
    rank_same = 0
    rank_up = 0
    rank_unknown = 0
    rank_saved_same = 0
    rank_definite_down = 0

    for record in records:
        reqs = record.get("reqs") or []
        verify_lens = [
            int(req["verify_len"])
            for req in reqs
            if "verify_len" in req
        ]

        # Actual logical verify tokens selected by the active mode.
        actual = sum(verify_lens)

        # Static baseline: every request verifies gamma tokens.
        static = len(reqs) * gamma

        # Final graph key actually replayed by the current run.
        graph = int(record.get("verify_tokens_graph_key", actual))

        # Intermediate values, if runtime exposed them.
        local_tier = int(record.get("verify_tokens_local", actual))
        dp_tier = int(record.get("verify_tokens_dp_synced", -1))

        # Offline estimate of the graph tier a static baseline would select.
        est_static_graph = pick_graph_tier(static, graph_tiers)

        actual_list.append(actual)
        static_list.append(static)
        graph_list.append(graph)
        local_list.append(local_tier)

        if dp_tier >= 0:
            dp_list.append(dp_tier)

        rank_req_count += len(verify_lens)
        for value in verify_lens:
            verify_hist[value] += 1
            global_verify_hist[value] += 1
            if value != gamma:
                rank_bad_gamma += 1
                bad_vs_gamma += 1
            if value != 6:
                rank_bad_six += 1
                bad_vs_six += 1

        global_graph_hist[graph] += 1

        if est_static_graph is not None:
            est_static_graph_list.append(est_static_graph)
            global_static_graph_hist[est_static_graph] += 1
            transition_hist[(est_static_graph, graph)] += 1

            if graph < est_static_graph:
                rank_down += 1
                down_tier_steps += 1
            elif graph == est_static_graph:
                rank_same += 1
                same_tier_steps += 1
                if actual < static:
                    rank_saved_same += 1
                    logical_saved_same_graph_steps += 1
            else:
                rank_up += 1
                up_tier_steps += 1
        else:
            rank_unknown += 1
            unknown_tier_steps += 1

        # Conservative lower bound: if graph < raw static logical tokens,
        # a graph down-tier definitely happened.
        if graph < static:
            rank_definite_down += 1
            definite_graph_reduction_steps += 1

    steps = len(records)
    actual_sum = sum(actual_list)
    static_sum = sum(static_list)
    graph_sum = sum(graph_list)
    est_static_graph_sum = sum(est_static_graph_list)

    logical_reduction = 1.0 - actual_sum / static_sum if static_sum else 0.0
    graph_reduction = (
        1.0 - graph_sum / est_static_graph_sum
        if est_static_graph_sum
        else 0.0
    )

    avg_verify_len = (
        sum(v * c for v, c in verify_hist.items()) / rank_req_count
        if rank_req_count
        else float("nan")
    )
    avg_graph = statistics.mean(graph_list) if graph_list else float("nan")
    avg_static_graph = (
        statistics.mean(est_static_graph_list)
        if est_static_graph_list
        else float("nan")
    )

    print(f"DP{rank:02d}: steps={steps:4d}, reqs={rank_req_count:5d}, gamma={gamma}")
    print(
        f"       avg verify_len={avg_verify_len:6.2f}, "
        f"verify_len distribution={dict(sorted(verify_hist.items()))}"
    )
    print(
        f"       all verify_len == gamma: {'YES' if rank_bad_gamma == 0 and rank_req_count else 'NO'} "
        f"(mismatch={rank_bad_gamma})"
    )
    if gamma == 6:
        print(
            f"       all verify_len == 6    : {'YES' if rank_bad_six == 0 and rank_req_count else 'NO'} "
            f"(mismatch={rank_bad_six})"
        )
    print(
        f"       avg actual={statistics.mean(actual_list):6.2f}, "
        f"avg static baseline={statistics.mean(static_list):6.2f}, "
        f"logical reduction={logical_reduction * 100:6.2f}%"
    )
    print(
        f"       avg actual graph={avg_graph:6.2f}, "
        f"avg est.static graph={avg_static_graph:6.2f}, "
        f"estimated graph reduction={graph_reduction * 100:6.2f}%"
    )
    print(
        f"       graph tier: down={rank_down:4d} ({pct(rank_down, steps):5.1f}%), "
        f"same={rank_same:4d} ({pct(rank_same, steps):5.1f}%), "
        f"up={rank_up:4d} ({pct(rank_up, steps):5.1f}%), "
        f"unknown={rank_unknown:4d}"
    )
    print(
        f"       logical saved but SAME graph tier: {rank_saved_same:4d} "
        f"({pct(rank_saved_same, steps):5.1f}%)"
    )
    print(
        f"       definite graph down-tier lower bound: {rank_definite_down:4d} "
        f"({pct(rank_definite_down, steps):5.1f}%)"
    )
    if dp_list:
        print(
            f"       avg local tier={statistics.mean(local_list):6.2f}, "
            f"avg dp-synced={statistics.mean(dp_list):6.2f}"
        )
    else:
        print(
            f"       avg local tier={statistics.mean(local_list):6.2f}, "
            "avg dp-synced=N/A"
        )
    print()

    global_steps += steps
    global_reqs += rank_req_count
    global_actual += actual_sum
    global_static += static_sum
    global_graph += graph_sum
    global_local += sum(local_list)

    if dp_list:
        global_dp += sum(dp_list)
        global_dp_valid_steps += len(dp_list)

    global_est_static_graph += est_static_graph_sum
    global_est_static_graph_steps += len(est_static_graph_list)

# -----------------------------------------------------------------------------
# Unified global summary
# -----------------------------------------------------------------------------
logical_reduction = (
    1.0 - global_actual / global_static if global_static else 0.0
)
estimated_graph_reduction = (
    1.0 - global_graph / global_est_static_graph
    if global_est_static_graph
    else 0.0
)
graph_padding_vs_actual = (
    global_graph / global_actual - 1.0 if global_actual else 0.0
)
avg_verify_len = (
    sum(v * c for v, c in global_verify_hist.items()) / global_reqs
    if global_reqs
    else float("nan")
)

print("=" * 120)
print("GLOBAL SUMMARY")
print("=" * 120)
print(f"Mode                                : {selected_mode.upper()}")
print(f"Total decode steps                  : {global_steps}")
print(f"Total verify requests               : {global_reqs}")
print(f"Gamma by rank                       : {dict(sorted(gamma_hist.items()))}")
print()
print(f"Average verify_len                  : {avg_verify_len:.4f}")
print(f"verify_len distribution             : {dict(sorted(global_verify_hist.items()))}")
print(f"verify_len != gamma count           : {bad_vs_gamma}")
print(
    f"ALL verify_len == gamma             : "
    f"{'YES' if bad_vs_gamma == 0 and global_reqs else 'NO'}"
)
if len(gamma_hist) == 1 and next(iter(gamma_hist)) == 6:
    print(f"verify_len != 6 count               : {bad_vs_six}")
    print(
        f"ALL verify_len == 6                 : "
        f"{'YES' if bad_vs_six == 0 and global_reqs else 'NO'}"
    )
print()
print(f"Static baseline verify tokens       : {global_static}")
print(f"Actual scheduled verify tokens      : {global_actual}")
print(f"Logical token saving                : {global_static - global_actual}")
print(f"Logical reduction                   : {logical_reduction * 100:.2f}%")
print()
print(f"Estimated STATIC graph slots        : {global_est_static_graph}")
print(f"Actual graph slots                  : {global_graph}")
if global_est_static_graph:
    print(f"Estimated effective graph reduction : {estimated_graph_reduction * 100:.2f}%")
else:
    print("Estimated effective graph reduction : N/A")
print(f"Graph padding vs logical            : {graph_padding_vs_actual * 100:.2f}%")
print()

valid_compare_steps = down_tier_steps + same_tier_steps + up_tier_steps
print("Graph tier comparison:")
print(
    f"  actual < estimated static         : {down_tier_steps:6d} "
    f"({pct(down_tier_steps, valid_compare_steps):6.2f}%)"
)
print(
    f"  actual = estimated static         : {same_tier_steps:6d} "
    f"({pct(same_tier_steps, valid_compare_steps):6.2f}%)"
)
print(
    f"  actual > estimated static         : {up_tier_steps:6d} "
    f"({pct(up_tier_steps, valid_compare_steps):6.2f}%)"
)
print(f"  unknown                           : {unknown_tier_steps:6d}")
print()
print(
    f"Logical saved BUT graph unchanged   : {logical_saved_same_graph_steps:6d} "
    f"({pct(logical_saved_same_graph_steps, valid_compare_steps):6.2f}%)"
)
print(
    f"Definite graph down-tier lower bound: {definite_graph_reduction_steps:6d} "
    f"({pct(definite_graph_reduction_steps, global_steps):6.2f}%)"
)

# -----------------------------------------------------------------------------
# Unified intermediate / padding breakdown
# -----------------------------------------------------------------------------
print()
print("=" * 120)
print("SCHEDULING / GRAPH PADDING BREAKDOWN")
print("=" * 120)
print(f"Actual logical tokens               : {global_actual}")
print(f"Local tier tokens                   : {global_local}")
if global_dp_valid_steps:
    print(f"DP-synced tokens                    : {global_dp}")
else:
    print("DP-synced tokens                    : N/A")
print(f"Final graph slots                   : {global_graph}")

if global_actual:
    print(
        f"Logical -> local overhead           : "
        f"{(global_local / global_actual - 1.0) * 100:.2f}%"
    )
if global_dp_valid_steps and global_local:
    print(
        f"Local -> DP sync overhead           : "
        f"{(global_dp / global_local - 1.0) * 100:.2f}%"
    )
if global_dp_valid_steps and global_dp:
    print(
        f"DP sync -> graph overhead           : "
        f"{(global_graph / global_dp - 1.0) * 100:.2f}%"
    )

# -----------------------------------------------------------------------------
# Unified distributions
# -----------------------------------------------------------------------------
print()
print("=" * 120)
print("VERIFY LENGTH DISTRIBUTION")
print("=" * 120)
for value, count in sorted(global_verify_hist.items()):
    print(
        f"verify_len={value}: {count:8d} "
        f"({pct(count, global_reqs):6.2f}%)"
    )

print()
print("=" * 120)
print("GRAPH KEY DISTRIBUTION")
print("=" * 120)
for tier, count in sorted(global_graph_hist.items()):
    print(
        f"graph={tier:4d}: {count:6d} steps "
        f"({pct(count, global_steps):6.2f}%)"
    )

print()
print("=" * 120)
print("ESTIMATED STATIC GRAPH -> ACTUAL GRAPH TRANSITIONS")
print("=" * 120)
for (static_graph, actual_graph), count in sorted(
    transition_hist.items(), key=lambda item: (item[0][0], item[0][1])
):
    if actual_graph < static_graph:
        flag = "DOWN"
    elif actual_graph == static_graph:
        flag = "SAME"
    else:
        flag = "UP"

    print(
        f"{static_graph:4d} -> {actual_graph:4d} : {count:6d} steps "
        f"({pct(count, valid_compare_steps):6.2f}%) [{flag}]"
    )

print()
print("=" * 120)
print("NOTE")
print("=" * 120)
print(
    "Static baseline verify tokens are computed as num_requests * gamma for both modes."
)
print(
    "Estimated STATIC graph slots are inferred from graph tiers observed in the selected run."
)
print(
    "For exact static-vs-compact graph comparison, run the same workload once in static mode "
    "and once in compact mode and compare the two saved outputs."
)
