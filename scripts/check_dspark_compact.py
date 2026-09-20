import requests
import statistics
from collections import Counter

URL = "http://127.0.0.1:30100/server_info"

info = requests.get(URL, timeout=60).json()
states = info.get("internal_states", [])

# ============================================================
# Helper
# ============================================================

def pick_graph_tier(tokens, graph_tiers):
    """
    用当前运行中实际观察到的 graph key，估算：
    如果 static token 数为 tokens，会选择哪个 graph tier。

    例如观察到 graph tiers:
        [6, 12, 18, 24, 30, 36, 48]

    tokens = 14 -> 18
    tokens = 18 -> 18
    tokens = 19 -> 24

    注意：
    这是 offline estimate。
    如果 static 模式下还有额外 DP sync 行为，
    最精确的方法仍然是在 runtime 内直接记录 static_graph_key。
    """
    for tier in graph_tiers:
        if tier >= tokens:
            return tier

    # 超过当前观察到的最大 graph tier，无法可靠估计
    return None


# ============================================================
# First pass:
# 收集所有有效 record，同时收集真实出现过的 graph tiers
# ============================================================

rank_payloads = []
all_graph_tiers = set()

for rank, state in enumerate(states):
    payload = state.get("dspark_info_record")

    if not payload:
        continue

    gamma = int(payload.get("verify_num_draft_tokens", 6))

    records = [
        r
        for r in payload.get("records", [])
        if r.get("mode") == "compact"
        and r.get("num_running_reqs", 0) > 0
        and r.get("reqs")
    ]

    if not records:
        continue

    for r in records:
        if "verify_tokens_graph_key" in r:
            all_graph_tiers.add(int(r["verify_tokens_graph_key"]))

    rank_payloads.append((rank, gamma, records))


graph_tiers = sorted(all_graph_tiers)

print("=" * 120)
print("DSpark Compact Scheduling / Graph Tier Analysis")
print("=" * 120)
print(f"Observed graph tiers: {graph_tiers}")
print()


# ============================================================
# Global statistics
# ============================================================

global_actual = 0
global_static = 0
global_graph = 0

global_local = 0
global_dp = 0
global_dp_valid_steps = 0

global_est_static_graph = 0
global_est_static_graph_steps = 0

global_steps = 0

# Graph tier comparison
down_tier_steps = 0
same_tier_steps = 0
up_tier_steps = 0
unknown_tier_steps = 0

# 有 logical saving，但是 Graph tier 根本没降
logical_saved_same_graph_steps = 0

# 一个非常保守、完全不依赖 tier 推测的指标：
# compact graph < raw static logical
# 则 static graph 至少不可能比 raw static 小，
# 所以这一步一定发生了 Graph 降档
definite_graph_reduction_steps = 0

transition_hist = Counter()
graph_hist = Counter()
static_graph_hist = Counter()
verify_len_global_hist = Counter()


# ============================================================
# Per-rank
# ============================================================

for rank, gamma, records in rank_payloads:

    actual_list = []
    static_list = []
    graph_list = []
    local_list = []
    dp_list = []
    est_static_graph_list = []

    verify_len_hist = Counter()

    rank_down = 0
    rank_same = 0
    rank_up = 0
    rank_unknown = 0
    rank_saved_same = 0
    rank_definite_down = 0

    for r in records:
        reqs = r.get("reqs") or []

        # ----------------------------------------------------
        # 1. Compact 真正需要 verify 的 logical tokens
        # ----------------------------------------------------
        actual = sum(int(req["verify_len"]) for req in reqs)

        # ----------------------------------------------------
        # 2. 如果完全不用 compact
        #    每个 request 固定 gamma
        # ----------------------------------------------------
        static = len(reqs) * gamma

        # ----------------------------------------------------
        # 3. 当前 Compact 最终真正 replay 的 graph key
        # ----------------------------------------------------
        graph = int(r.get("verify_tokens_graph_key", actual))

        # ----------------------------------------------------
        # 4. Compact 中间过程
        # ----------------------------------------------------
        local_tier = int(r.get("verify_tokens_local", actual))
        dp_tier = int(r.get("verify_tokens_dp_synced", -1))

        # ----------------------------------------------------
        # 5. 估算 Static 情况会落到哪个 Graph tier
        #
        # 例如：
        # static logical = 18
        # observed tier = [6, 12, 18, 24, ...]
        # => static graph ~= 18
        # ----------------------------------------------------
        est_static_graph = pick_graph_tier(static, graph_tiers)

        actual_list.append(actual)
        static_list.append(static)
        graph_list.append(graph)
        local_list.append(local_tier)

        if dp_tier >= 0:
            dp_list.append(dp_tier)

        for req in reqs:
            v = int(req["verify_len"])
            verify_len_hist[v] += 1
            verify_len_global_hist[v] += 1

        graph_hist[graph] += 1

        # ----------------------------------------------------
        # Graph tier comparison
        # ----------------------------------------------------
        if est_static_graph is not None:
            est_static_graph_list.append(est_static_graph)
            static_graph_hist[est_static_graph] += 1

            transition_hist[(est_static_graph, graph)] += 1

            if graph < est_static_graph:
                rank_down += 1
                down_tier_steps += 1

            elif graph == est_static_graph:
                rank_same += 1
                same_tier_steps += 1

                # logical token 确实省了
                # 但是 graph 完全没降
                if actual < static:
                    rank_saved_same += 1
                    logical_saved_same_graph_steps += 1

            else:
                rank_up += 1
                up_tier_steps += 1

        else:
            rank_unknown += 1
            unknown_tier_steps += 1

        # ----------------------------------------------------
        # 这个是不依赖 graph-tier 推断的保守指标
        #
        # 如果：
        #     static logical = 18
        #     compact graph = 12
        #
        # 那么 static graph 无论如何至少要容纳 18，
        # 所以这里肯定真正发生了降档
        # ----------------------------------------------------
        if graph < static:
            rank_definite_down += 1
            definite_graph_reduction_steps += 1

    # --------------------------------------------------------
    # Rank summary
    # --------------------------------------------------------

    actual_sum = sum(actual_list)
    static_sum = sum(static_list)
    graph_sum = sum(graph_list)

    logical_reduction = (
        1.0 - actual_sum / static_sum
        if static_sum > 0
        else 0
    )

    if est_static_graph_list:
        est_static_graph_sum = sum(est_static_graph_list)

        estimated_graph_reduction = (
            1.0 - graph_sum / est_static_graph_sum
            if est_static_graph_sum > 0
            else 0
        )

        avg_static_graph = statistics.mean(est_static_graph_list)

    else:
        est_static_graph_sum = 0
        estimated_graph_reduction = 0
        avg_static_graph = float("nan")

    steps = len(records)

    print(
        f"DP{rank:02d}: "
        f"steps={steps:4d}, "
        f"avg actual={statistics.mean(actual_list):6.2f}, "
        f"avg static={statistics.mean(static_list):6.2f}, "
        f"avg compact graph={statistics.mean(graph_list):6.2f}, "
        f"avg est.static graph={avg_static_graph:6.2f}"
    )

    print(
        f"       logical reduction={logical_reduction * 100:6.2f}%, "
        f"estimated graph reduction={estimated_graph_reduction * 100:6.2f}%"
    )

    print(
        f"       graph tier: "
        f"down={rank_down:4d} ({rank_down / steps * 100:5.1f}%), "
        f"same={rank_same:4d} ({rank_same / steps * 100:5.1f}%), "
        f"up={rank_up:4d} ({rank_up / steps * 100:5.1f}%), "
        f"unknown={rank_unknown:4d}"
    )

    print(
        f"       logical saved but SAME graph tier: "
        f"{rank_saved_same:4d} "
        f"({rank_saved_same / steps * 100:5.1f}%)"
    )

    print(
        f"       definite graph down-tier lower bound: "
        f"{rank_definite_down:4d} "
        f"({rank_definite_down / steps * 100:5.1f}%)"
    )

    print(
        f"       verify_len distribution: "
        f"{dict(sorted(verify_len_hist.items()))}"
    )

    if dp_list:
        print(
            f"       avg local tier={statistics.mean(local_list):6.2f}, "
            f"avg dp-synced={statistics.mean(dp_list):6.2f}"
        )
    else:
        print(
            f"       avg local tier={statistics.mean(local_list):6.2f}, "
            f"avg dp-synced=N/A"
        )

    print()

    # global
    global_actual += actual_sum
    global_static += static_sum
    global_graph += graph_sum

    global_local += sum(local_list)

    if dp_list:
        global_dp += sum(dp_list)
        global_dp_valid_steps += len(dp_list)

    global_est_static_graph += est_static_graph_sum
    global_est_static_graph_steps += len(est_static_graph_list)

    global_steps += steps


# ============================================================
# Global result
# ============================================================

print("=" * 120)
print("GLOBAL SUMMARY")
print("=" * 120)

logical_reduction = (
    1.0 - global_actual / global_static
    if global_static > 0
    else 0
)

graph_padding_vs_actual = (
    global_graph / global_actual - 1.0
    if global_actual > 0
    else 0
)

if global_est_static_graph > 0:
    estimated_graph_reduction = (
        1.0 - global_graph / global_est_static_graph
    )
else:
    estimated_graph_reduction = 0


print(f"Total decode steps                  : {global_steps}")
print()

print(f"Static logical verify tokens        : {global_static}")
print(f"Compact scheduled tokens            : {global_actual}")
print(
    f"Logical token saving                : "
    f"{global_static - global_actual}"
)
print(
    f"Compact logical reduction           : "
    f"{logical_reduction * 100:.2f}%"
)

print()

print(f"Estimated STATIC graph slots         : {global_est_static_graph}")
print(f"Actual COMPACT graph slots           : {global_graph}")

if global_est_static_graph > 0:
    print(
        f"Estimated effective graph reduction  : "
        f"{estimated_graph_reduction * 100:.2f}%"
    )

print(
    f"Compact graph padding vs logical     : "
    f"{graph_padding_vs_actual * 100:.2f}%"
)

print()

valid_compare_steps = (
    down_tier_steps
    + same_tier_steps
    + up_tier_steps
)

if valid_compare_steps > 0:
    print("Graph tier comparison:")
    print(
        f"  compact < static                  : "
        f"{down_tier_steps:6d} "
        f"({down_tier_steps / valid_compare_steps * 100:6.2f}%)"
    )
    print(
        f"  compact = static                  : "
        f"{same_tier_steps:6d} "
        f"({same_tier_steps / valid_compare_steps * 100:6.2f}%)"
    )
    print(
        f"  compact > static                  : "
        f"{up_tier_steps:6d} "
        f"({up_tier_steps / valid_compare_steps * 100:6.2f}%)"
    )

    print()
    print(
        f"Logical saved BUT graph unchanged    : "
        f"{logical_saved_same_graph_steps:6d} "
        f"({logical_saved_same_graph_steps / valid_compare_steps * 100:6.2f}%)"
    )

print()

print(
    f"Definite graph down-tier lower bound : "
    f"{definite_graph_reduction_steps:6d} "
    f"({definite_graph_reduction_steps / global_steps * 100:6.2f}%)"
)

print()


# ============================================================
# Padding decomposition
# ============================================================

print("=" * 120)
print("COMPACT PADDING BREAKDOWN")
print("=" * 120)

print(f"Compact logical tokens       : {global_actual}")
print(f"Local tier tokens            : {global_local}")

if global_dp_valid_steps > 0:
    print(f"DP-synced tokens             : {global_dp}")
else:
    print(f"DP-synced tokens             : N/A")

print(f"Final graph slots            : {global_graph}")

if global_actual > 0:
    print(
        f"Logical -> local overhead    : "
        f"{(global_local / global_actual - 1) * 100:.2f}%"
    )

if global_dp_valid_steps > 0 and global_local > 0:
    print(
        f"Local -> DP sync overhead    : "
        f"{(global_dp / global_local - 1) * 100:.2f}%"
    )

if global_dp_valid_steps > 0 and global_dp > 0:
    print(
        f"DP sync -> graph overhead    : "
        f"{(global_graph / global_dp - 1) * 100:.2f}%"
    )


# ============================================================
# Distribution
# ============================================================

print()
print("=" * 120)
print("VERIFY LENGTH DISTRIBUTION")
print("=" * 120)

total_verify_reqs = sum(verify_len_global_hist.values())

for v, count in sorted(verify_len_global_hist.items()):
    pct = count / total_verify_reqs * 100 if total_verify_reqs else 0

    print(
        f"verify_len={v}: "
        f"{count:8d} "
        f"({pct:6.2f}%)"
    )


print()
print("=" * 120)
print("GRAPH KEY DISTRIBUTION")
print("=" * 120)

for tier, count in sorted(graph_hist.items()):
    print(
        f"compact graph={tier:4d}: "
        f"{count:6d} steps "
        f"({count / global_steps * 100:6.2f}%)"
    )


# ============================================================
# Static -> Compact transition histogram
# ============================================================

print()
print("=" * 120)
print("ESTIMATED STATIC GRAPH -> COMPACT GRAPH TRANSITIONS")
print("=" * 120)

for (static_graph, compact_graph), count in sorted(
    transition_hist.items(),
    key=lambda x: (x[0][0], x[0][1]),
):
    if compact_graph < static_graph:
        flag = "DOWN"
    elif compact_graph == static_graph:
        flag = "SAME"
    else:
        flag = "UP"

    print(
        f"{static_graph:4d} -> {compact_graph:4d} : "
        f"{count:6d} steps "
        f"({count / valid_compare_steps * 100:6.2f}%) "
        f"[{flag}]"
    )


print()
print("=" * 120)
print("NOTE")
print("=" * 120)
print(
    "Estimated STATIC graph slots are inferred from graph tiers observed "
    "during the compact run."
)
print(
    "Therefore 'estimated effective graph reduction' is an offline estimate."
)
print(
    "If static mode performs additional DP synchronization before graph-tier "
    "selection, the exact baseline should be recorded inside the runtime."
)