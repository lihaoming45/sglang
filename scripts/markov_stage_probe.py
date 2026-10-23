#!/usr/bin/env python3
"""Read-only stage diagnostics for SGLang MarkovSampleStep on an Ascend NPU.

Uses the checkout's ORIGINAL kernels, never execute() or the legacy adapter.
No model weights or distributed processes are required. See --help.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

NO_INDEX = 2147483647
MASK32 = 0xFFFFFFFF


def philox_numpy(key, c0, c1, c2, c3):
    """Independent uint64 host reference for Philox4x32-10 (no signed widening)."""
    c0, c1, c2, c3 = [x.astype(np.uint64) for x in np.broadcast_arrays(c0, c1, c2, c3)]
    k0, k1 = np.uint64(key & MASK32), np.uint64((key >> 32) & MASK32)
    mask = np.uint64(MASK32)
    for _ in range(10):
        a = c0 * np.uint64(0xD2511F53)
        b = c2 * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = (
            (b >> np.uint64(32)) ^ c1 ^ k0,
            b & mask,
            (a >> np.uint64(32)) ^ c3 ^ k1,
            a & mask,
        )
        k0, k1 = (k0 + np.uint64(0x9E3779B9)) & mask, (k1 + np.uint64(0xBB67AE85)) & mask
    return c0, c1, c2, c3


def score_keys(value, temperatures, *, key, offset, step, start=0, sampled=True):
    rows, padded = value.shape
    if not sampled:
        return value.copy()
    q = np.arange((padded + 3) // 4, dtype=np.uint64)[None, :] + np.uint64(start // 4)
    words = philox_numpy(key, q, np.arange(rows, dtype=np.uint64)[:, None],
                         np.array(step, dtype=np.uint64), np.array(offset & MASK32, dtype=np.uint64))
    bits = np.stack(words, axis=-1).reshape(rows, -1)[:, :padded]
    u = (bits.astype(np.float32) + np.float32(0.5)) * np.float32(2.0 ** -32)
    safe = np.where(temperatures != 0, temperatures, np.float32(1))[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        noise = -np.log1p(-u)
        sampled_keys = value / safe - np.log(noise)
    return np.where((temperatures != 0)[:, None], sampled_keys, value)


def reduce_scores(values, ids):
    """Last axis: NaNs first, otherwise maximum; ties choose minimum valid ID."""
    valid = ids != NO_INDEX
    nan = valid & np.isnan(values)
    has_nan = nan.any(axis=-1, keepdims=True)
    finite = np.where(valid & ~nan, values, -np.inf)
    maximum = finite.max(axis=-1, keepdims=True)
    wins = np.where(has_nan, nan, valid & (finite == maximum))
    return np.where(wins, ids, NO_INDEX).min(axis=-1).astype(np.int64)


def decode_candidates(packed, rows, parts, world=1):
    z = np.ascontiguousarray(packed, dtype=np.float32).reshape(world, rows, parts, 2)
    values = z[..., 0].transpose(1, 0, 2).reshape(rows, world * parts)
    # Interpret bits on CPU: do NOT convert numeric float values to integer IDs.
    ids = (z.view(np.int32)[..., 1] & NO_INDEX).transpose(1, 0, 2).reshape(rows, world * parts)
    return values, ids


def finish_reference(packed, rows, parts, world=1):
    return reduce_scores(*decode_candidates(packed, rows, parts, world))


def expected_partial(keys, width, block, start=0):
    rows, padded = keys.shape
    parts = (padded + block - 1) // block
    scores = np.full((rows, parts), -np.inf, np.float32)
    ids = np.full((rows, parts), NO_INDEX, np.int32)
    gaps = np.full((rows, parts), np.inf, np.float32)
    for p in range(parts):
        lo, hi = p * block, min((p + 1) * block, width)
        if hi <= lo:
            continue
        v = keys[:, lo:hi]
        ix = np.broadcast_to(np.arange(lo + start, hi + start, dtype=np.int32), v.shape)
        chosen = reduce_scores(v, ix)
        ids[:, p] = chosen
        scores[:, p] = v[np.arange(rows), chosen - start - lo]
        if hi - lo >= 2:
            sorted_v = np.sort(v, axis=-1)
            with np.errstate(invalid="ignore"):
                gaps[:, p] = sorted_v[:, -1] - sorted_v[:, -2]
    return scores, ids, gaps


def compare_partial(packed, keys, width, block, start, atol):
    rows, padded = keys.shape
    parts = (padded + block - 1) // block
    actual_v, actual_i = decode_candidates(packed, rows, parts)
    ref_v, ref_i, gap = expected_partial(keys, width, block, start)
    same_i = actual_i == ref_i
    same_v = np.isclose(actual_v, ref_v, rtol=2e-5, atol=atol, equal_nan=True)
    near = (~same_i) & np.isfinite(gap) & (gap <= 4 * atol)
    severe = (~same_i) & ~near
    bad = (~same_i) | (~same_v)
    samples = []
    for row, part in np.argwhere(bad)[:12]:
        samples.append(dict(row=int(row), part=int(part), expected_id=int(ref_i[row, part]),
                            actual_id=int(actual_i[row, part]), expected_score=float(ref_v[row, part]),
                            actual_score=float(actual_v[row, part]), host_top_gap=float(gap[row, part])))
    return dict(id_mismatch=int((~same_i).sum()), score_mismatch=int((~same_v).sum()),
                near_tie_id_mismatch=int(near.sum()), strong_id_mismatch=int(severe.sum()),
                nan_scores=int(np.isnan(actual_v).sum()), sentinel_ids=int((actual_i == NO_INDEX).sum()),
                invalid_ids=int(((actual_i != NO_INDEX) & ((actual_i < start) | (actual_i >= start + width))).sum()),
                samples=samples)


def synthetic_candidates(rows, parts, world):
    rng = np.random.default_rng(456)
    shape = (world, rows, parts)
    vals = rng.uniform(-8, -1, shape).astype(np.float32)
    ids = rng.choice(np.array([0, 1, 1023, 1024, 65535, 129279], np.int32), shape)
    for row, token in enumerate([0, 1, 1023, 1024]):
        vals[0, row, 0], ids[0, row, 0] = 20, token
    vals[0, 4, 0], ids[0, 4, 0] = 30, 101
    if parts * world > 1:
        vals[-1, 4, -1], ids[-1, 4, -1] = 30, 7
    vals[0, 5, 0], ids[0, 5, 0] = np.nan, 9
    if parts * world > 1:
        vals[-1, 5, -1], ids[-1, 5, -1] = np.nan, 3
    ids[:, 6, :], vals[:, 6, :] = NO_INDEX, -np.inf
    # -0 is the finish kernel's documented transport sentinel for token ID zero.
    ids[0, 7, 0] = np.array(0x80000000, dtype=np.uint32).view(np.int32).item()
    vals[0, 7, 0] = 40
    packed = np.empty((*shape, 2), np.float32)
    packed[..., 0] = vals
    packed.view(np.int32)[..., 1] = ids
    return packed.reshape(world * rows, parts * 2)


def host_self_test():
    # Random123 Philox4x32-10 published zero-counter/zero-key vector.
    w = philox_numpy(0, *[np.array(0, np.uint64) for _ in range(4)])
    assert [int(x) for x in w] == [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
    checks = 0
    for parts in [1, 3, 127]:
        for world in [1, 2, 3]:
            p = synthetic_candidates(8, parts, world)
            expected = [0, 1, 1023, 1024, 7 if parts * world > 1 else 101,
                        3 if parts * world > 1 else 9, NO_INDEX, 0]
            assert finish_reference(p, 8, parts, world).tolist() == expected
            v, ids, _ = expected_partial(np.arange(2 * 2051, dtype=np.float32).reshape(2, 2051), 2049, 1024)
            assert ids[0].tolist() == [1023, 2047, 2048]
            checks += 1
    rng = np.random.default_rng(123)
    for _ in range(300):
        rows, parts, world = 4, int(rng.integers(1, 160)), int(rng.integers(1, 5))
        vals = rng.normal(size=(world, rows, parts)).astype(np.float32)
        ids = rng.integers(0, 130000, size=vals.shape, dtype=np.int32)
        packed = np.empty((*vals.shape, 2), np.float32)
        packed[..., 0], packed.view(np.int32)[..., 1] = vals, ids
        gold = []
        for r in range(rows):
            candidates = [(float(vals[w, r, p]), int(ids[w, r, p])) for w in range(world) for p in range(parts)]
            mx = max(x[0] for x in candidates)
            gold.append(min(x[1] for x in candidates if x[0] == mx))
        assert finish_reference(packed, rows, parts, world).tolist() == gold
        checks += 1
    print('HOST_SELF_TEST PASS:', checks, 'candidate cases + Philox known vector', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', help='SGLang checkout root; imports its python/ tree')
    parser.add_argument('--device', type=int, default=0, help='logical visible NPU index')
    parser.add_argument('--suite', choices=['finish', 'eager', 'graph', 'all'], default='all')
    parser.add_argument('--full', action='store_true', help='more dtypes and random inputs')
    parser.add_argument('--input', help='own trusted .pt input/dump; no gather callbacks')
    parser.add_argument('--report', default='markov_stage_report.json')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--score-atol', type=float, default=2e-4)
    parser.add_argument('--dump-limit', type=int, default=3)
    parser.add_argument('--host-self-test', action='store_true', help='no torch/NPU needed')
    args = parser.parse_args()
    if args.host_self_test:
        host_self_test()
        return 0
    if args.repo:
        sys.path.insert(0, str(Path(args.repo).resolve() / 'python'))
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(version=1, args=vars(args), cases=[], limitations=[
        'Synthetic inputs do not prove end-to-end model correctness.',
        'Finish WORLD>1 tests only rank-major memory layout, not real collectives.',
        'No production source is modified; execute()/legacy dispatch is bypassed.',
        'Near ties can differ because log/log1p implementations round differently.',
    ])
    dump_count = 0
    try:
        import torch
        import torch_npu
        import triton
        torch.npu.set_device(args.device)
        device = torch.device('npu', args.device)
        mod = importlib.import_module('sglang.kernels.ops.speculative.dspark.markov_sample')
        required = ['MarkovSampleStep', '_launch_partial', '_markov_sample_finish_kernel', '_BLOCK']
        for name in required:
            if not hasattr(mod, name):
                raise RuntimeError('Checkout API changed; missing: ' + name)
        block = int(getattr(mod._BLOCK, 'value', mod._BLOCK))
        source = Path(inspect.getfile(mod)).resolve()
        report['environment'] = dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
            triton=triton.__version__, module=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            device=str(device), block=block, libdevice=str(getattr(mod.libdevice, '__file__', None)))
        print(json.dumps(report['environment'], indent=2), flush=True)

        def cpu_np(t):
            return t.detach().cpu().numpy().copy()

        def finish(packed, rows, parts, world=1):
            out = torch.empty(rows, dtype=torch.int64, device=device)
            mod._markov_sample_finish_kernel[(rows,)](packed, out, rows, parts, world,
                triton.next_power_of_2(parts * world), num_warps=4)
            return out

        def record(name, fn):
            try:
                result = fn()
            except Exception as exc:
                result = dict(status='ERROR', exception=repr(exc), traceback=traceback.format_exc())
            result['name'] = name
            report['cases'].append(result)
            report_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding='utf-8')
            print(result['status'], name, json.dumps({k: v for k, v in result.items() if k not in ['name', 'status', 'traceback']}, allow_nan=True), flush=True)
            return result

        def finish_case(parts, world):
            original = synthetic_candidates(8, parts, world)
            p = torch.from_numpy(original.copy()).to(device)
            torch.npu.synchronize()
            received = cpu_np(p)
            bit_changed = int((original.view(np.uint32) != received.view(np.uint32)).sum())
            # Score NaNs may canonicalize during transfer. Track ID lane separately.
            id_changed = int((original.reshape(world, 8, parts, 2).view(np.uint32)[..., 1] !=
                              received.reshape(world, 8, parts, 2).view(np.uint32)[..., 1]).sum())
            before = finish_reference(original, 8, parts, world)
            expected = finish_reference(received, 8, parts, world)
            actual = []
            for _ in range(max(1, args.repeat)):
                result = finish(p, 8, parts, world)
                torch.npu.synchronize()
                actual.append(cpu_np(result))
            same = all(np.array_equal(x, expected) for x in actual)
            status = 'INPUT_ID_BITS_CHANGED' if id_changed else ('PASS' if same else 'MISMATCH')
            return dict(status=status, input_bits_changed=bit_changed, input_id_bits_changed=id_changed,
                expected_before_transfer=before.tolist(), expected_from_received=expected.tolist(),
                actual=[x.tolist() for x in actual], finish_matches_received_candidates=same,
                repeat_stable=all(np.array_equal(x, actual[0]) for x in actual))

        def make_case(rows, padded, width, dtype_name, mode, seed=123):
            g = torch.Generator(device='cpu').manual_seed(seed)
            dtype = getattr(torch, dtype_name)
            base = torch.randn(rows, padded, generator=g, device='cpu').mul_(0.3).to(dtype)
            bias = torch.randn(rows, width, generator=g, device='cpu').mul_(0.2).to(dtype)
            temps = torch.ones(rows, dtype=torch.float32, device='cpu')
            if mode == 'greedy':
                temps.zero_()
                ids = [0, 1, 1023, 1024, 65535, width - 1, 2, 4095]
                for row in range(rows):
                    base[row, ids[row % len(ids)] % width] = 50
            elif mode == 'mixed':
                temps[::2] = 0
                temps[1::4] = 0.3
            return dict(base_logits=base, bias=bias, temperatures=temps, key=12345,
                        offset=torch.tensor([7], dtype=torch.int64, device='cpu'), step=2,
                        step_dtype=dtype, vocab_start=0,
                        has_sampled_rows=False if mode == 'greedy' else None)

        def host_data(kw):
            base, bias = kw['base_logits'].cpu(), kw['bias'].cpu()
            value = base.float().clone()
            value[:, :bias.shape[1]] += bias.float()
            value = value.to(kw['step_dtype']).float()
            keys = score_keys(cpu_np(value), cpu_np(kw['temperatures'].cpu()), key=kw['key'],
                offset=int(kw['offset'].cpu().item()), step=kw['step'], start=kw.get('vocab_start', 0),
                sampled=kw.get('has_sampled_rows') is not False)
            return value, keys

        def to_device(kw):
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kw.items()}

        def launch(kw, store):
            return mod._launch_partial(kw['base_logits'], kw['bias'], width=kw['bias'].shape[1],
                store=store, temperatures=kw['temperatures'], key=kw['key'], offset=kw['offset'],
                step=kw['step'], step_dtype=kw['step_dtype'], has_sampled_rows=kw.get('has_sampled_rows'),
                vocab_start=kw.get('vocab_start', 0), select=True)

        def dump_failure(name, kw, partial, result):
            nonlocal dump_count
            if dump_count >= args.dump_limit:
                return
            dump_count += 1
            path = report_path.with_name(report_path.stem + '_failure_' + str(dump_count) + '.pt')
            inp = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in kw.items()}
            inp['step_dtype'] = str(inp['step_dtype']).replace('torch.', '')
            torch.save(dict(name=name, inputs=inp, partial=partial.detach().cpu(), result=result), path)
            result['dump'] = str(path)

        def eager_case(name, kw_cpu):
            rows, padded = kw_cpu['base_logits'].shape
            width = kw_cpu['bias'].shape[1]
            parts = triton.cdiv(padded, block)
            value, keys = host_data(kw_cpu)
            kw = to_device(kw_cpu)
            store = torch.empty((rows, padded), dtype=kw['step_dtype'], device=device)
            # Synchronizing each stage deliberately removes producer/consumer races.
            p = launch(kw, store)
            torch.npu.synchronize()
            packed = cpu_np(p)
            details = compare_partial(packed, keys, width, block, kw.get('vocab_start', 0), args.score_atol)
            stored = store.detach().cpu().float()
            logits_equal = bool(torch.equal(stored, value))
            finish_expected = finish_reference(packed, rows, parts)
            finished = finish(p, rows, parts)
            torch.npu.synchronize()
            actual_finish = cpu_np(finished)
            cpu_logits = torch.empty((rows, padded), dtype=kw_cpu['step_dtype'], device='cpu')
            cpu_tokens = mod.MarkovSampleStep.torch(**kw_cpu, logits_out=cpu_logits).detach().cpu()
            torch_logits = torch.empty_like(store)
            torch_tokens = mod.MarkovSampleStep.torch(**kw, logits_out=torch_logits)
            torch.npu.synchronize()
            t_np = cpu_np(torch_tokens)
            tri_outputs = []
            tri_logits_equal = True
            for _ in range(max(1, args.repeat)):
                tri_logits = torch.empty_like(store)
                tri = mod.MarkovSampleStep.triton(**kw, logits_out=tri_logits)
                torch.npu.synchronize()
                tri_outputs.append(cpu_np(tri))
                tri_logits_equal &= bool(torch.equal(tri_logits.cpu().float(), value))
            gold = cpu_np(cpu_tokens)
            finish_ok = bool(np.array_equal(actual_finish, finish_expected))
            partial_ok = details['id_mismatch'] == details['score_mismatch'] == 0
            full_ok = all(np.array_equal(x, gold) for x in tri_outputs)
            torch_ok = bool(np.array_equal(t_np, gold))
            stable = all(np.array_equal(x, tri_outputs[0]) for x in tri_outputs)
            status = 'PASS' if all([partial_ok, logits_equal, finish_ok, full_ok, torch_ok, tri_logits_equal, stable]) else 'MISMATCH'
            if not finish_ok:
                location = 'FINISH: same received candidates, different reduction result'
            elif not logits_equal:
                location = 'PARTIAL_ADD_ROUND_STORE: stored logits differ'
            elif details['invalid_ids'] or details['strong_id_mismatch']:
                location = 'PARTIAL_SELECT_OR_ID_PACK: finish agrees with the candidates it received'
            elif not torch_ok:
                location = 'TORCH_NPU_REFERENCE_DIFFERS_FROM_CPU: do not blame finish alone'
            elif not full_ok or not stable:
                location = 'FULL_CALL_OR_REPEAT: staged path must be compared with the complete call'
            elif not partial_ok:
                location = 'PARTIAL_SCORE_OR_NEAR_TIE: inspect score gaps before assigning root cause'
            else:
                location = 'No mismatch in this case'
            result = dict(status=status, shape=[rows, padded], width=width, dtype=str(kw['step_dtype']),
                partial=details, partial_logits_equal=logits_equal,
                finish_matches_received_candidates=finish_ok, full_logits_equal=tri_logits_equal,
                torch_npu_matches_torch_cpu=torch_ok, full_triton_matches_torch_cpu=full_ok,
                repeat_stable=stable, torch_cpu_tokens=gold.tolist(), torch_npu_tokens=t_np.tolist(),
                staged_finish_tokens=actual_finish.tolist(), full_triton_tokens=[x.tolist() for x in tri_outputs],
                diagnosis=location)
            if status != 'PASS':
                dump_failure(name, kw_cpu, p, result)
            return result

        def graph_case(stage):
            kw_cpu = make_case(8, 9000, 8990, 'float32', 'mixed')
            kw = to_device(kw_cpu)
            rows, padded = kw['base_logits'].shape
            parts = triton.cdiv(padded, block)
            store = torch.empty((rows, padded), dtype=kw['step_dtype'], device=device)

            def call():
                if stage == 'staged':
                    p = launch(kw, store)
                    return p, finish(p, rows, parts)
                return None, mod.MarkovSampleStep.triton(**kw, logits_out=store)

            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                for _ in range(3):
                    call()
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                p, out = call()
            replays = []
            for off in [7, 8, 123, 7]:
                kw['offset'].fill_(off)
                torch.npu.synchronize()
                graph.replay()
                torch.npu.synchronize()
                ref_kw = dict(kw_cpu, offset=torch.tensor([off], dtype=torch.int64, device='cpu'))
                _, keys = host_data(ref_kw)
                expected = cpu_np(mod.MarkovSampleStep.torch(**ref_kw))
                actual = cpu_np(out)
                entry = dict(offset=off, tokens=actual.tolist(), expected=expected.tolist(),
                             tokens_equal=bool(np.array_equal(actual, expected)))
                if p is not None:
                    pc = cpu_np(p)
                    entry['partial'] = compare_partial(pc, keys, 8990, block, 0, args.score_atol)
                    entry['finish_matches_received_candidates'] = bool(np.array_equal(actual, finish_reference(pc, rows, parts)))
                replays.append(entry)
            passed = all(x['tokens_equal'] and x.get('finish_matches_received_candidates', True) and
                         x.get('partial', {}).get('strong_id_mismatch', 0) == 0 for x in replays)
            return dict(status='PASS' if passed else 'MISMATCH', replays=replays)

        if args.suite in ['finish', 'all']:
            for parts in [1, 3, 127]:
                for world in [1, 2, 3]:
                    record('finish_parts%d_world%d' % (parts, world), lambda p=parts, w=world: finish_case(p, w))
        if args.suite in ['eager', 'all']:
            cases = [
                ('greedy_fp32', make_case(8, 2048, 2048, 'float32', 'greedy')),
                ('greedy_vocab129280_bf16', make_case(8, 129280, 129280, 'bfloat16', 'greedy')),
                ('sampled_fp32', make_case(8, 9000, 8990, 'float32', 'sampled')),
                ('mixed_fp32', make_case(8, 9000, 8990, 'float32', 'mixed')),
            ]
            if args.full:
                for dt in ['bfloat16', 'float16']:
                    cases.append(('mixed_' + dt, make_case(8, 9000, 8990, dt, 'mixed')))
                cases.append(('sampled_vocab129280_bf16', make_case(16, 129280, 129280, 'bfloat16', 'sampled')))
            if args.input:
                obj = torch.load(args.input, map_location='cpu', weights_only=True)
                kw = obj.get('inputs', obj)
                kw = dict(kw)
                if isinstance(kw.get('step_dtype'), str):
                    kw['step_dtype'] = getattr(torch, kw['step_dtype'].replace('torch.', ''))
                keep = ['base_logits', 'bias', 'temperatures', 'key', 'offset', 'step', 'step_dtype', 'vocab_start', 'has_sampled_rows']
                kw = {k: v for k, v in kw.items() if k in keep}
                cases.append(('provided_input', kw))
            for name, kw in cases:
                record(name, lambda n=name, k=kw: eager_case(n, k))
        if args.suite in ['graph', 'all']:
            for stage in ['staged', 'full']:
                record('graph_' + stage, lambda s=stage: graph_case(s))
    except Exception as exc:
        report['fatal'] = dict(exception=repr(exc), traceback=traceback.format_exc())
        print(report['fatal']['traceback'], file=sys.stderr)
    finally:
        report['summary'] = dict(Counter(x['status'] for x in report['cases']))
        report_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding='utf-8')
        print('REPORT:', report_path, 'SUMMARY:', report['summary'], flush=True)
    return 1 if 'fatal' in report or any(x['status'] != 'PASS' for x in report['cases']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
