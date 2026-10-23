#!/usr/bin/env python3
"""Locate the first bad value in Markov random sampling, without editing SGLang.

Keep markov_stage_probe.py beside this file. Loads the local _select_block AST
and extracts its real Philox/conversion/math expressions into a diagnostic kernel.
Also tests the uint32->int64 & 0xffffffff expression from upstream kernel PR #871.
Instrumentation can change compilation; a passing probe cannot invalidate a failing
original kernel. This is a diagnostic comparison, not a production patch.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import sys
import textwrap
import traceback
from pathlib import Path

import numpy as np
import markov_stage_probe as reference

SOURCE_MODULE = 'sglang.kernels.ops.speculative.dspark.markov_sample'


def assignment_to(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


def extract_source(source):
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_select_block')
    branches = [n for n in ast.walk(fn) if isinstance(n, ast.If) and
                any(assignment_to(x, 'uniform') for x in n.body) and
                any(assignment_to(x, 'exp_noise') for x in n.body)]
    if len(branches) != 1:
        raise RuntimeError('Cannot identify the random branch of _select_block; provide its current source.')
    body = branches[0].body
    split = next(i for i, n in enumerate(body) if assignment_to(n, 'bits') and 'interleave' in ast.unparse(n))
    tail = body[split + 1:]
    if not any(assignment_to(n, 'key') for n in tail):
        raise RuntimeError('Cannot find sampled key expression')
    def render(nodes):
        return ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    return render(body[:split + 1]), render(tail)


def kernel_source(prefix, current_tail):
    upstream_tail = '''bits = bits.to(tl.int64) & 0xFFFFFFFF
uniform = (bits.to(tl.float32) + 0.5) * _UNIFORM_SCALE
exp_noise = -libdevice.log1p(-uniform)
key = value / temperature - tl.log(exp_noise)'''
    head = '''import triton
import triton.language as tl
from sglang.kernels.ops.speculative.dspark.markov_sample import libdevice, _UNIFORM_SCALE

@triton.jit
def noise_probe(VALUE, T, OFFSET, FIXED_BITS, RAW_OUT, U_OUT, E_OUT, K_OUT,
                PADDED: tl.constexpr, WIDTH: tl.constexpr, KEY: tl.constexpr,
                STEP: tl.constexpr, VOCAB_START: tl.constexpr, BLOCK: tl.constexpr,
                PHILOX_ROUNDS: tl.constexpr, FIXED: tl.constexpr, MODE: tl.constexpr):
    row, part = tl.program_id(0), tl.program_id(1)
    i = part * BLOCK + tl.arange(0, BLOCK)
    pos = row * PADDED + i
    valid = i < WIDTH
    memory = i < PADDED
    value = tl.load(VALUE + pos, memory, 0).to(tl.float32)
    temperature = tl.load(T + row)
    # Greedy rows are ignored by the host; keep this diagnostic division defined.
    temperature = tl.where(temperature == 0.0, 1.0, temperature)
    if FIXED:
        bits = tl.load(FIXED_BITS + pos, memory, 0).to(tl.uint32, bitcast=True)
    else:
'''
    body = head + textwrap.indent(prefix, '        ') + '\n    raw_bits = bits\n    if MODE == 0:\n'
    body += textwrap.indent(current_tail, '        ')
    body += '\n    else:\n' + textwrap.indent(upstream_tail, '        ')
    body += '''
    tl.store(RAW_OUT + pos, raw_bits.to(tl.int32, bitcast=True), memory)
    tl.store(U_OUT + pos, uniform, memory)
    tl.store(E_OUT + pos, exp_noise, memory)
    tl.store(K_OUT + pos, key, memory)
'''
    ast.parse(body)
    return body


def compare_array(actual, expected, active, atol, rtol):
    good = np.isclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)
    bad = active & ~good
    finite = active & np.isfinite(actual) & np.isfinite(expected)
    finite_error = np.abs(actual[finite] - expected[finite])
    examples = [dict(row=int(r), token_column=int(c), actual=float(actual[r, c]),
                     expected=float(expected[r, c])) for r, c in np.argwhere(bad)[:10]]
    return dict(mismatches=int(bad.sum()), active=int(active.sum()),
                actual_nan=int((active & np.isnan(actual)).sum()),
                expected_nan=int((active & np.isnan(expected)).sum()),
                max_finite_abs_error=float(finite_error.max()) if finite_error.size else None,
                examples=examples)


def self_test(source_path):
    reference.host_self_test()
    src = Path(source_path).read_text()
    prefix, tail = extract_source(src)
    out = kernel_source(prefix, tail)
    assert 'tl.philox' in out and 'MODE == 0' in out and 'bits.to(tl.int64) & 0xFFFFFFFF' in out
    # Verify extraction also handles the direct cast variant without a high/low split.
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_select_block')
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and any(assignment_to(x, 'uniform') for x in node.body):
            j = next(i for i, x in enumerate(node.body) if assignment_to(x, 'bits') and 'interleave' in ast.unparse(x))
            node.body[j + 1:] = ast.parse('uniform = (bits.to(tl.float32) + 0.5) * _UNIFORM_SCALE\nexp_noise = -libdevice.log1p(-uniform)\nkey = value / temperature - tl.log(exp_noise)').body
    p, t = extract_source(ast.unparse(tree))
    ast.parse(kernel_source(p, t))
    a = np.array([0.1, np.nan, np.inf, -np.inf], np.float32)[None, :]
    assert compare_array(a, a, np.ones(a.shape, bool), 1e-5, 1e-5)['mismatches'] == 0
    print('NOISE_HOST_SELF_TEST PASS: AST extraction, both source variants, comparison helper')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', required=False)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--input', help='failure_1.pt produced by markov_stage_probe (own trusted file)')
    ap.add_argument('--report', default='markov_noise_report.json')
    ap.add_argument('--save-arrays', action='store_true')
    ap.add_argument('--host-self-test', action='store_true')
    ap.add_argument('--source-file', help='local markov source, used only by host self-test')
    args = ap.parse_args()
    if args.host_self_test:
        if not args.source_file:
            ap.error('--host-self-test requires --source-file')
        self_test(args.source_file)
        return 0
    if args.repo:
        sys.path.insert(0, str(Path(args.repo).resolve() / 'python'))
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(version=1, args=vars(args), tests=[], limitations=[
        'Diagnostic stores and a separate kernel can change compiler behavior.',
        'upstream_871 tests only its conversion expression, not the entire upstream implementation.',
        'Greedy rows are excluded; graph replay and real multi-rank collectives are not tested here.',
    ])
    try:
        import torch
        import torch_npu
        import triton
        torch.npu.set_device(args.device)
        dev = torch.device('npu', args.device)
        mod = importlib.import_module(SOURCE_MODULE)
        source_path = Path(mod.__file__).resolve()
        source = source_path.read_text()
        prefix, tail = extract_source(source)
        generated = report_path.with_name(report_path.stem + '_generated.py')
        generated.write_text(kernel_source(prefix, tail), encoding='utf-8')
        spec = importlib.util.spec_from_file_location('markov_noise_generated_' + str(abs(hash(str(generated)))), generated)
        generated_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = generated_mod
        spec.loader.exec_module(generated_mod)
        report['environment'] = dict(module=str(source_path), sha256=hashlib.sha256(source.encode()).hexdigest(),
            torch=torch.__version__, torch_npu=torch_npu.__version__, triton=triton.__version__,
            libdevice=str(mod.libdevice.__file__), generated_kernel=str(generated))
        report['actual_source_expressions'] = dict(philox=prefix, conversion_and_math=tail)
        if args.input:
            obj = torch.load(args.input, map_location='cpu', weights_only=True)
            kw = dict(obj.get('inputs', obj))
            if isinstance(kw['step_dtype'], str):
                kw['step_dtype'] = getattr(torch, kw['step_dtype'].replace('torch.', ''))
            kw = {k: v for k, v in kw.items() if k in ['base_logits', 'bias', 'temperatures', 'key', 'offset', 'step', 'step_dtype', 'vocab_start', 'has_sampled_rows']}
        else:
            g = torch.Generator(device='cpu').manual_seed(123)
            kw = dict(base_logits=torch.randn(8, 9000, generator=g, device='cpu').mul_(0.3),
                      bias=torch.randn(8, 8990, generator=g, device='cpu').mul_(0.2),
                      temperatures=torch.ones(8, device='cpu'), key=12345,
                      offset=torch.tensor([7], dtype=torch.int64, device='cpu'), step=2,
                      step_dtype=torch.float32, vocab_start=0, has_sampled_rows=None)
        rows, padded = kw['base_logits'].shape
        width = kw['bias'].shape[1]
        value = kw['base_logits'].float().clone()
        value[:, :width] += kw['bias'].float()
        value = value.to(kw['step_dtype']).float()
        block = int(getattr(mod._BLOCK, 'value', mod._BLOCK))

        def arr(t):
            return t.detach().cpu().numpy().copy()

        # Confirm that the original uninstrumented partial still exhibits the issue.
        try:
            gpu_kw = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in kw.items()}
            p = mod._launch_partial(gpu_kw['base_logits'], gpu_kw['bias'], width=width,
                store=None, temperatures=gpu_kw['temperatures'], key=kw['key'], offset=gpu_kw['offset'],
                step=kw['step'], step_dtype=kw['step_dtype'], has_sampled_rows=kw.get('has_sampled_rows'),
                vocab_start=kw.get('vocab_start', 0), select=True)
            torch.npu.synchronize()
            pv, pi = reference.decode_candidates(arr(p), rows, triton.cdiv(padded, block))
            report['original_partial'] = dict(nan_candidates=int(np.isnan(pv).sum()),
                total_candidates=int(pv.size), first_row_ids=pi[0].tolist())
        except Exception as exc:
            report['original_partial'] = dict(exception=repr(exc), traceback=traceback.format_exc())

        def test(name, mode, fixed=False):
            if fixed:
                # High-bit values expose signed widening/cast behavior immediately.
                pattern = np.array([0, 1, 0xFFFF, 0x10000, 0x7FFFFFFF, 0x80000000,
                                    0x80000001, 0xC0000000, 0xFFFFFF00, 0xFFFFFFFF], np.uint32)
                bits_ref = np.resize(pattern, (1, block)).copy()
                values = torch.zeros((1, block), dtype=torch.float32, device='cpu')
                temps = torch.ones(1, dtype=torch.float32, device='cpu')
                w, start = block, 0
            else:
                values, temps, w, start = value, kw['temperatures'].cpu(), width, kw.get('vocab_start', 0)
                rr, pp = values.shape
                words = reference.philox_numpy(kw['key'],
                    np.arange((pp + 3) // 4, dtype=np.uint64)[None, :] + np.uint64(start // 4),
                    np.arange(rr, dtype=np.uint64)[:, None], np.array(kw['step'], np.uint64),
                    np.array(int(kw['offset'].cpu().item()) & 0xFFFFFFFF, np.uint64))
                bits_ref = np.stack(words, axis=-1).reshape(rr, -1)[:, :pp].astype(np.uint32)
            rr, pp = values.shape
            dval = values.to(dev)
            dtemps = temps.to(dev)
            offset = kw['offset'].to(dev)
            fixed_tensor = torch.from_numpy(bits_ref.view(np.int32).copy()).to(dev)
            raw = torch.empty((rr, pp), dtype=torch.int32, device=dev)
            u, e, k = [torch.empty((rr, pp), dtype=torch.float32, device=dev) for _ in range(3)]
            generated_mod.noise_probe[(rr, triton.cdiv(pp, block))](dval, dtemps, offset, fixed_tensor,
                raw, u, e, k, pp, w, kw['key'], kw['step'], start, block, 10, fixed, mode, num_warps=4)
            torch.npu.synchronize()
            actual_bits = arr(raw).view(np.uint32)
            actual_u, actual_e, actual_k = arr(u), arr(e), arr(k)
            active = np.broadcast_to(np.arange(pp)[None, :] < w, (rr, pp)).copy()
            active &= arr(temps)[:, None] != 0
            # Stage-by-stage reference uses the OBSERVED preceding stage.
            expected_u = (actual_bits.astype(np.float32) + np.float32(0.5)) * np.float32(2 ** -32)
            u_cpu = torch.from_numpy(actual_u.copy())
            expected_e = arr(-torch.log1p(-u_cpu))
            safe = torch.where(temps != 0, temps, torch.ones_like(temps))[:, None]
            expected_k = arr(values / safe - torch.log(torch.from_numpy(actual_e.copy())))
            bit_bad = active & (actual_bits != bits_ref)
            u_check = compare_array(actual_u, expected_u, active, 2e-7, 2e-6)
            e_check = compare_array(actual_e, expected_e, active, 2e-5, 2e-5)
            k_check = compare_array(actual_k, expected_k, active, 2e-4, 2e-5)
            earliest = ('PHILOX_BITS' if bit_bad.any() else 'UINT32_TO_UNIFORM' if u_check['mismatches']
                        else 'LOG1P_TO_EXP_NOISE' if e_check['mismatches']
                        else 'LOG_OR_SCORE' if k_check['mismatches'] else 'NONE_OBSERVED')
            invalid_u = active & ((actual_u <= 0) | (actual_u > 1) | ~np.isfinite(actual_u))
            examples = []
            bad = active & (bit_bad | invalid_u | np.isnan(actual_e) | (actual_e < 0) | np.isnan(actual_k))
            if not bad.any():
                bad = active
            for r, c in np.argwhere(bad)[:12]:
                examples.append(dict(row=int(r), token_column=int(c), bits_hex=hex(int(actual_bits[r, c])),
                    bits_expected_hex=hex(int(bits_ref[r, c])), uniform=float(actual_u[r, c]),
                    uniform_expected=float(expected_u[r, c]), exp_noise=float(actual_e[r, c]),
                    exp_noise_expected=float(expected_e[r, c]), score=float(actual_k[r, c]),
                    score_expected_from_actual_exp_noise=float(expected_k[r, c])))
            result = dict(name=name, status='PASS' if earliest == 'NONE_OBSERVED' and not invalid_u.any() and not (active & np.isnan(actual_k)).any() else 'MISMATCH',
                first_bad_stage=earliest, active_tokens=int(active.sum()), philox_bit_mismatch=int(bit_bad.sum()),
                uniform=u_check, exp_noise=e_check, score=k_check,
                invalid_uniform=int(invalid_u.sum()), negative_exp_noise=int((active & (actual_e < 0)).sum()),
                nan_score=int((active & np.isnan(actual_k)).sum()), examples=examples)
            if args.save_arrays:
                out = report_path.with_name(report_path.stem + '_' + name + '.npz')
                np.savez_compressed(out, bits=actual_bits, expected_bits=bits_ref, uniform=actual_u,
                    exp_noise=actual_e, score=actual_k, expected_uniform=expected_u,
                    expected_exp_noise_from_actual_uniform=expected_e, expected_score_from_actual_exp_noise=expected_k)
                result['arrays'] = str(out)
            return result

        for fixed in [True, False]:
            for mode, label in [(0, 'current'), (1, 'upstream_871_conversion')]:
                name = ('fixed_bits_' if fixed else 'philox_') + label
                try:
                    result = test(name, mode, fixed)
                except Exception as exc:
                    result = dict(name=name, status='ERROR', exception=repr(exc), traceback=traceback.format_exc())
                report['tests'].append(result)
                report_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding='utf-8')
                print(result['status'], name, result.get('first_bad_stage', result.get('exception')), flush=True)
        print('Original partial:', report['original_partial'], flush=True)
    except Exception as exc:
        report['fatal'] = dict(exception=repr(exc), traceback=traceback.format_exc())
        print(report['fatal']['traceback'], file=sys.stderr)
    finally:
        report_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding='utf-8')
        print('REPORT:', report_path, flush=True)
    return int('fatal' in report or any(x['status'] != 'PASS' for x in report['tests']))


if __name__ == '__main__':
    raise SystemExit(main())
