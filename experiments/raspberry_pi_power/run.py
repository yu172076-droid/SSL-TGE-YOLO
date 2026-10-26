"""Pi PP1 SHORT300 CPU video-cost queue, using separately pinned target release.

SHORT300 or FULL1475, complete blocks >=60 s, independent processes, same boot,
no external power claim, no automatic retries/reboots/settings changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid

DRAFT = Path(__file__).resolve().parent


def child_environment():
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME'): env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    return env


def birth_ticks(pid):
    if platform.system()!='Linux': return None
    try: return int(Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19])
    except (OSError,ValueError,IndexError): return None


def commit_state(out,report):
    temp=out/'queue_state.tmp'
    temp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,out/'queue_state.json')


def execute_child(command,log,timeout,on_start):
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=child_environment())
    metadata=dict(actual_PID=process.pid,process_start_ticks=birth_ticks(process.pid),timed_out=False)
    try:
        on_start(metadata)
        try: metadata['actual_exit']=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            metadata['timed_out']=True; process.kill(); metadata['actual_exit']=process.wait()
    except BaseException:
        if process.poll() is None: process.kill(); process.wait()
        raise
    return metadata


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bootstrap(args):
    need(sha(DRAFT / 'core_manifest.json') == args.draft_sha, 'Core manifest SHA differs')
    for row in read(DRAFT / 'core_manifest.json')['files']:
        path = DRAFT / row['path']
        need(path.name == row['path'] and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Draft payload differs')
    bundle = Path(args.model_bundle).resolve()
    sys.path.insert(0, str(bundle))
    from integrity import verify
    _, models, dataset = verify(bundle, args.bundle_sha)
    link = read(DRAFT / 'bundle_link.json')
    need(link['model_bundle_manifest_sha256'] == args.bundle_sha, 'Draft linked to a different model package')
    return bundle, models, dataset


def qualified(args, bundle, models):
    need(sha(args.release) == args.release_sha, 'Release SHA differs')
    q = read(args.release)
    need(q['schema'] == 'pi_CPU_formal_release_v1' and q['released'] is True, 'Formal draft has no released target configuration')
    need(q['bundle_manifest_sha256'] == args.bundle_sha and q['draft_manifest_sha256'] == args.draft_sha, 'Release package binding differs')
    need(q['platform_system'] == platform.system() and q['platform_machine'] == platform.machine(), 'Release platform differs')
    need(q['profile'] in ('SHORT300', 'FULL1475') and q['minimum_measurement_s'] >= 60, 'Whole-block video profile not qualified')
    if q['scope']=='Pi_formal': need(q['profile']=='SHORT300','This PP1 release freezes SHORT300')
    allowed = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    need(type(q['threads']) is int and 1 <= q['threads'] <= min(4, allowed), 'CPU allowance differs')
    need(q['scope'] in ('Pi_formal', 'local_CPU_validation'), 'Bad release scope')
    p1_path = Path(q['accepted_P1_parent_report'])
    need(sha(p1_path) == q['accepted_P1_parent_report_sha256'], 'Accepted P1 report differs')
    p1 = read(p1_path)
    need(p1['status'] == 'P1_queue_completed_review_required' and p1['mode'] == 'probe'
         and p1['bundle_manifest_sha256'] == args.bundle_sha and len(p1['records']) == 6
         and all(r['child_exit'] == 0 for r in p1['records']), 'P1 compatibility not completed')
    if q['scope'] == 'Pi_formal':
        need(platform.system() == 'Linux' and platform.machine() in ('aarch64', 'arm64')
             and q['resource_guard_required'] is True and p1['scope'] == 'Pi_P1', 'Pi formal requires target P1 and resource guarding')
        need(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == q['expected_boot_id'], 'Boot changed from release')
        need(sys.version_info[:3]==(3,13,5),'Python changed from accepted P1')
        need(sorted(os.sched_getaffinity(0))==[0,1,2,3],'CPU affinity changed from accepted P1')
        need(q['warmup_minimum_s'] >= 60 and q['idle_policy']['minimum_s'] >= 60
             and q['idle_policy']['tail_s'] >= 30 and q['loaded_policy']['tail_s'] >= 30, 'Stable tail/warmup minima weakened')
        mem = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:'))
        need(mem == q['MemTotal_bytes'] and q['runtime_policy']['min_available_bytes'] >= max(256 * 1024**2, int(.1 * mem)), 'Capacity release not qualified')
        need(q['idle_policy']['cpu_policy_version']=='PF2_tick_weighted_mean20_peak40_v1' and q['idle_policy']['cpu_aggregation']=='tick_weighted_tail_mean' and q['idle_policy']['max_idle_cpu_percent']==20 and q['idle_policy']['max_idle_cpu_peak_percent']==40,'PP1 retained-background idle policy changed')
    else:
        need(q['resource_guard_required'] is False and p1['scope'] == 'local_CPU_validation', 'Wrong local validation scope')
    if q['scope']=='Pi_formal':
        need(q['protocol_version']=='PP1_partial_PMIC_rails_v1','Wrong PP1 protocol')
        need(q['idle_policy']['minimum_s']==75 and q['idle_policy']['tail_s']==60,'PP1 baseline policy changed')
        pf2=Path(q['accepted_PF2_parent_report'])
        need(sha(pf2)==q['accepted_PF2_parent_report_sha256'],'Accepted PF2 parent changed')
        accepted=read(pf2)
        need(accepted['exitcode']==0 and len(accepted['records'])==18 and all(r['cost_exit']==0 for r in accepted['records']),'PF2 cost chain not accepted')
        need(sha(DRAFT/'accepted_PF2'/'main_acceptance.json')==q['accepted_PF2_main_acceptance_sha256'],'PF2 main acceptance SHA differs')
        proof=read(DRAFT/'accepted_PF2'/'main_acceptance.json')
        need(proof['accepted'] is True and proof['accuracy_groups']==12,'Main-window PF2 precision acceptance missing')
    return q


def artifact_report(out, report):
    from integrity import write_report
    report['artifacts'] = [dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]
    write_report(out, report)


def cost(args, q, bundle, model, dataset):
    from integrity import write_new
    out = Path(args.output).resolve()
    out.mkdir(parents=False, exist_ok=False)
    report = dict(schema='pi_CPU_cost_worker_v1', status='failed', exitcode=3, session=args.session,
        job=model['id'], slot=args.slot, replicate=args.replicate, scope=q['scope'], hardware_is_Pi=q['scope'] == 'Pi_formal',
        bundle_manifest_sha256=args.bundle_sha, draft_manifest_sha256=args.draft_sha, release_sha256=args.release_sha,
        onnx_sha256=model['onnx_sha256'], weight_sha256=model['weight_sha256'], PID=os.getpid(),
        process_started_ns=time.time_ns(), profile=q['profile'], conf=.25, iou=.65, max_det=300,
        power_measured=False, whole_board_power_measured=False, power_scope='PMIC_output_rails_only', power_W=None, J_per_frame=None, FPS_per_W=None, AP_evaluated=False,
        accuracy_gate_pending=False, FPS_measured=False, idle_cpu_policy_version=q.get('idle_policy',{}).get('cpu_policy_version'))
    report.update(PPID=os.getppid(),process_birth_ticks=birth_ticks(os.getpid()))
    guard, frames = None, None
    result = 3
    try:
        os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS=str(q['threads']))
        import numpy as np
        import cv2
        import onnxruntime as ort
        from postprocess import prepare, postprocess
        from diagnostic import session_for
        from guard import Guard
        versions = dict(numpy=np.__version__, opencv=cv2.__version__, onnxruntime=ort.__version__)
        need(versions == q['runtime_versions'], 'Runtime version differs')
        cv2.setNumThreads(1)
        report['environment'] = dict(versions=versions, platform=platform.platform(), threads=q['threads'],
            providers=['CPUExecutionProvider'], inter_threads=1, opencv_threads=1, allow_spinning=False,
            execution_mode='ORT_SEQUENTIAL', boot_id=q.get('expected_boot_id'))
        report['environment']['clock_ticks_per_second']=os.sysconf('SC_CLK_TCK') if hasattr(os,'sysconf') else None
        guard = Guard(out, q)
        idle_proof=guard.wait_ready('idle')
        baseline=dict(start_monotonic_ns=idle_proof['tail_start_ns'],end_monotonic_ns=idle_proof['tail_end_ns'])
        guard.set_phase('load')
        load_start = time.monotonic_ns()
        session, name = session_for(model, q, ort)
        report['model_load_ns'] = time.monotonic_ns() - load_start
        guard.sync()
        image_row = dataset['golden_image']
        row = next(r for r in dataset['images'] if r['split'] == image_row['split'] and r['id'] == image_row['id'])
        image = cv2.imdecode(np.fromfile(str(bundle / row['path']), np.uint8), cv2.IMREAD_COLOR)
        need(image is not None and image.shape == (row['height'], row['width'], 3), 'Golden image decode differs')
        tensor, ratio, pad = prepare(image, cv2)
        need(np.array_equal(tensor, np.load(str(bundle / dataset['golden_input']), allow_pickle=False)), 'Golden input differs')
        raw = session.run(None, {name: tensor})[0]
        postprocess(raw, model['backend'], image.shape, ratio, pad, conf=.25)
        need(np.allclose(raw, np.load(str(bundle / model['golden_raw']), allow_pickle=False), rtol=.001, atol=.001), 'Golden raw differs')
        report['golden_forward_calls'] = 1
        guard.sync()
        frames = (out / 'frames.jsonl').open('x', encoding='utf-8', buffering=1)
        records = []
        video = bundle / dataset['video']['path']
        need(sha(video) == dataset['video']['sha256'], 'Video input changed before run')

        def block(count, phase, number):
            capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
            try:
                need(capture.isOpened() and capture.getBackendName() == 'FFMPEG', 'Video FFMPEG unavailable')
                need(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 1475, 'Video frame header differs')
                if 'video_header' not in report:
                    report['video_header'] = dict(fps=capture.get(cv2.CAP_PROP_FPS), frames=1475, backend='FFMPEG')
                for index in range(1, count + 1):
                    guard.check()
                    t0 = time.monotonic_ns()
                    ok, image = capture.read()
                    t1 = time.monotonic_ns()
                    need(ok and image.shape == (2160, 3840, 3), 'Native 4K frame absent or scaled')
                    if phase == 'warmup' and number == 1 and index == 1:
                        guard.sync()  # First decoded frame resident, before first video forward.
                    tensor, ratio, pad = prepare(image, cv2)
                    t2 = time.monotonic_ns()
                    raw = session.run(None, {name: tensor})[0]
                    t3 = time.monotonic_ns()
                    det = postprocess(raw, model['backend'], image.shape, ratio, pad, conf=.25)
                    t4 = time.monotonic_ns()
                    rec = dict(phase=phase, block=number, frame_index=index, decode_ns=t1-t0,
                        processing_ns=t4-t1, forward_ns=t3-t2, detections=len(det),
                        detections_sha256=hashlib.sha256(det.tobytes()).hexdigest())
                    frames.write(json.dumps(rec, allow_nan=False) + '\n')
                    records.append(rec)
                    guard.check()
                    if phase == 'measurement' and (time.monotonic_ns() - guard.start_ns) / 1e9 > q['maximum_measurement_s']:
                        raise RuntimeError('Bounded measurement budget exhausted')
                if count == 1475:
                    ok, _ = capture.read()
                    need(not ok, 'Unexpected frame after 1475; full EOF not verified')
            finally:
                capture.release()
            frames.flush()

        guard.set_phase('warmup')
        warm_start = time.monotonic_ns()
        warm_blocks = 0
        while True:
            warm_blocks += 1
            block(30, 'warmup', warm_blocks)
            elapsed = (time.monotonic_ns() - warm_start) / 1e9
            need(elapsed <= q['maximum_warmup_s'], 'Bounded warmup exhausted')
            if elapsed >= q['warmup_minimum_s']:
                break
        report['warmup'] = dict(blocks=warm_blocks, frames=30 * warm_blocks, duration_s=elapsed,
                                 compatibility_forward_excluded=1, partial_blocks=True)
        guard.set_phase('loaded_quiet')
        frames.flush()
        start = guard.wait_ready('loaded', transition=True)
        count = 300 if q['profile'] == 'SHORT300' else 1475
        passes = 0
        while True:
            passes += 1
            block(count, 'measurement', passes)
            if (time.monotonic_ns() - start) / 1e9 >= q['minimum_measurement_s']:
                break
        frames.flush()
        frames.close()
        frames = None
        end = guard.finish_measurement()
        measured = [r for r in records if r['phase'] == 'measurement']
        need(len(measured) == count * passes and end > start, 'Window frame accounting differs')
        need(sha(video) == dataset['video']['sha256'] and sha(bundle / model['onnx']) == model['onnx_sha256'], 'Video/model changed after run')
        duration = (end - start) / 1e9
        report.update(status='video_cost_and_PMIC_rails_completed_review_required', exitcode=0, FPS_measured=True,
            measurement=dict(start_monotonic_ns=start, end_monotonic_ns=end, duration_s=duration,
                frames=len(measured), complete_blocks=passes, frames_per_block=count, window_FPS=len(measured)/duration,
                intentionally_partial=q['profile'] == 'SHORT300', full_EOF_verified=q['profile'] == 'FULL1475',
                includes_capture_open_decode_preprocess_forward_NMS_record_flush_close_and_endpoint_probe=True,
                includes_admission_commit_record=True,
                forward_mean_ms=float(np.mean([r['forward_ns'] for r in measured]) / 1e6),
                processing_median_ms=float(np.median([r['processing_ns'] for r in measured]) / 1e6)),
            all_forward_calls=1 + 30 * warm_blocks + len(measured), video_unchanged=True)
        report['PMIC']=guard.pmic.summary(baseline,start,end,len(measured))
        report['power_measured']=True
        result = 0
    except Exception as error:
        report.update(status='failed',exitcode=3,FPS_measured=False,power_measured=False)
        report['error'] = repr(error)
        (out / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print(traceback.format_exc(), flush=True)
    finally:
        if frames:
            frames.close()
        if guard:
            report['resource_guard'] = guard.close()
            if report['resource_guard']['failure']:
                report.update(status='failed', exitcode=3, FPS_measured=False, power_measured=False)
                result = 3
        report['process_finished_ns'] = time.time_ns()
        artifact_report(out, report)
    return result


def bind_child(path, session, job, args, expected):
    report = read(path / 'report.json')
    need((path / 'report.sha256').read_text().split()[0] == sha(path / 'report.json'), 'Child sidecar differs')
    need(report['exitcode'] == 0 and report['session'] == session and report['job'] == job, 'Child status/session differs')
    need(report['bundle_manifest_sha256'] == args.bundle_sha, 'Child bundle differs')
    need(all(report.get(key) == value for key, value in expected.items()), 'Child expected fields differ')
    for row in report['artifacts']:
        need(Path(row['path']).name == row['path'] and (path / row['path']).stat().st_size == row['bytes']
             and sha(path / row['path']) == row['sha256'], 'Child artifact differs')
    from pmic_reader import parse_adc,integrate,CHANNELS
    import math
    need(report['power_measured'] is True and report['whole_board_power_measured'] is False
         and report['power_scope']=='PMIC_output_rails_only','PMIC scope differs')
    raw=[json.loads(line) for line in (path/'pmic.jsonl').read_text(encoding='utf-8').splitlines()]
    need(len(raw)==report['resource_guard']['PMIC']['samples'] and raw and not report['resource_guard']['PMIC']['failure'],'PMIC raw query count/failure differs')
    for sample in raw:
        need(sample['valid'] is True and sample['actual_exit']==0 and sample['resource_boot_id']==report['environment']['boot_id'],'Invalid PMIC raw sample')
        need(sample['mid_monotonic_ns']==(sample['capture_start_monotonic_ns']+sample['capture_end_monotonic_ns'])//2,'PMIC midpoint differs')
        parsed=parse_adc(sample['stdout'])
        need(parsed['rails']==sample['rails'] and parsed['voltage_only_V']==sample['voltage_only_V'],'PMIC raw conversion differs')
    pmic=report['PMIC'];window=report['measurement']
    need(pmic['scope']=='PMIC_output_rails_only' and pmic['whole_board_power_measured'] is False and pmic['channel_sum_computed'] is False,'PMIC summary scope differs')
    need(pmic['baseline']['duration_s']>=60 and pmic['frames']==window['frames'],'PMIC baseline/frames differ')
    need(pmic['measurement']['start_monotonic_ns']==window['start_monotonic_ns'] and pmic['measurement']['end_monotonic_ns']==window['end_monotonic_ns'],'PMIC/throughput windows differ')
    for kind in ('baseline','measurement'):
        saved=pmic[kind]
        proof=integrate(raw,saved['start_monotonic_ns'],saved['end_monotonic_ns'])
        need(set(saved['rails'])==set(CHANNELS),'PMIC rail set differs')
        for rail in CHANNELS:
            for key in ('mean_W','energy_J'):
                need(math.isclose(saved['rails'][rail][key],proof['rails'][rail][key],rel_tol=1e-12,abs_tol=1e-12),'PMIC independent integral differs')
    for rail in CHANNELS:
        need(math.isclose(pmic['incremental_mean_W'][rail],pmic['measurement']['rails'][rail]['mean_W']-pmic['baseline']['rails'][rail]['mean_W'],rel_tol=1e-12,abs_tol=1e-12),'PMIC baseline subtraction differs')
    return sha(path / 'report.json')


def queue(args, q, bundle, models):
    from integrity import write_new, write_report
    out = Path(args.output).resolve()
    if out.exists():
        need(bool(args.launcher_session) and bool(args.launcher_identity_sha),'Existing output without launcher binding')
        need(sha(out/'job_identity.json')==args.launcher_identity_sha,'Launcher identity SHA differs')
        identity=read(out/'job_identity.json')
        need(identity['session']==args.launcher_session and identity['boot_id']==q['expected_boot_id'],'Launcher session/boot differs')
        need({p.name for p in out.iterdir()} <= {'job_identity.json','launch.json'},'Existing output already contains work; no resume')
    else:
        need(q['scope']=='local_CPU_validation','Target queue must use the released detached launcher')
        out.mkdir(parents=False,exist_ok=False)
    report = dict(schema='pi_CPU_formal_queue_v1', session=args.launcher_session or str(uuid.uuid4()), scope=q['scope'],
        profile=q['profile'], bundle_manifest_sha256=args.bundle_sha, draft_manifest_sha256=args.draft_sha,
        release_sha256=args.release_sha, status='incomplete', exitcode=3, records=[],
        AP_evaluated=False, accuracy_gate_pending=False, power_measured=False, whole_board_power_measured=False, power_scope='PMIC_output_rails_only', idle_cpu_policy_version=q.get('idle_policy',{}).get('cpu_policy_version'),PID=os.getpid())
    code = 3
    try:
        with (bundle / '.P1_active.lock').open('a+') as lock:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            slots = read(Path(q['power_plan_path']))['slots']
            need(sha(Path(q['power_plan_path']))==q['queue_plan_sha256'],'Released slot plan SHA differs')
            need(len(slots) == 12 and len({(s['model'], s['replicate']) for s in slots}) == 12, 'Frozen slot plan differs')
            for slot in slots:
                record = dict(**slot, cost_exit=None, accuracy_exit=None)
                report['records'].append(record)
                report['active_stage']='cost'; commit_state(out,report)
                child = out / ('slot%02d_%s_rep%02d' % (slot['slot'], slot['model'], slot['replicate']))
                command = [sys.executable, '-B', str(Path(__file__).resolve()), '--model-bundle', str(bundle),
                    '--bundle-sha', args.bundle_sha, '--draft-sha', args.draft_sha, '--release', str(Path(args.release).resolve()),
                    '--release-sha', args.release_sha, '--output', str(child), '--internal-job', slot['model'],
                    '--slot', str(slot['slot']), '--replicate', str(slot['replicate']), '--session', report['session']]
                with (out / (child.name + '_console.txt')).open('x', encoding='utf-8') as log:
                    def started(metadata):
                        record['cost_process']=dict(metadata); commit_state(out,report)
                    process=execute_child(command,log,q['cost_child_timeout_s'],started)
                    record['cost_process']=process; record['cost_exit']=process['actual_exit']
                    if process['timed_out']: record['failure']='cost_timeout_own_child_killed_and_reaped'
                need(record['cost_exit'] == 0, 'Cost child failed; queue stopped')
                model = next(m for m in models if m['id'] == slot['model'])
                record['cost_report_sha256'] = bind_child(child, report['session'], slot['model'], args,
                    dict(status='video_cost_and_PMIC_rails_completed_review_required', release_sha256=args.release_sha,
                         draft_manifest_sha256=args.draft_sha, profile=q['profile'], slot=slot['slot'],
                         replicate=slot['replicate'], onnx_sha256=model['onnx_sha256'], weight_sha256=model['weight_sha256'],
                         PID=process['actual_PID'],PPID=os.getpid(),process_birth_ticks=process['process_start_ticks']))
                commit_state(out,report)
                # Live receipts are atomic; final report+sidecar is only written at completion/stop.
                report['active_stage']='slot_bound'; commit_state(out,report)
                print('FORMAL_SLOT_BOUND ' + str(slot['slot']), flush=True)
            report.update(status='12_PMIC_costs_completed_review_required', exitcode=0,power_measured=True)
            code = 0
    except Exception as error:
        report['error'] = repr(error)
        (out / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
    commit_state(out,report)  # Final snapshot now reflects actual completion/failure.
    report['artifacts']=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]
    write_report(out, report)
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-bundle', required=True)
    p.add_argument('--bundle-sha', required=True)
    p.add_argument('--draft-sha', required=True)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--release')
    p.add_argument('--release-sha')
    p.add_argument('--output')
    p.add_argument('--internal-job', help=argparse.SUPPRESS)
    p.add_argument('--session', help=argparse.SUPPRESS)
    p.add_argument('--slot', type=int, help=argparse.SUPPRESS)
    p.add_argument('--replicate', type=int, help=argparse.SUPPRESS)
    p.add_argument('--local-validation-job', help=argparse.SUPPRESS)
    p.add_argument('--launcher-session',help=argparse.SUPPRESS)
    p.add_argument('--launcher-identity-sha',help=argparse.SUPPRESS)
    args = p.parse_args()
    bundle, models, dataset = bootstrap(args)
    if args.verify_only:
        print('PP1_CORE_INTEGRITY_PASSED; no experiment started')
        return 0
    need(args.release and args.release_sha and args.output, 'Target-qualified release required')
    q = qualified(args, bundle, models)
    if args.internal_job or args.local_validation_job:
        if args.local_validation_job:
            need(q['scope'] == 'local_CPU_validation', 'Local validation flag cannot execute a Pi formal job')
            args.internal_job = args.local_validation_job
            args.session, args.slot, args.replicate = 'local_CPU_pipeline_validation', 0, 0
        need(bool(args.session) and args.slot is not None and args.replicate is not None, 'Child identity missing')
        model = next(m for m in models if m['id'] == args.internal_job)
        return cost(args, q, bundle, model, dataset)
    need(q['scope'] == 'Pi_formal', 'Full queue only for separately qualified Pi release')
    return queue(args, q, bundle, models)


if __name__ == '__main__':
    raise SystemExit(main())
