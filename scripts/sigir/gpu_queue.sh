#!/usr/bin/env bash
# Single-GPU job queue for the SIGIR project (server). Keeps the 4090 busy and lets registered stages preempt the long
# next-item audit, which is resumable (chunked + atomic parts) and so acts as the FILLER job.
#
#   start (once):   setsid nohup bash scripts/sigir/gpu_queue.sh > /root/autodl-tmp/gpuq/queue.log 2>&1 < /dev/null &
#   enqueue:        cp job.sh /root/autodl-tmp/gpuq/pending/20_gatefix_stage1.sh       (lowest name runs first)
#   inspect:        ls /root/autodl-tmp/gpuq/{pending,running,done,failed}; tail /root/autodl-tmp/gpuq/logs/*.log
#   stop filler:    touch /root/autodl-tmp/gpuq/PAUSE_FILLER ; stop the queue: touch /root/autodl-tmp/gpuq/STOP
#
# A job is any bash script; it is run with `bash` from /root/autodl-tmp, its output goes to logs/<name>.log, and it moves to
# done/ on exit 0, else failed/ (never retried automatically: a failed registered stage needs a human look).
# Only one job runs at a time. While a job is pending the filler (FILLER_SCRIPT, default run_nextitem_audit.sh) is
# killed (SIGTERM to its process group; vLLM's engine child dies with it) and restarted afterwards where it stopped.
set -uo pipefail
Q="${GPUQ:-/root/autodl-tmp/gpuq}"
FILLER_SCRIPT="${FILLER_SCRIPT:-/root/autodl-tmp/run_nextitem_audit.sh}"
mkdir -p "$Q"/{pending,running,done,failed,logs}
log() { echo "[$(date '+%F %T')] $*"; }

wait_gpu_free() {   # up to 3 minutes for the GPU memory of a killed job to be released
  for _ in $(seq 1 36); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')
    [ "${used:-99999}" -lt 2000 ] && return 0
    sleep 5
  done
  log "GPU memory still in use after 180 s; killing stray vllm/pyes_scorer processes"
  pkill -9 -f "pyes_scorer|vllm|EngineCore" || true
  sleep 10
}

pending_jobs() { ls -1 "$Q"/pending/*.sh 2>/dev/null | sort; }

log "gpu_queue started; filler = $FILLER_SCRIPT"
while true; do
  [ -e "$Q/STOP" ] && { log "STOP file found, exiting"; exit 0; }
  next=$(pending_jobs | head -n1)
  if [ -n "$next" ]; then
    name=$(basename "$next")
    mv "$next" "$Q/running/$name"
    log "RUN  $name"
    wait_gpu_free
    ( cd /root/autodl-tmp && bash "$Q/running/$name" ) > "$Q/logs/$name.log" 2>&1
    rc=$?
    if [ "$rc" = 0 ]; then mv "$Q/running/$name" "$Q/done/$name"; log "DONE $name"
    else mv "$Q/running/$name" "$Q/failed/$name"; log "FAIL $name (rc=$rc), see logs/$name.log"; fi
    continue
  fi
  if [ -e "$Q/PAUSE_FILLER" ] || [ ! -f "$FILLER_SCRIPT" ]; then sleep 30; continue; fi
  log "FILLER start"
  setsid bash "$FILLER_SCRIPT" > "$Q/logs/filler.log" 2>&1 &
  fpid=$!
  while kill -0 "$fpid" 2>/dev/null; do
    if [ -n "$(pending_jobs)" ] || [ -e "$Q/PAUSE_FILLER" ] || [ -e "$Q/STOP" ]; then
      log "FILLER preempted (pid $fpid)"
      kill -TERM -- "-$fpid" 2>/dev/null || kill -TERM "$fpid" 2>/dev/null
      sleep 5; pkill -TERM -f "pyes_scorer" 2>/dev/null || true
      break
    fi
    sleep 10
  done
  wait "$fpid" 2>/dev/null; frc=$?
  log "FILLER stopped (rc=$frc)"
  wait_gpu_free
  # a filler that finished by itself (rc 0) has nothing left to do: idle until a job arrives
  if [ "$frc" = 0 ]; then
    touch "$Q/FILLER_DONE"; log "filler finished; waiting for jobs"
    while [ -z "$(pending_jobs)" ] && [ ! -e "$Q/STOP" ]; do sleep 30; done
    rm -f "$Q/FILLER_DONE"
  fi
done
