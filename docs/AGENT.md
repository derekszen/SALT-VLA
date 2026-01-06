# Agent Notes

Training health check
- Command: python check_training_health.py <log_file>
- Exit codes: 0 healthy, 1 warning, 2 error
- Key fields: status, recommendation, issues, progress, loss, performance

Suggested decision logic
- healthy: continue
- warning: investigate but continue
- error: recommend cancel

Find logs and processes
- Active runs: ps aux | rg -i "train_.*\.py"
- Recent task logs: ls -lt /tmp/claude/-home-derekszen-Projects-SALT-VLA/tasks/*.output | head -5
- Run logs (preferred): run_logs/

Queueing
- queue_training.py can chain runs; check its status output for next steps.

Save metrics snapshot
- python check_training_health.py <log_file> > run_summary_$(date +%Y%m%d_%H%M%S).json
