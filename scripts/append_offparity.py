from pathlib import Path

p = Path("scripts/master_queue.sh")
text = p.read_text(encoding="utf-8")
if "verify_plan_loss_offparity" in text:
    print("already appended")
else:
    anchor = "touch reports/MASTER_QUEUE_COMPLETE"
    if anchor not in text:
        raise SystemExit("anchor missing in master_queue.sh")
    block = (
        'echo "=== [5/5] plan-loss off-parity verification ==="\n'
        "bash scripts/verify_plan_loss_offparity.sh\n\n"
        + anchor
    )
    p.write_text(text.replace(anchor, block, 1), encoding="utf-8")
    print("appended off-parity step to the master queue")
