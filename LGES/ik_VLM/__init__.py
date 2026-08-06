"""ik_VLM — supervised execution layer over the scripted ik_demo pipeline.

Nominal behavior is 100% the scripted IK sequence (ik_demo). This package adds:

  Tier 0  safe-hold      in-descent abort (suction.place tick_cb hook) + small lift
  Tier 1  re-entry       re-detect the world, classify, resume the script at the
                         right point (resume_matrix)
  Tier 2  VLM advisor    open-set situations: a local VLM looks at the head
                         camera + context and proposes a skill from a bounded
                         library (operator-approved before execution)

Detection of "something other than the scripted situation" is open-set: a
per-phase force-envelope monitor trained on NOMINAL runs only (monitor.py),
not a classifier over enumerated failures.
"""
