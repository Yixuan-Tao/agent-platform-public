# Evaluation tools

This directory contains evaluation runners, judge adapters, statistics and generic keyword normalization. Historical corpora, answer sets, score snapshots and corpus-specific testsets are intentionally absent.

Supply an authorized JSON testset explicitly using `python evaluation/run_testset.py --testset <path> --output <local-output>`, from `packages/rag`. Model judging requires separately configured credentials and can incur usage. Keep generated answers and contexts outside Git.

The platform evaluation package (`packages/evaluation`) supplies synthetic tasks for software behavior. Its scores do not establish real-world fault detection, standards compliance or production model accuracy.
