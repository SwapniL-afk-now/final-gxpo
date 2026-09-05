#!/usr/bin/env python3
"""Fail-closed tokenizer identity check for offline top-K KD-SFT."""
import argparse

from build_teacher_topk import validate_tokenizer_identity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-tokenizer", required=True)
    ap.add_argument("--teacher-path", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    student = AutoTokenizer.from_pretrained(args.student_tokenizer, trust_remote_code=True)
    teacher = AutoTokenizer.from_pretrained(args.teacher_path, trust_remote_code=True)
    fingerprint = validate_tokenizer_identity(student, teacher)
    print(f"tokenizer identity OK: {fingerprint}")


if __name__ == "__main__":
    main()
