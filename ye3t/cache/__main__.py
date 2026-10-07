"""Inspect or explicitly prune verified YE3T cache envelopes."""

import argparse
import json

from .artifacts import YE3TArtifactStore


def main():
    parser = argparse.ArgumentParser(prog="python -m ye3t.cache")
    parser.add_argument("--cache-dir", help="Cache root (defaults to YE3T_CACHE_DIR)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="Report local JSON envelope integrity")
    prune = commands.add_parser("prune", help="Prune selected cache entries")
    prune.add_argument("--entry", action="append", required=True,
                       help="Entry path relative to the cache root; repeat as needed")
    prune.add_argument("--apply", action="store_true",
                       help="Perform deletion; the default only previews")
    prune.add_argument("--include-valid", action="store_true",
                       help="Permit explicit deletion of integrity-valid entries")
    args = parser.parse_args()
    store = YE3TArtifactStore(directory=args.cache_dir)
    try:
        if args.command == "inspect":
            result = store.inspect()
        else:
            result = store.prune(args.entry, dry_run=not args.apply,
                                 invalid_only=not args.include_valid)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
