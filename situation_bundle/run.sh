#!/usr/bin/env bash
# Convenience: same as situation_stack/run.sh
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/situation_stack/run.sh" "$@"
