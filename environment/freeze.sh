#!/usr/bin/env bash
# Regenerate environment/requirements.txt and environment/apt-packages.txt
# from the currently running devcontainer. Run this after installing new
# packages, then commit the diff.
set -euo pipefail
cd "$(dirname "$0")"

/opt/venv/bin/pip freeze --local > requirements.txt
apt-mark showmanual | sort > apt-packages.txt
