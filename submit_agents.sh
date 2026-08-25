#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 ENTITY/PROJECT/SWEEP_ID" >&2
  echo "Launch additional agents by running this command again in another process." >&2
  exit 2
fi

exec wandb agent "$1"
