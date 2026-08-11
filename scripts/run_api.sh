#!/bin/zsh
project_dir="${0:A:h:h}"
set -a
source "$project_dir/.env"
set +a
cd "$project_dir"
exec "$project_dir/.venv/bin/anima-mlx-api"
