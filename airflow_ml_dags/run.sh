#!/bin/bash

local_dir=$(pwd)"/data/"

export LOCAL_DATA_DIR=$local_dir
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")

docker compose up --build