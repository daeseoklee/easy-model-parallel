#!/bin/bash

uv pip compile deps/requirements_dev.in -o deps/lock/a/requirements_dev.txt
uv pip compile deps/requirements.in -o deps/lock/a/requirements.txt
uv pip sync deps/lock/a/requirements_dev.txt
uv pip install -e .
