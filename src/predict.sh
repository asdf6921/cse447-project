#!/usr/bin/env bash
set -e
set -v

# python myprogram.py train --work_dir work
python myprogram.py test --work_dir work --test_data $1 --test_output $2
