#!/usr/bin/env python
#  coding=utf-8
import pytest
import argparse
import sys
from pathlib import Path


TESTS_FOLDER = (Path(__file__).parent / 'unittest_reinvent').resolve()


parser = argparse.ArgumentParser(description='Run reinvent_scoring tests')
parser.add_argument(
    '--unittests', action='store_true',
    help='Only run unittests (Please indicate either integration or unittests flag)'
)
parser.add_argument(
    '--integration', action='store_true',
    help='Only run integration tests (Please indicate either integration or unittests flag)'
)

args, _ = parser.parse_known_args()


if args.unittests:
    pytest_args = ['-m', 'not integration', str(TESTS_FOLDER)]
elif args.integration:
    pytest_args = ['-m', 'integration', str(TESTS_FOLDER)]
else:
    raise Exception('Please provide either --unittests or --integration flag.')


if __name__ == '__main__':
    sys.exit(pytest.main(pytest_args))
