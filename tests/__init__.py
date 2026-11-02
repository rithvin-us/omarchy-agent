"""Test suite for the Omarchy Agent.

Every test is hermetic: temporary directories, temporary SQLite databases and
mocked external commands and HTTP responses. No test touches the real
``~/Projects`` tree, the real data directory, or the network.
"""
