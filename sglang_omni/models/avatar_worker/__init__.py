# SPDX-License-Identifier: Apache-2.0
"""Avatar models served by a chunk worker in their own environment (LiveAct, Avatar Forcing, DyStream): a node of a Session graph that sends each chunk of the agent's speech, and of the user's audio it listens to, to the worker over HTTP; no model code runs in the sglang-omni process."""
