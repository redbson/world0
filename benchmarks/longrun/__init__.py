"""LongRun: a long-horizon Agent stream with hidden ground truth.

The same stream of events is given to World 0 and to the memory strategies
an Agent would otherwise use (raw context window, retrieval over past
events, summary buffer, fact store, knowledge graph); each is asked, at the
same token budget, to hand the Agent a context for a new query, and the
context is scored against what the hidden world says the Agent needed.
See ``docs/eval/`` for the design, the results and their limits.
"""
