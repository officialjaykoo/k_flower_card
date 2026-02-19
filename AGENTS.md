# Project Agent Rules

These rules are mandatory for this repository.

1. Deletion safety
- Before deleting anything, always provide a detailed delete list to the user.
- The list must include paths and clear scope.
- Do not delete until the user explicitly approves.

2. Simulation execution control
- Do not run simulation commands on your own.
- Run simulation only when the user explicitly requests it in the current conversation.

3. Test game count lock
- For testing simulations, use exactly 1000 games.
- If a planned test simulation uses a different game count, stop and ask the user first.
