"""
Import smoke test for higher-level bidding modules.

Two tiers:
  Tier 1 — Pure Python modules: actually imported, any error is a real bug.
  Tier 2 — Framework-dependent modules (verl/transformers/ray): syntax-checked
            via ast.parse so we catch bugs in our own code without needing the
            full training cluster environment.

Run from the DrMAS_bidding_setup1 directory:
    python -X utf8 smoke_test_imports.py
"""

import sys, os, ast, importlib, traceback

sys.path.insert(0, os.path.dirname(__file__))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PASS    = "  PASS   "
FAIL    = "  FAIL   "
SYNTAX  = "  SYNTAX "
SKIP    = "  SKIP   "

errors = []

# Framework packages absent in this dev environment (present on training cluster)
FRAMEWORK_PKGS = {"verl", "ray", "transformers", "omegaconf", "hydra"}


def _is_framework_error(exc: ImportError) -> bool:
    msg = str(exc)
    return any(pkg in msg for pkg in FRAMEWORK_PKGS)


def try_import(label: str, module: str):
    """Actually import the module. Fails on any error including syntax."""
    try:
        importlib.import_module(module)
        print(f"{PASS} {label}")
    except ImportError as e:
        if _is_framework_error(e):
            # Framework not installed here — not our bug
            print(f"{SKIP} {label}  (framework dep: {e})")
        else:
            print(f"{FAIL} {label}  -- ImportError: {e}")
            errors.append(f"{label}: {e}")
    except SyntaxError as e:
        print(f"{FAIL} {label}  -- SyntaxError at line {e.lineno}: {e.msg}")
        errors.append(f"{label}: SyntaxError line {e.lineno}: {e.msg}")
    except Exception as e:
        print(f"{FAIL} {label}  -- {type(e).__name__}: {e}")
        errors.append(f"{label}: {type(e).__name__}: {e}")


def try_syntax(label: str, rel_path: str):
    """
    Parse the file with ast.parse to check for syntax errors.
    This works even when framework imports would fail at runtime.
    """
    full = os.path.join(os.path.dirname(__file__), rel_path)
    if not os.path.exists(full):
        print(f"{FAIL} {label}  -- file not found: {rel_path}")
        errors.append(f"{label}: file not found")
        return
    try:
        src = open(full, encoding="utf-8").read()
        ast.parse(src, filename=rel_path)
        print(f"{SYNTAX} {label}  (syntax OK)")
    except SyntaxError as e:
        print(f"{FAIL} {label}  -- SyntaxError at line {e.lineno}: {e.msg}")
        errors.append(f"{label}: SyntaxError line {e.lineno}: {e.msg}")


# -----------------------------------------------------------------------
# Tier 1 — Pure Python: actually import
# -----------------------------------------------------------------------
print("\n=== Tier 1: Pure Python modules (real import) ===\n")

try_import(
    "bidding.projection",
    "agent_system.environments.env_package.bidding.projection",
)
try_import(
    "bidding.bidding_reward",
    "agent_system.environments.env_package.bidding.bidding_reward",
)
try_import(
    "bidding.envs",
    "agent_system.environments.env_package.bidding.envs",
)
try_import(
    "bidding.__init__ (exports bidding_projection + build_bidding_envs)",
    "agent_system.environments.env_package.bidding",
)
try_import(
    "memory.base",
    "agent_system.memory.base",
)
try_import(
    "memory.memory (BiddingMemory)",
    "agent_system.memory.memory",
)
try_import(
    "environments.prompts.bidding",
    "agent_system.environments.prompts.bidding",
)

# Verify the two symbols env_manager.py imports from bidding __init__
print()
try:
    from agent_system.environments.env_package.bidding import (
        bidding_projection,
        build_bidding_envs,
    )
    print(f"{PASS} bidding.__init__ exports: bidding_projection + build_bidding_envs")
except Exception as e:
    msg = f"bidding.__init__ missing exports: {e}"
    print(f"{FAIL} {msg}")
    errors.append(msg)

# Verify memory exports
try:
    from agent_system.memory import BiddingMemory, SimpleMemory, SearchMemory
    print(f"{PASS} memory.__init__ exports: BiddingMemory, SimpleMemory, SearchMemory")
except Exception as e:
    msg = f"memory.__init__ missing exports: {e}"
    print(f"{FAIL} {msg}")
    errors.append(msg)

# -----------------------------------------------------------------------
# Tier 2 — Framework-dependent: syntax check only
# -----------------------------------------------------------------------
print("\n=== Tier 2: Framework-dependent modules (syntax check) ===\n")

try_syntax(
    "bidding_agents.py",
    "agent_system/agent/agents/bidding/bidding_agents.py",
)
try_syntax(
    "detector_agent.py",
    "agent_system/agent/agents/bidding/detector_agent.py",
)
try_syntax(
    "bidding_orchestra.py",
    "agent_system/agent/orchestra/bidding/bidding_orchestra.py",
)
try_syntax(
    "env_manager.py",
    "agent_system/environments/env_manager.py",
)
try_syntax(
    "rollout_loop.py",
    "agent_system/multi_turn_rollout/rollout_loop.py",
)
try_syntax(
    "episode.py (reward manager)",
    "agent_system/reward_manager/episode.py",
)
try_syntax(
    "main_ppo.py",
    "verl/trainer/main_ppo.py",
)
try_syntax(
    "drmas_bidding.py (dataset preprocessor)",
    "examples/data_preprocess/drmas_bidding.py",
)
# run_bidding.sh is a bash script — checked in structural section below, not here

# Shell scripts: just check the file exists and is non-empty
sh = os.path.join(os.path.dirname(__file__), "examples/drmas_trainer/run_bidding.sh")
if os.path.exists(sh) and os.path.getsize(sh) > 0:
    print(f"{PASS} run_bidding.sh  (file exists, non-empty)")
else:
    msg = "run_bidding.sh missing or empty"
    print(f"{FAIL} {msg}")
    errors.append(msg)

# -----------------------------------------------------------------------
# Tier 3 — Structural checks (no import needed)
# -----------------------------------------------------------------------
print("\n=== Tier 3: Structural checks ===\n")

required_files = {
    "BiddingEnv implementation":
        "agent_system/environments/env_package/bidding/envs.py",
    "bidding __init__":
        "agent_system/environments/env_package/bidding/__init__.py",
    "bidding_reward":
        "agent_system/environments/env_package/bidding/bidding_reward.py",
    "projection":
        "agent_system/environments/env_package/bidding/projection.py",
    "BiddingMemory":
        "agent_system/memory/memory.py",
    "BiddingEnvironmentManager":
        "agent_system/environments/env_manager.py",
    "BiddingOrchestra":
        "agent_system/agent/orchestra/bidding/bidding_orchestra.py",
    "BidderA/B agents":
        "agent_system/agent/agents/bidding/bidding_agents.py",
    "DetectorAgent":
        "agent_system/agent/agents/bidding/detector_agent.py",
    "rollout_loop":
        "agent_system/multi_turn_rollout/rollout_loop.py",
    "bidding prompts":
        "agent_system/environments/prompts/bidding.py",
    "dataset preprocessor":
        "examples/data_preprocess/drmas_bidding.py",
    "launch script":
        "examples/drmas_trainer/run_bidding.sh",
}

for label, rel_path in required_files.items():
    full = os.path.join(os.path.dirname(__file__), rel_path)
    exists = os.path.exists(full)
    size = os.path.getsize(full) if exists else 0
    non_empty = size > 10  # ignore truly empty files
    if exists and non_empty:
        print(f"{PASS} {label}  ({rel_path})")
    elif exists and not non_empty:
        msg = f"{label} exists but is empty ({rel_path})"
        print(f"{FAIL} {msg}")
        errors.append(msg)
    else:
        msg = f"{label} MISSING ({rel_path})"
        print(f"{FAIL} {msg}")
        errors.append(msg)

# Check build_bidding_envs is in __init__.py
init_path = os.path.join(
    os.path.dirname(__file__),
    "agent_system/environments/env_package/bidding/__init__.py",
)
content = open(init_path).read()
if "build_bidding_envs" in content:
    print(f"{PASS} build_bidding_envs exported in bidding/__init__.py")
else:
    msg = "build_bidding_envs NOT exported in bidding/__init__.py"
    print(f"{FAIL} {msg}")
    errors.append(msg)

# Check done=False in envs.py
envs_path = os.path.join(
    os.path.dirname(__file__),
    "agent_system/environments/env_package/bidding/envs.py",
)
envs_src = open(envs_path).read()
if "done=False" in envs_src or "False, info" in envs_src:
    print(f"{PASS} BiddingEnv returns done=False (multi-round episodes)")
else:
    msg = "BiddingEnv still returns done=True — multi-round episodes broken"
    print(f"{FAIL} {msg}")
    errors.append(msg)

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
if errors:
    print(f"FAILED -- {len(errors)} issue(s):")
    for e in errors:
        print(f"  x {e}")
    sys.exit(1)
else:
    print("All checks passed.")
