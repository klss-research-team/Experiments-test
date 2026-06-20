# ============================================================
# Connect VS Code to this Colab A100 via SSH + bore TCP tunnel
#
# HOW TO USE:
#   1. Run this cell in Colab (Runtime → A100 GPU)
#   2. It prints an SSH config block — paste into C:\Users\HP\.ssh\config
#   3. In VS Code: Ctrl+Shift+P → "Remote-SSH: Connect to Host" → colab-a100
#   4. Password is printed below
# ============================================================

import subprocess, time, re, os

# ── 1. Install bore (raw TCP tunnel — no websocket issues) ──
print("📦 Installing bore tunnel...")
subprocess.run([
    "bash", "-c",
    "curl -fsSL https://github.com/ekzhang/bore/releases/download/v0.5.1/bore-v0.5.1-x86_64-unknown-linux-musl.tar.gz "
    "| tar xz -C /usr/local/bin/"
], check=True)
print("   ✅ bore ready")

# ── 2. Set up SSH server ────────────────────────────────────
print("🔧 Setting up SSH server...")
subprocess.run(["apt-get", "install", "-y", "-q", "openssh-server"], check=True)
subprocess.run(["mkdir", "-p", "/var/run/sshd"], check=True)

PASSWORD = "drmas2026"
subprocess.run(["bash", "-c", f"echo 'root:{PASSWORD}' | chpasswd"], check=True)

with open("/etc/ssh/sshd_config", "a") as f:
    f.write("\nPermitRootLogin yes\nPasswordAuthentication yes\n")

subprocess.Popen(["/usr/sbin/sshd"])
time.sleep(1)
print("   ✅ SSH server started")

# ── 3. Pull latest code ─────────────────────────────────────
print("📥 Pulling latest code from GitHub...")
repo_path = "/content/Experiments-test"
if os.path.exists(repo_path):
    subprocess.run(["git", "-C", repo_path, "pull"], check=True)
else:
    subprocess.run([
        "git", "clone",
        "https://github.com/klss-research-team/Experiments-test.git",
        repo_path
    ], check=True)
print("   ✅ Code up to date")

# ── 4. Start bore tunnel (raw TCP on bore.pub) ──────────────
print("🌐 Starting bore tunnel...")
tunnel = subprocess.Popen(
    ["bore", "local", "22", "--to", "bore.pub"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)

port = None
for _ in range(40):
    line = tunnel.stdout.readline()
    match = re.search(r'listening at bore\.pub:(\d+)', line)
    if match:
        port = match.group(1)
        break

if not port:
    raise RuntimeError("❌ Could not get bore port. Re-run the cell.")

# ── 5. Print instructions ───────────────────────────────────
print(f"""
{'='*62}
✅  Tunnel is live!

Paste this into  C:\\Users\\HP\\.ssh\\config  on your Windows machine
(replace any previous colab-a100 block):

Host colab-a100
    HostName bore.pub
    User root
    Port {port}

Then in VS Code:
  Ctrl+Shift+P → "Remote-SSH: Connect to Host" → colab-a100

SSH password: {PASSWORD}

Repo is at: {repo_path}/DrMAS_bidding_setup1
{'='*62}
""")
