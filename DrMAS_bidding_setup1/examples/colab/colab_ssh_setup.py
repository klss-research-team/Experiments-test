# ============================================================
# Connect VS Code to this Colab A100 via SSH + cloudflared tunnel
#
# HOW TO USE:
#   1. Run this cell in Colab (Runtime → A100 GPU)
#   2. Copy the SSH config block it prints
#   3. Paste into C:\Users\HP\.ssh\config on your local machine
#   4. In VS Code: Ctrl+Shift+P → "Remote-SSH: Connect to Host" → "colab-a100"
#   5. Password is printed below
# ============================================================

import subprocess, threading, time, re, os

# ── 1. Install cloudflared ──────────────────────────────────
print("📦 Installing cloudflared...")
subprocess.run([
    "wget", "-q", "-O", "/usr/local/bin/cloudflared",
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
], check=True)
subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"], check=True)
print("   ✅ cloudflared ready")

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

# ── 4. Start cloudflared tunnel ─────────────────────────────
print("🌐 Starting cloudflared tunnel (this takes ~10s)...")
tunnel = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "ssh://localhost:22"],
    stderr=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True,
)

url = None
for _ in range(40):
    line = tunnel.stderr.readline()
    match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
    if match:
        url = match.group(0)
        break

if not url:
    raise RuntimeError("❌ Could not get tunnel URL. Re-run the cell.")

hostname = url.replace("https://", "")

# ── 5. Print instructions ───────────────────────────────────
print(f"""
{'='*62}
✅  Tunnel is live!

Add this block to  C:\\Users\\HP\\.ssh\\config  on your Windows machine:

Host colab-a100
    HostName {hostname}
    User root
    Port 22
    ProxyCommand cloudflared access ssh --hostname %h

Then in VS Code:
  Ctrl+Shift+P → "Remote-SSH: Connect to Host" → colab-a100

SSH password: {PASSWORD}

Repo is at: {repo_path}/DrMAS_bidding_setup1
{'='*62}
""")
