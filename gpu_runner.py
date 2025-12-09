#!/usr/bin/env python3
"""
DigitalOcean GPU Droplet Runner
Persistent server mode with background auto-shutdown timer.

Usage:
    # Start server (backgrounds itself, auto-destroys after 60 min)
    python gpu_runner.py --start-server --keep-alive 60

    # Submit jobs to running server
    python gpu_runner.py --submit ./cuda_app
    python gpu_runner.py --submit ./cuda_app --exe-args "--iterations 1000"

    # Check server status
    python gpu_runner.py --status

    # Shutdown server early
    python gpu_runner.py --shutdown

    # List available images/sizes
    python gpu_runner.py --list-images
    python gpu_runner.py --list-gpu-sizes
"""

import os
import sys
import time
import json
import uuid
import signal
import argparse
import requests
import paramiko
from scp import SCPClient
from pathlib import Path
from dotenv import load_dotenv

script_dir = Path(__file__).parent
load_dotenv(script_dir / ".env")

# Configuration from environment variables
CONFIG = {
    "api_token": os.environ.get("DO_API_TOKEN"),
    "ssh_key_path": os.environ.get("SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_rsa")),
    "ssh_key_fingerprint": os.environ.get("DO_SSH_KEY_FINGERPRINT"),
    "image_id": os.environ.get("DO_IMAGE_ID"),
    "region": os.environ.get("DO_REGION", "nyc1"),
    "gpu_size": os.environ.get("DO_GPU_SIZE", "gpu-h100x1-80gb"),
}

API_BASE = "https://api.digitalocean.com/v2"
STATE_DIR = Path.home() / ".gpu_runner"
STATE_FILE = STATE_DIR / "server.json"


def save_state(droplet_id: int, droplet_ip: str, timer_pid: int, keep_alive: int):
    """Save server state to file."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "droplet_id": droplet_id,
        "droplet_ip": droplet_ip,
        "timer_pid": timer_pid,
        "keep_alive_minutes": keep_alive,
        "started_at": time.time(),
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))


def load_state() -> dict | None:
    """Load server state from file."""
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return None


def clear_state():
    """Remove state file."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def destroy_droplet_by_id(droplet_id: int):
    """Destroy a droplet by ID (standalone function for background process)."""
    headers = {
        "Authorization": f"Bearer {CONFIG['api_token']}",
        "Content-Type": "application/json",
    }
    response = requests.delete(f"{API_BASE}/droplets/{droplet_id}", headers=headers)
    return response.status_code == 204


class DigitalOceanGPURunner:
    def __init__(self, config: dict):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config['api_token']}",
            "Content-Type": "application/json",
        }
        self.droplet_id = None
        self.droplet_ip = None
        self.ssh_client = None
        self.droplet_name = f"gpu-{uuid.uuid4().hex[:8]}-droplet"

    def create_droplet(self) -> dict:
        """Create a GPU droplet from the saved image."""
        print(f"🚀 Creating GPU droplet '{self.droplet_name}'...")

        payload = {
            "name": self.droplet_name,
            "region": self.config["region"],
            "size": self.config["gpu_size"],
            "image": self.config["image_id"],
            "ssh_keys": [self.config["ssh_key_fingerprint"]] if self.config["ssh_key_fingerprint"] else [],
            "backups": False,
            "ipv6": False,
            "monitoring": True,
        }

        response = requests.post(
            f"{API_BASE}/droplets",
            headers=self.headers,
            json=payload,
        )

        if response.status_code != 202:
            raise Exception(f"Failed to create droplet: {response.status_code} - {response.text}")

        data = response.json()
        self.droplet_id = data["droplet"]["id"]
        print(f"✅ Droplet created with ID: {self.droplet_id}")
        return data

    def wait_for_droplet(self, timeout: int = 300) -> str:
        """Wait for droplet to be active and return its IP address."""
        print("⏳ Waiting for droplet to become active...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            response = requests.get(
                f"{API_BASE}/droplets/{self.droplet_id}",
                headers=self.headers,
            )

            if response.status_code != 200:
                raise Exception(f"Failed to get droplet status: {response.text}")

            droplet = response.json()["droplet"]
            status = droplet["status"]

            if status == "active":
                for network in droplet["networks"]["v4"]:
                    if network["type"] == "public":
                        self.droplet_ip = network["ip_address"]
                        print(f"✅ Droplet active at IP: {self.droplet_ip}")
                        return self.droplet_ip

            print(f"   Status: {status}...")
            time.sleep(10)

        raise TimeoutError("Droplet did not become active in time")

    def wait_for_ssh(self, timeout: int = 120) -> None:
        """Wait for SSH to become available."""
        print("⏳ Waiting for SSH to become available...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                self.ssh_client = paramiko.SSHClient()
                self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                self.ssh_client.connect(
                    self.droplet_ip,
                    username="root",
                    key_filename=self.config["ssh_key_path"],
                    timeout=10,
                )
                print("✅ SSH connection established")
                return
            except Exception as e:
                print(f"   SSH not ready yet: {e}")
                time.sleep(10)

        raise TimeoutError("SSH did not become available in time")

    def connect_ssh(self, ip: str) -> None:
        """Connect to an existing droplet via SSH."""
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_client.connect(
            ip,
            username="root",
            key_filename=self.config["ssh_key_path"],
            timeout=10,
        )

    def transfer_file(self, local_path: str, remote_path: str) -> None:
        """Transfer a file to the droplet via SCP."""
        print(f"📤 Transferring {local_path} to {remote_path}...")

        with SCPClient(self.ssh_client.get_transport()) as scp:
            scp.put(local_path, remote_path)

        print("✅ File transferred successfully")

    def run_command(self, command: str, stream_output: bool = True) -> tuple[int, str, str]:
        """Run a command on the droplet and optionally stream output."""
        print(f"🔧 Running: {command}")

        stdin, stdout, stderr = self.ssh_client.exec_command(command, get_pty=True)

        output_lines = []
        if stream_output:
            for line in iter(stdout.readline, ""):
                print(line, end="")
                output_lines.append(line)

        exit_code = stdout.channel.recv_exit_status()
        stdout_str = "".join(output_lines) if stream_output else stdout.read().decode()
        stderr_str = stderr.read().decode()

        return exit_code, stdout_str, stderr_str

    def run_cuda_executable(self, remote_exe_path: str, args: str = "") -> int:
        """Make executable and run the CUDA program."""
        self.run_command(f"chmod +x {remote_exe_path}", stream_output=False)

        print("\n" + "=" * 60)
        print("📊 CUDA EXECUTABLE OUTPUT")
        print("=" * 60 + "\n")

        exit_code, stdout, stderr = self.run_command(f"{remote_exe_path} {args}")

        print("\n" + "=" * 60)
        if exit_code == 0:
            print(f"✅ Execution completed successfully (exit code: {exit_code})")
        else:
            print(f"❌ Execution failed (exit code: {exit_code})")
            if stderr:
                print(f"STDERR:\n{stderr}")
        print("=" * 60)
        return exit_code

    def destroy_droplet(self) -> None:
        """Destroy the droplet to stop billing."""
        if not self.droplet_id:
            return

        print(f"🗑️  Destroying droplet {self.droplet_id}...")

        response = requests.delete(
            f"{API_BASE}/droplets/{self.droplet_id}",
            headers=self.headers,
        )

        if response.status_code == 204:
            print("✅ Droplet destroyed successfully")
        else:
            print(f"⚠️  Warning: Failed to destroy droplet: {response.text}")

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.ssh_client:
            self.ssh_client.close()

    def list_images(self) -> None:
        """List available private images (snapshots)."""
        print("📷 Listing your saved images/snapshots...")

        response = requests.get(
            f"{API_BASE}/images?private=true",
            headers=self.headers,
        )

        if response.status_code != 200:
            raise Exception(f"Failed to list images: {response.text}")

        images = response.json()["images"]
        print(f"\nFound {len(images)} private image(s):\n")
        for img in images:
            print(f"  ID: {img['id']}")
            print(f"  Name: {img['name']}")
            print(f"  Created: {img['created_at']}")
            print(f"  Size: {img['size_gigabytes']} GB")
            print(f"  Regions: {', '.join(img['regions'])}")
            print("-" * 40)

    def list_gpu_sizes(self) -> None:
        """List available GPU droplet sizes."""
        print("🖥️  Listing available GPU droplet sizes...")

        response = requests.get(
            f"{API_BASE}/sizes",
            headers=self.headers,
        )

        if response.status_code != 200:
            raise Exception(f"Failed to list sizes: {response.text}")

        sizes = response.json()["sizes"]
        gpu_sizes = [s for s in sizes if "gpu" in s["slug"].lower()]

        print(f"\nFound {len(gpu_sizes)} GPU size(s):\n")
        for size in gpu_sizes:
            print(f"  Slug: {size['slug']}")
            print(f"  vCPUs: {size['vcpus']}, Memory: {size['memory']} MB, Disk: {size['disk']} GB")
            print(f"  Price: ${size['price_hourly']}/hour (${size['price_monthly']}/month)")
            print(f"  Regions: {', '.join(size['regions'][:3])}...")
            print("-" * 40)


def cmd_start_server(args):
    """Start a GPU server in the background with auto-shutdown timer."""
    # Check if server already running
    state = load_state()
    if state:
        print(f"⚠️  Server already running at {state['droplet_ip']}")
        print(f"   Use --submit to run jobs, or --shutdown to stop it")
        sys.exit(1)

    if not CONFIG["image_id"]:
        print("❌ Error: DO_IMAGE_ID not set in environment/.env")
        sys.exit(1)

    keep_alive = args.keep_alive or 10  # Default 60 minutes

    runner = DigitalOceanGPURunner(CONFIG)

    try:
        runner.create_droplet()
        runner.wait_for_droplet()
        runner.wait_for_ssh()
        runner.cleanup()  # Close SSH, we just needed to verify it works
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        runner.destroy_droplet()
        sys.exit(1)

    # Fork the background timer process
    pid = os.fork()

    if pid > 0:
        # Parent process - save state and exit
        save_state(runner.droplet_id, runner.droplet_ip, pid, keep_alive)
        print(f"\n✅ Server started successfully!")
        print(f"   Droplet ID: {runner.droplet_id}")
        print(f"   IP Address: {runner.droplet_ip}")
        print(f"   SSH: ssh root@{runner.droplet_ip}")
        print(f"   Timer PID: {pid}")
        print(f"   Auto-shutdown in: {keep_alive} minutes")
        print(f"\n   Use 'python gpu_runner.py --submit <exe>' to run jobs")
        print(f"   Use 'python gpu_runner.py --shutdown' to stop early")
        sys.exit(0)

    else:
        # Child process - become background timer daemon
        os.setsid()  # Detach from terminal

        # Close standard file descriptors
        sys.stdin.close()
        sys.stdout.close()
        sys.stderr.close()

        # Sleep for keep-alive duration
        time.sleep(keep_alive * 60)

        # Time's up - destroy droplet
        destroy_droplet_by_id(runner.droplet_id)
        clear_state()
        os._exit(0)


def cmd_submit(args):
    """Submit a job to the running server."""
    state = load_state()
    if not state:
        print("❌ No server running. Start one with --start-server")
        sys.exit(1)

    if not os.path.exists(args.submit):
        print(f"❌ Error: Executable not found: {args.submit}")
        sys.exit(1)

    runner = DigitalOceanGPURunner(CONFIG)
    runner.droplet_ip = state["droplet_ip"]

    try:
        print(f"🔗 Connecting to server at {runner.droplet_ip}...")
        runner.connect_ssh(runner.droplet_ip)

        remote_path = f"/root/cuda_program_{uuid.uuid4().hex[:8]}"
        runner.transfer_file(args.submit, remote_path)
        exit_code = runner.run_cuda_executable(remote_path, args.exe_args)

        # Cleanup remote file
        runner.run_command(f"rm -f {remote_path}", stream_output=False)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        runner.cleanup()

    # Calculate remaining time
    elapsed = (time.time() - state["started_at"]) / 60
    remaining = max(0, state["keep_alive_minutes"] - elapsed)
    print(f"\n⏰ Server time remaining: {remaining:.1f} minutes")


def cmd_status(args):
    """Show server status."""
    state = load_state()
    if not state:
        print("❌ No server running")
        sys.exit(0)

    elapsed = (time.time() - state["started_at"]) / 60
    remaining = max(0, state["keep_alive_minutes"] - elapsed)

    print(f"✅ Server running")
    print(f"   Droplet ID: {state['droplet_id']}")
    print(f"   IP Address: {state['droplet_ip']}")
    print(f"   SSH: ssh root@{state['droplet_ip']}")
    print(f"   Timer PID: {state['timer_pid']}")
    print(f"   Elapsed: {elapsed:.1f} minutes")
    print(f"   Remaining: {remaining:.1f} minutes")

    # Check if timer process is still running
    try:
        os.kill(state["timer_pid"], 0)
        print(f"   Timer: running")
    except OSError:
        print(f"   Timer: ⚠️  not running (may have crashed)")


def cmd_shutdown(args):
    """Shutdown the server immediately."""
    state = load_state()
    if not state:
        print("❌ No server running")
        sys.exit(0)

    # Kill the timer process
    try:
        os.kill(state["timer_pid"], signal.SIGTERM)
        print(f"✅ Killed timer process (PID: {state['timer_pid']})")
    except OSError:
        print(f"⚠️  Timer process already dead")

    # Destroy the droplet
    print(f"🗑️  Destroying droplet {state['droplet_id']}...")
    if destroy_droplet_by_id(state["droplet_id"]):
        print("✅ Droplet destroyed successfully")
    else:
        print("⚠️  Failed to destroy droplet (may already be destroyed)")

    clear_state()
    print("✅ Server shutdown complete")


def main():
    parser = argparse.ArgumentParser(
        description="DigitalOcean GPU Droplet Runner - Persistent Server Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with 60 min timeout
  python gpu_runner.py --start-server --keep-alive 60

  # Submit jobs
  python gpu_runner.py --submit ./cuda_app
  python gpu_runner.py --submit ./cuda_app --exe-args "--iterations 1000"

  # Check status
  python gpu_runner.py --status

  # Shutdown early
  python gpu_runner.py --shutdown
        """,
    )

    # Server commands
    parser.add_argument("--start-server", action="store_true", help="Start a GPU server (backgrounds itself)")
    parser.add_argument("--keep-alive", type=int, metavar="MINUTES", help="Auto-shutdown after N minutes (default: 60)")
    parser.add_argument("--submit", type=str, metavar="EXE", help="Submit executable to running server")
    parser.add_argument("--exe-args", type=str, default="", help="Arguments to pass to the executable")
    parser.add_argument("--status", action="store_true", help="Show server status")
    parser.add_argument("--shutdown", action="store_true", help="Shutdown server immediately")

    # Info commands
    parser.add_argument("--list-images", action="store_true", help="List available private images")
    parser.add_argument("--list-gpu-sizes", action="store_true", help="List available GPU sizes")

    args = parser.parse_args()

    # Validate API token
    if not CONFIG["api_token"]:
        print("❌ Error: DO_API_TOKEN not set in environment/.env")
        sys.exit(1)

    # Route to appropriate command
    if args.list_images:
        runner = DigitalOceanGPURunner(CONFIG)
        runner.list_images()
    elif args.list_gpu_sizes:
        runner = DigitalOceanGPURunner(CONFIG)
        runner.list_gpu_sizes()
    elif args.start_server:
        cmd_start_server(args)
    elif args.submit:
        cmd_submit(args)
    elif args.status:
        cmd_status(args)
    elif args.shutdown:
        cmd_shutdown(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()