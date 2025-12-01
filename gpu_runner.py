#!/usr/bin/env python3
"""
DigitalOcean GPU Droplet Runner
Spins up a GPU droplet, transfers and runs a CUDA executable, displays logs.
"""

import os
import sys
import time
import argparse
import requests
import paramiko
from scp import SCPClient
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuration - update these or use environment variables
CONFIG = {
    "api_token": os.environ.get("DO_API_TOKEN", "your-api-token-here"),
    "ssh_key_path": os.environ.get("SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_rsa")),
    "ssh_key_fingerprint": os.environ.get("DO_SSH_KEY_FINGERPRINT", ""),  # Your SSH key fingerprint in DO
    "image_id": os.environ.get("DO_IMAGE_ID", ""),  # Your saved image ID or slug
    "region": os.environ.get("DO_REGION", "nyc1"),
    "gpu_size": os.environ.get("DO_GPU_SIZE", "gpu-h100x1-80gb"),  # GPU droplet size
    "droplet_name": "gpu-runner-temp",
}

API_BASE = "https://api.digitalocean.com/v2"


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

    def create_droplet(self) -> dict:
        """Create a GPU droplet from the saved image."""
        print(f"🚀 Creating GPU droplet '{self.config['droplet_name']}'...")
        
        payload = {
            "name": self.config["droplet_name"],
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
                # Get public IPv4
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

    def run_cuda_executable(self, remote_exe_path: str, args: str = "") -> None:
        """Make executable and run the CUDA program."""
        # Make it executable
        self.run_command(f"chmod +x {remote_exe_path}", stream_output=False)
        
        # Run the CUDA executable
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


def main():
    parser = argparse.ArgumentParser(
        description="Run CUDA executable on DigitalOcean GPU droplet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a CUDA executable
  python gpu_runner.py --exe ./my_cuda_program --image 12345678

  # List your saved images
  python gpu_runner.py --list-images

  # List available GPU sizes
  python gpu_runner.py --list-gpu-sizes

  # Run with custom options
  python gpu_runner.py --exe ./cuda_app --image my-gpu-image \\
      --region nyc1 --size gpu-h100x1-80gb --keep

Environment variables:
  DO_API_TOKEN          - DigitalOcean API token
  SSH_KEY_PATH          - Path to SSH private key
  DO_SSH_KEY_FINGERPRINT - SSH key fingerprint in DigitalOcean
  DO_IMAGE_ID           - Default image ID to use
  DO_REGION             - Default region
  DO_GPU_SIZE           - Default GPU droplet size
        """,
    )
    
    parser.add_argument("--exe", type=str, help="Path to local CUDA executable")
    parser.add_argument("--exe-args", type=str, default="", help="Arguments to pass to the executable")
    parser.add_argument("--remote-path", type=str, default="/root/cuda_program", help="Remote path for executable")
    parser.add_argument("--image", type=str, help="Image ID or slug to use")
    parser.add_argument("--region", type=str, default=CONFIG["region"], help="Region for droplet")
    parser.add_argument("--size", type=str, default=CONFIG["gpu_size"], help="GPU droplet size")
    parser.add_argument("--name", type=str, default=CONFIG["droplet_name"], help="Droplet name")
    parser.add_argument("--keep", action="store_true", help="Keep droplet running after execution")
    parser.add_argument("--list-images", action="store_true", help="List available private images")
    parser.add_argument("--list-gpu-sizes", action="store_true", help="List available GPU sizes")
    parser.add_argument("--token", type=str, help="DigitalOcean API token")
    parser.add_argument("--ssh-key", type=str, help="Path to SSH private key")
    parser.add_argument("--ssh-fingerprint", type=str, help="SSH key fingerprint in DigitalOcean")
    
    args = parser.parse_args()
    
    # Update config with command-line arguments
    config = CONFIG.copy()
    if args.token:
        config["api_token"] = args.token
    if args.ssh_key:
        config["ssh_key_path"] = args.ssh_key
    if args.ssh_fingerprint:
        config["ssh_key_fingerprint"] = args.ssh_fingerprint
    if args.image:
        config["image_id"] = args.image
    if args.region:
        config["region"] = args.region
    if args.size:
        config["gpu_size"] = args.size
    if args.name:
        config["droplet_name"] = args.name
    
    # Validate token
    if config["api_token"] == "your-api-token-here":
        print("❌ Error: Please set DO_API_TOKEN environment variable or use --token")
        sys.exit(1)
    
    runner = DigitalOceanGPURunner(config)
    
    # Handle list commands
    if args.list_images:
        runner.list_images()
        return
    
    if args.list_gpu_sizes:
        runner.list_gpu_sizes()
        return
    
    # Validate required arguments for running
    if not args.exe:
        print("❌ Error: --exe is required")
        parser.print_help()
        sys.exit(1)
    
    if not config["image_id"]:
        print("❌ Error: --image is required or set DO_IMAGE_ID")
        sys.exit(1)
    
    if not os.path.exists(args.exe):
        print(f"❌ Error: Executable not found: {args.exe}")
        sys.exit(1)
    
    # Run the workflow
    try:
        runner.create_droplet()
        runner.wait_for_droplet()
        runner.wait_for_ssh()
        runner.transfer_file(args.exe, args.remote_path)
        runner.run_cuda_executable(args.remote_path, args.exe_args)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    finally:
        runner.cleanup()
        if not args.keep:
            runner.destroy_droplet()
        else:
            print(f"\n💡 Droplet kept running at {runner.droplet_ip}")
            print(f"   SSH: ssh root@{runner.droplet_ip}")
            print(f"   Don't forget to destroy it when done!")


if __name__ == "__main__":
    main()
