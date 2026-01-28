#!/bin/bash
# OcuMamba AWS GPU Runner
# This script syncs your code to AWS EC2 and runs commands there
#
# SETUP (run once):
# 1. Edit the variables below with your EC2 details
# 2. chmod +x scripts/gpu_run.sh
# 3. Test: ./scripts/gpu_run.sh test

# ============ CONFIGURE THESE ============
EC2_HOST="ubuntu@YOUR_EC2_IP"          # Change to your EC2 public IP
EC2_KEY="~/.ssh/your-key.pem"          # Path to your SSH key
REMOTE_DIR="~/Backend"                  # Where code lives on EC2
# =========================================

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

sync_code() {
    echo -e "${YELLOW}📤 Syncing code to EC2...${NC}"
    rsync -avz --delete \
        --exclude '__pycache__' \
        --exclude '.git' \
        --exclude 'cache' \
        --exclude '*.pyc' \
        -e "ssh -i $EC2_KEY" \
        "$LOCAL_DIR/" \
        "$EC2_HOST:$REMOTE_DIR/"
    echo -e "${GREEN}✓ Sync complete${NC}"
}

run_remote() {
    echo -e "${YELLOW}🚀 Running on GPU...${NC}"
    ssh -i "$EC2_KEY" "$EC2_HOST" "cd $REMOTE_DIR && PYTHONPATH=$REMOTE_DIR $@"
}

case "$1" in
    sync)
        sync_code
        ;;
    test)
        sync_code
        echo -e "${YELLOW}🧪 Running tests...${NC}"
        run_remote 'python3 -c "
import torch
print(f\"CUDA available: {torch.cuda.is_available()}\")
if torch.cuda.is_available():
    print(f\"GPU: {torch.cuda.get_device_name(0)}\")

from Backend.indexing.mamba import VisionMambaEncoder
encoder = VisionMambaEncoder()
print(f\"Mamba status: {encoder.status}\")
"'
        ;;
    run)
        sync_code
        shift
        run_remote "$@"
        ;;
    install)
        echo -e "${YELLOW}📦 Installing dependencies on EC2...${NC}"
        run_remote 'pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121'
        run_remote 'pip install mamba-ssm>=2.0.0 causal-conv1d>=1.1.0 numpy opencv-python pillow'
        ;;
    *)
        echo "Usage: $0 {sync|test|run|install}"
        echo ""
        echo "Commands:"
        echo "  sync     - Sync local code to EC2"
        echo "  test     - Sync and run GPU tests"
        echo "  run CMD  - Sync and run any command"
        echo "  install  - Install GPU dependencies on EC2"
        echo ""
        echo "Example:"
        echo "  $0 run python3 -c 'import torch; print(torch.cuda.is_available())'"
        ;;
esac
