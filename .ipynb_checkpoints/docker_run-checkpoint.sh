#!/bin/bash
# =============================================================================
# AbXtract Docker Build & Run Script
# =============================================================================
#
# Usage:
#   ./docker_run.sh build              # Build the Docker image
#   ./docker_run.sh run [args...]      # Run AbXtract CLI
#   ./docker_run.sh shell              # Open interactive shell
#   ./docker_run.sh help               # Show help
#
# Examples:
#   ./docker_run.sh build
#   ./docker_run.sh run -i data/input/antibodies.csv -o data/output/ -m r
#   ./docker_run.sh shell
#
# =============================================================================

set -e

IMAGE_NAME="abxtract"
IMAGE_TAG="latest"
CONTAINER_NAME="abxtract-cli"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_usage() {
    echo ""
    echo "AbXtract Docker Runner"
    echo "======================"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  run         Run AbXtract CLI with provided arguments"
    echo "  shell       Open an interactive shell in the container"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 run -i data/test/input.csv -o data/output/ -m r"
    echo "  $0 run -i input.csv -o results/ -m wd --pH 6.5"
    echo "  $0 shell"
    echo ""
    echo "For CLI options, run: $0 run --help"
    echo ""
}

build_image() {
    echo -e "${GREEN}Building Docker image ${IMAGE_NAME}:${IMAGE_TAG}...${NC}"
    
    # Check if Dockerfile exists
    if [ ! -f "${SCRIPT_DIR}/Dockerfile" ]; then
        echo -e "${RED}Error: Dockerfile not found in ${SCRIPT_DIR}${NC}"
        exit 1
    fi
    
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}/.."
    
    echo -e "${GREEN}Build complete!${NC}"
}

run_abxtract() {
    echo -e "${GREEN}Running AbXtract...${NC}"
    
    # Create data directories if they don't exist
    mkdir -p "${SCRIPT_DIR}/../data/test"
    mkdir -p "${SCRIPT_DIR}/../data/output"
    
    # Check if image exists
    if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" &> /dev/null; then
        echo -e "${YELLOW}Image not found. Building first...${NC}"
        build_image
    fi
    
    # Run the container
    docker run --rm \
        -v "${SCRIPT_DIR}/../data/test:/data/test:ro" \
        -v "${SCRIPT_DIR}/../data/output:/data/output:rw" \
        -v "${SCRIPT_DIR}/..:/app/AbXtract_host:ro" \
        --name "${CONTAINER_NAME}" \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        python /app/run_abxtract.py "$@"
}

run_shell() {
    echo -e "${GREEN}Opening interactive shell...${NC}"
    
    # Create data directories if they don't exist
    mkdir -p "${SCRIPT_DIR}/../data/test"
    mkdir -p "${SCRIPT_DIR}/../data/output"
    
    # Check if image exists
    if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" &> /dev/null; then
        echo -e "${YELLOW}Image not found. Building first...${NC}"
        build_image
    fi
    
    docker run --rm -it \
        -v "${SCRIPT_DIR}/../data/test:/data/test:rw" \
        -v "${SCRIPT_DIR}/../data/output:/data/output:rw" \
        --name "${CONTAINER_NAME}" \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        bash
}

# Main logic
case "${1:-}" in
    build)
        build_image
        ;;
    run)
        shift
        if [ $# -eq 0 ]; then
            echo -e "${YELLOW}No arguments provided. Showing help...${NC}"
            run_abxtract --help
        else
            run_abxtract "$@"
        fi
        ;;
    shell)
        run_shell
        ;;
    help|--help|-h|"")
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        print_usage
        exit 1
        ;;
esac
