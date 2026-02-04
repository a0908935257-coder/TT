#!/bin/bash
# ==========================================================================
# Grid Trading Bot - Cloud Deployment Script
# ==========================================================================
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh [command]
#
# Commands:
#   install     - Install Docker and dependencies
#   setup       - Configure environment and start services
#   start       - Start all services
#   stop        - Stop all services
#   restart     - Restart all services
#   logs        - Show trading bot logs
#   status      - Check service status
#   update      - Pull latest code and restart
#   backup      - Backup database
#   restore     - Restore database from backup
#
# ==========================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==========================================================================
# Helper Functions
# ==========================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "Do not run this script as root. Use a regular user with sudo access."
        exit 1
    fi
}

# ==========================================================================
# Installation
# ==========================================================================

install_docker() {
    log_info "Installing Docker..."

    if command -v docker &> /dev/null; then
        log_success "Docker is already installed"
        docker --version
        return
    fi

    # Install Docker using official script
    curl -fsSL https://get.docker.com | sh

    # Add current user to docker group
    sudo usermod -aG docker "$USER"

    log_success "Docker installed successfully"
    log_warn "Please log out and log back in for docker group changes to take effect"
    log_warn "Or run: newgrp docker"
}

install_compose() {
    log_info "Installing Docker Compose..."

    if docker compose version &> /dev/null; then
        log_success "Docker Compose is already installed"
        docker compose version
        return
    fi

    # Docker Compose V2 is included in Docker Engine 20.10+
    # If not available, install the plugin
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin

    log_success "Docker Compose installed successfully"
}

install_dependencies() {
    log_info "Installing system dependencies..."

    sudo apt-get update
    sudo apt-get install -y \
        git \
        curl \
        htop \
        nano \
        unzip

    log_success "Dependencies installed"
}

# ==========================================================================
# Setup
# ==========================================================================

setup_env() {
    log_info "Setting up environment..."

    if [[ ! -f .env ]]; then
        if [[ -f .env.example ]]; then
            cp .env.example .env
            log_warn "Created .env from .env.example"
            log_warn "Please edit .env and configure your API keys:"
            log_warn "  nano .env"
        else
            log_error ".env.example not found"
            exit 1
        fi
    else
        log_success ".env file exists"
    fi

    # Validate required environment variables
    source .env 2>/dev/null || true

    local missing=()

    [[ -z "$BINANCE_API_KEY" || "$BINANCE_API_KEY" == "your_api_key" ]] && missing+=("BINANCE_API_KEY")
    [[ -z "$BINANCE_API_SECRET" || "$BINANCE_API_SECRET" == "your_api_secret" ]] && missing+=("BINANCE_API_SECRET")
    [[ -z "$DB_PASSWORD" || "$DB_PASSWORD" == "your_password" ]] && missing+=("DB_PASSWORD")

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing required environment variables:"
        for var in "${missing[@]}"; do
            log_error "  - $var"
        done
        log_warn "Please edit .env and set these values"
        return 1
    fi

    log_success "Environment variables configured"
}

setup_directories() {
    log_info "Setting up directories..."

    mkdir -p logs
    chmod 755 logs

    log_success "Directories created"
}

# ==========================================================================
# Service Management
# ==========================================================================

start_services() {
    log_info "Starting services..."

    # Validate config first
    docker compose config --quiet

    # Build and start
    docker compose up -d --build

    log_success "Services started"

    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 10

    show_status
}

stop_services() {
    log_info "Stopping services..."

    docker compose down

    log_success "Services stopped"
}

restart_services() {
    log_info "Restarting services..."

    docker compose restart

    log_success "Services restarted"
}

show_logs() {
    local service="${1:-trading}"
    local lines="${2:-100}"

    docker compose logs -f --tail="$lines" "$service"
}

show_status() {
    log_info "Service status:"
    echo ""
    docker compose ps
    echo ""

    # Check if trading bot is running
    if docker compose ps | grep -q "trading-bot.*Up"; then
        log_success "Trading bot is running"

        # Show recent logs
        echo ""
        log_info "Recent logs:"
        docker compose logs --tail=20 trading
    else
        log_warn "Trading bot is not running"
    fi
}

# ==========================================================================
# Update
# ==========================================================================

update_code() {
    log_info "Updating code..."

    # Stash any local changes
    git stash

    # Pull latest
    git pull origin main

    # Rebuild and restart
    docker compose up -d --build

    log_success "Update complete"
}

# ==========================================================================
# Backup / Restore
# ==========================================================================

backup_database() {
    local backup_file="backup_$(date +%Y%m%d_%H%M%S).sql"

    log_info "Backing up database to $backup_file..."

    docker compose exec -T postgres pg_dump -U postgres trading_bot > "$backup_file"

    # Compress
    gzip "$backup_file"

    log_success "Backup created: ${backup_file}.gz"
}

restore_database() {
    local backup_file="$1"

    if [[ -z "$backup_file" ]]; then
        log_error "Usage: $0 restore <backup_file.sql.gz>"
        exit 1
    fi

    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        exit 1
    fi

    log_warn "This will overwrite the current database!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Restoring database from $backup_file..."

        if [[ "$backup_file" == *.gz ]]; then
            gunzip -c "$backup_file" | docker compose exec -T postgres psql -U postgres trading_bot
        else
            docker compose exec -T postgres psql -U postgres trading_bot < "$backup_file"
        fi

        log_success "Database restored"
    else
        log_info "Restore cancelled"
    fi
}

# ==========================================================================
# Health Check
# ==========================================================================

health_check() {
    log_info "Running health check..."

    local errors=0

    # Check Docker
    if ! docker info &> /dev/null; then
        log_error "Docker is not running"
        ((errors++))
    else
        log_success "Docker is running"
    fi

    # Check containers
    for service in postgres redis trading; do
        if docker compose ps | grep -q "${service}.*Up"; then
            log_success "$service is running"
        else
            log_error "$service is not running"
            ((errors++))
        fi
    done

    # Check database connection
    if docker compose exec -T postgres pg_isready -U postgres &> /dev/null; then
        log_success "PostgreSQL is accepting connections"
    else
        log_error "PostgreSQL is not accepting connections"
        ((errors++))
    fi

    # Check Redis connection
    if docker compose exec -T redis redis-cli ping | grep -q "PONG"; then
        log_success "Redis is responding"
    else
        log_error "Redis is not responding"
        ((errors++))
    fi

    echo ""
    if [[ $errors -eq 0 ]]; then
        log_success "All health checks passed!"
    else
        log_error "$errors health check(s) failed"
        return 1
    fi
}

# ==========================================================================
# Main
# ==========================================================================

show_help() {
    echo "Grid Trading Bot - Deployment Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  install     Install Docker and dependencies"
    echo "  setup       Configure environment and start services"
    echo "  start       Start all services"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  logs        Show trading bot logs (use: logs [service] [lines])"
    echo "  status      Check service status"
    echo "  health      Run health checks"
    echo "  update      Pull latest code and restart"
    echo "  backup      Backup database"
    echo "  restore     Restore database (use: restore <file>)"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 install              # First-time setup"
    echo "  $0 setup                # Configure and start"
    echo "  $0 logs trading 200     # Show last 200 lines of trading logs"
    echo "  $0 backup               # Create database backup"
}

main() {
    local command="${1:-help}"

    case "$command" in
        install)
            check_root
            install_dependencies
            install_docker
            install_compose
            ;;
        setup)
            setup_env || exit 1
            setup_directories
            start_services
            ;;
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        logs)
            show_logs "$2" "$3"
            ;;
        status)
            show_status
            ;;
        health)
            health_check
            ;;
        update)
            update_code
            ;;
        backup)
            backup_database
            ;;
        restore)
            restore_database "$2"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
