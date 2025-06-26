#!/bin/bash

# SMPC System Management Script

case "$1" in
    "build")
        echo "Building SMPC system..."
        docker-compose build
        ;;
    "start")
        echo "Starting SMPC system..."
        docker-compose up -d
        echo "Web interface available at: http://localhost:8080"
        ;;
    "stop")
        echo "Stopping SMPC system..."
        docker-compose down
        ;;
    "restart")
        echo "Restarting SMPC system..."
        docker-compose down
        docker-compose up -d
        ;;
    "rebuild")
        echo "Rebuilding SMPC system from scratch..."
        echo "Stopping containers..."
        docker-compose down
        echo "Removing existing containers and images..."
        docker-compose down --rmi all --volumes --remove-orphans
        echo "Building fresh images..."
        docker-compose build --no-cache
        echo "Starting system with new build..."
        docker-compose up -d
        echo "✅ Rebuild complete! Web interface available at: http://localhost:8080"
        ;;
    "logs")
        if [ -z "$2" ]; then
            echo "Showing logs for all parties (orchestrator, party1, party2, party3)..."
            docker-compose logs -f orchestrator party1 party2 party3
        else
            docker-compose logs -f "$2"
        fi
        ;;
    "status")
        echo "SMPC System Status:"
        docker-compose ps
        ;;
    "clean")
        echo "Cleaning up SMPC system..."
        docker-compose down
        docker system prune -f
        echo "Cleaned up containers and images"
        ;;
    "shell")
        if [ -z "$2" ]; then
            echo "Usage: $0 shell <service>"
            echo "Available services: orchestrator, party1, party2, party3"
        else
            docker-compose exec "$2" bash
        fi
        ;;
    "validate")
        python3 validate_setup.py
        ;;
    *)
        echo "SMPC System Management"
        echo "Usage: $0 {build|start|stop|restart|rebuild|logs|status|clean|shell|validate}"
        echo ""
        echo "Commands:"
        echo "  build     - Build the Docker images"
        echo "  start     - Start the system in background"
        echo "  stop      - Stop the system"
        echo "  restart   - Restart the system"
        echo "  rebuild   - Complete rebuild with fresh images and containers"
        echo "  logs      - Show logs for all parties (add service name for specific logs)"
        echo "  status    - Show system status"
        echo "  clean     - Clean up containers and images"
        echo "  shell     - Access container shell (specify service)"
        echo "  validate  - Validate setup and check requirements"
        echo ""
        echo "Examples:"
        echo "  $0 logs orchestrator"
        echo "  $0 shell party1"
        echo "  $0 rebuild"
        ;;
esac
