# Phase 9: Docker & Kubernetes Deployment Guide

## Overview

This guide covers deploying the NEO Hybrid AI Trading System using Docker containers and Kubernetes orchestration.

## Prerequisites

- Docker Desktop or Docker Engine (v20.10+)
- Docker Compose (v2.0+)
- Kubernetes cluster (1.24+) or minikube for local development
- kubectl CLI configured with cluster access
- At least 4GB RAM available for containerized services

## Local Development: Docker Compose

### Quick Start

```bash
cd c:\Users\Pgiov\OneDrive\Documents\Custom programs\Neo

# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f neo-ai-service

# Stop services
docker-compose down

# Remove all data and start fresh
docker-compose down -v
```

### Service Overview

| Service | Port | Purpose |
|---------|------|---------|
| neo-ai-service | 8000 | Python AI trading engine |
| postgres-db | 5432 | Trading data persistence |
| redis-cache | 6379 | In-memory caching and state |
| prometheus | 9090 | Metrics and monitoring |

### Environment Variables

Configure in `docker-compose.yml`:
- `PYTHONUNBUFFERED=1`: Real-time log output
- `NEO_ENV`: Environment (development/staging/production)
- `LOG_LEVEL`: Logging verbosity (DEBUG/INFO/WARNING/ERROR)

### Health Checks

All services include health checks. View status:

```bash
docker-compose ps
```

To manually check service health:

```bash
docker exec neo-ai-service python -c "from python_ai.orchestration_core import get_orchestrator_integration; print('Healthy')"
docker exec neo-postgres pg_isready -U neo_user
docker exec neo-redis redis-cli ping
```

## Kubernetes Deployment

### Prerequisites Setup

```bash
# Create namespace
kubectl apply -f k8s-deployment.yaml

# Verify namespace created
kubectl get namespaces | grep neo-trading
```

### Build and Push Docker Image

For production deployment to a registry:

```bash
# Build production image
docker build --target production -t your-registry/neo-ai-service:1.0.0 .

# Push to registry
docker push your-registry/neo-ai-service:1.0.0

# Update image in k8s-deployment.yaml:
# image: your-registry/neo-ai-service:1.0.0
```

### Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f k8s-deployment.yaml

# Verify deployment
kubectl get deployments -n neo-trading
kubectl get pods -n neo-trading
kubectl get svc -n neo-trading

# Check pod status
kubectl describe pod <pod-name> -n neo-trading

# View logs
kubectl logs -f deployment/neo-ai-service -n neo-trading
kubectl logs -f deployment/neo-ai-service -n neo-trading --previous  # Previous pod logs
```

### Scale Deployment

```bash
# Manual scaling
kubectl scale deployment neo-ai-service --replicas=5 -n neo-trading

# Check HPA status
kubectl get hpa -n neo-trading
kubectl describe hpa neo-ai-service-hpa -n neo-trading

# View autoscaling events
kubectl get events -n neo-trading --sort-by='.lastTimestamp'
```

### Access Services

```bash
# Port-forward to local machine
kubectl port-forward svc/neo-ai-service 8000:8000 -n neo-trading

# Access via LoadBalancer (if available)
kubectl get svc neo-ai-service-lb -n neo-trading
LOAD_BALANCER_IP=$(kubectl get svc neo-ai-service-lb -n neo-trading -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "Access service at: http://$LOAD_BALANCER_IP"

# SSH into pod for debugging
kubectl exec -it <pod-name> -n neo-trading -- bash
```

### Database Access

```bash
# Connect to PostgreSQL
kubectl port-forward svc/neo-postgres 5432:5432 -n neo-trading

# Use psql client
psql -h localhost -U neo_user -d neo_trading_db

# Get credentials from secrets
kubectl get secret neo-db-credentials -n neo-trading -o jsonpath='{.data.postgres-username}' | base64 -d
kubectl get secret neo-db-credentials -n neo-trading -o jsonpath='{.data.postgres-password}' | base64 -d
```

## Configuration Management

### ConfigMaps

Store non-sensitive configuration:

```bash
# Create from file
kubectl create configmap neo-config --from-file=config/ -n neo-trading

# View ConfigMap
kubectl get configmap neo-config -n neo-trading -o yaml

# Update ConfigMap
kubectl patch configmap neo-config -n neo-trading --type merge -p '{"data":{"LOG_LEVEL":"DEBUG"}}'
```

### Secrets

Store sensitive data:

```bash
# Create secret
kubectl create secret generic neo-db-credentials \
  --from-literal=postgres-username=neo_user \
  --from-literal=postgres-password=secure_password \
  -n neo-trading

# View secret (base64 encoded)
kubectl get secret neo-db-credentials -n neo-trading -o yaml

# Update secret
kubectl delete secret neo-db-credentials -n neo-trading
kubectl create secret generic neo-db-credentials \
  --from-literal=postgres-username=neo_user \
  --from-literal=postgres-password=new_password \
  -n neo-trading
```

## Monitoring & Observability

### Prometheus Metrics

Access Prometheus dashboard:
```bash
kubectl port-forward svc/prometheus 9090:9090 -n neo-trading
# Visit: http://localhost:9090
```

### Pod Logs and Events

```bash
# Stream deployment logs
kubectl logs -f deployment/neo-ai-service -n neo-trading

# View recent events
kubectl describe deployment neo-ai-service -n neo-trading

# Export logs for analysis
kubectl logs deployment/neo-ai-service -n neo-trading > deployment_logs.txt
```

### Performance Monitoring

```bash
# View resource usage
kubectl top node
kubectl top pod -n neo-trading

# Monitor HPA scaling decisions
kubectl get hpa neo-ai-service-hpa -n neo-trading --watch
```

## Maintenance Operations

### Rolling Updates

```bash
# Update image without downtime (controlled by strategy)
kubectl set image deployment/neo-ai-service \
  neo-ai-service=your-registry/neo-ai-service:2.0.0 \
  -n neo-trading --record

# Monitor rollout
kubectl rollout status deployment/neo-ai-service -n neo-trading

# Rollback if needed
kubectl rollout undo deployment/neo-ai-service -n neo-trading
kubectl rollout history deployment/neo-ai-service -n neo-trading
```

### Data Persistence

```bash
# Check PVC status
kubectl get pvc -n neo-trading

# View PersistentVolume details
kubectl describe pvc neo-data-pvc -n neo-trading

# Backup PVC data
kubectl get pvc neo-data-pvc -n neo-trading -o yaml > pvc_backup.yaml
```

### Drain Node for Maintenance

```bash
# Safely drain a node (reschedules pods)
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# Uncordon when ready
kubectl uncordon <node-name>
```

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n neo-trading

# View pod logs
kubectl logs <pod-name> -n neo-trading

# Get previous logs if pod crashed
kubectl logs <pod-name> -n neo-trading --previous
```

### Service Not Accessible

```bash
# Check service endpoints
kubectl get endpoints neo-ai-service -n neo-trading

# Verify DNS resolution
kubectl run -it --rm debug --image=busybox --restart=Never -- \
  nslookup neo-ai-service.neo-trading.svc.cluster.local

# Test connectivity to pod
kubectl run -it --rm debug --image=busybox --restart=Never -- \
  wget -O- http://neo-ai-service:8000
```

### Database Connection Issues

```bash
# Check PostgreSQL pod
kubectl describe pod -l app=postgres -n neo-trading

# Test database connectivity from application pod
kubectl exec -it <neo-ai-pod> -n neo-trading -- python -c \
  "import psycopg2; conn = psycopg2.connect('postgresql://...'); print('Connected')"
```

### Resource Constraints

```bash
# Check node resources
kubectl describe nodes

# Check pod resource limits
kubectl get pod <pod-name> -n neo-trading -o yaml | grep -A 10 resources

# Increase pod resources if needed
kubectl set resources deployment neo-ai-service \
  -n neo-trading \
  --limits=cpu=1,memory=2Gi \
  --requests=cpu=500m,memory=1Gi
```

## Security Best Practices

1. **Non-root User**: Pods run as UID 1000 (not root)
2. **Resource Limits**: CPU and memory limits prevent resource exhaustion
3. **Network Policies**: Service-to-service communication via ClusterIP
4. **Secrets Management**: Database credentials stored as Kubernetes Secrets
5. **Pod Disruption Budgets**: Minimum availability during voluntary disruptions
6. **RBAC**: Limited permissions via ServiceAccount and Roles

## Performance Tuning

### HPA Configuration

Auto-scaling adjusts replicas based on:
- CPU utilization > 70%
- Memory utilization > 80%
- Min replicas: 2 (for HA)
- Max replicas: 10 (cost control)

Modify in `k8s-deployment.yaml`:
```yaml
spec:
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - resource:
      name: cpu
      target:
        averageUtilization: 70
```

### Resource Allocation

Current configuration:
- **Requests**: 250m CPU, 512Mi Memory (guaranteed minimum)
- **Limits**: 500m CPU, 1Gi Memory (maximum allowed)

Adjust based on workload profiling.

## Cleanup

```bash
# Delete entire deployment
kubectl delete namespace neo-trading

# Delete specific resources
kubectl delete deployment neo-ai-service -n neo-trading
kubectl delete pvc neo-data-pvc -n neo-trading

# Local cleanup
docker-compose down -v
docker rmi neo-ai-service:latest
```

## Next Steps

- **Phase 10**: Code review, performance profiling, and documentation
- Monitor metrics in Prometheus after deployment
- Set up CI/CD pipeline for automated deployments
- Configure backup strategy for PostgreSQL data
- Implement log aggregation (ELK stack, Splunk, etc.)

