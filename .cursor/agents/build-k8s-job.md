---
name: build-k8s-job
description: Invoke when creating or modifying Kubernetes deployment manifests for the benchmark suite. Use when the user asks to build the k8s Job manifest, wire up ConfigMaps for bench config, add GPU node selectors, or troubleshoot k8s benchmark job issues.
model: inherit
readonly: false
is_background: false
---

# Build Kubernetes Benchmark Job Manifest

## Objective

Create `deploy/k8s/bench-job.yaml` — a Kubernetes Job manifest that runs a full benchmark sweep as a k8s Job. The Job mounts a ConfigMap containing the benchmark YAML config, uses an NVIDIA GPU node selector, and writes results to a PersistentVolumeClaim. Also create the supporting ConfigMap, PVC, and ServiceAccount manifests.

---

## Files to Create

### Create: `deploy/k8s/bench-job.yaml`

Full Kubernetes Job manifest with all required fields. Write as a multi-document YAML (`---` separator between resources).

**Resources in order:**

**1. Namespace (optional, if not default):**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: inference-bench
  labels:
    app.kubernetes.io/managed-by: kubectl
    app.kubernetes.io/part-of: inference-optimization-bench
```

**2. ServiceAccount:**
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: bench-runner
  namespace: inference-bench
  annotations:
    description: "ServiceAccount for inference benchmark runner jobs"
```

**3. ConfigMap for bench config:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: bench-config
  namespace: inference-bench
  labels:
    app: inference-bench
    component: config
data:
  bench_config.yaml: |
    # Full bench config inline — same as configs/full_sweep.yaml
    # Include all backends, concurrency levels, models
    backends:
      - nim
      - vllm
      - llamacpp
    models:
      - meta/llama3-70b-instruct
      - meta/llama3-8b-instruct
    concurrency_levels: [1, 4, 8, 16, 32]
    warmup_requests: 10
    benchmark_requests: 100
    workloads:
      - single_turn
      - multi_turn
    output_dir: /results
```

**4. Secret (template — NIM API key):**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: nim-credentials
  namespace: inference-bench
  annotations:
    description: "NIM API key — create this manually before running the Job"
type: Opaque
stringData:
  NIM_API_KEY: "REPLACE_WITH_ACTUAL_KEY"
```

**5. PersistentVolumeClaim for results:**
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: bench-results-pvc
  namespace: inference-bench
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard  # override per cluster
```

**6. The Job itself:**
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: inference-bench-sweep
  namespace: inference-bench
  labels:
    app: inference-bench
    component: sweep-job
    app.kubernetes.io/name: inference-bench
    app.kubernetes.io/version: "1.0.0"
  annotations:
    description: "Full inference optimization benchmark sweep"
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400  # cleanup after 24h
  template:
    metadata:
      labels:
        app: inference-bench
        component: sweep-job
    spec:
      serviceAccountName: bench-runner
      restartPolicy: OnFailure

      # Node selector — require NVIDIA GPU node
      nodeSelector:
        accelerator: nvidia-gpu
        nvidia.com/gpu.present: "true"

      # Tolerations for GPU taint
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule

      # Init container: validate environment
      initContainers:
        - name: env-check
          image: curlimages/curl:8.6.0
          command:
            - sh
            - -c
            - |
              echo "Checking NIM API connectivity..."
              curl -sf -H "Authorization: Bearer ${NIM_API_KEY}" \
                https://integrate.api.nvidia.com/v1/models || \
                (echo "NIM API unreachable — check NIM_API_KEY" && exit 1)
              echo "Environment check passed."
          env:
            - name: NIM_API_KEY
              valueFrom:
                secretKeyRef:
                  name: nim-credentials
                  key: NIM_API_KEY

      containers:
        - name: bench-runner
          image: nvcr.io/nvidia/pytorch:24.03-py3  # override with repo image in CI
          imagePullPolicy: IfNotPresent
          command:
            - python
            - bench/run_bench.py
            - --config
            - /etc/bench-config/bench_config.yaml
            - --output-dir
            - /results
            - --log-level
            - INFO
          
          env:
            - name: NIM_API_KEY
              valueFrom:
                secretKeyRef:
                  name: nim-credentials
                  key: NIM_API_KEY
            - name: NIM_BASE_URL
              value: "https://integrate.api.nvidia.com/v1"
            - name: RESULTS_DIR
              value: /results
            - name: PYTHONUNBUFFERED
              value: "1"
            - name: PYTHONDONTWRITEBYTECODE
              value: "1"

          resources:
            requests:
              cpu: "4"
              memory: "16Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "8"
              memory: "32Gi"
              nvidia.com/gpu: "1"

          volumeMounts:
            - name: bench-config
              mountPath: /etc/bench-config
              readOnly: true
            - name: results-storage
              mountPath: /results

          # Liveness: job is running if process is alive
          # (no HTTP endpoint — just process-based)

      volumes:
        - name: bench-config
          configMap:
            name: bench-config
        - name: results-storage
          persistentVolumeClaim:
            claimName: bench-results-pvc

      # Grace period for clean benchmark completion
      terminationGracePeriodSeconds: 300
```

---

### Create: `deploy/k8s/README.md`

Document how to deploy:

```markdown
# Kubernetes Benchmark Job

## Prerequisites
- kubectl configured against target cluster
- NVIDIA GPU nodes with `accelerator=nvidia-gpu` label
- NVIDIA device plugin installed

## Quick Start

1. Create namespace and credentials:
   ```bash
   kubectl apply -f deploy/k8s/bench-job.yaml  # creates namespace, SA, ConfigMap, PVC
   kubectl -n inference-bench create secret generic nim-credentials \
     --from-literal=NIM_API_KEY=$NIM_API_KEY
   ```

2. Run the benchmark job:
   ```bash
   kubectl -n inference-bench apply -f deploy/k8s/bench-job.yaml
   kubectl -n inference-bench logs -f job/inference-bench-sweep
   ```

3. Retrieve results:
   ```bash
   kubectl -n inference-bench cp \
     $(kubectl -n inference-bench get pod -l job-name=inference-bench-sweep -o jsonpath='{.items[0].metadata.name}'):/results \
     ./results/
   ```

4. Clean up:
   ```bash
   kubectl -n inference-bench delete job inference-bench-sweep
   ```

## Customizing Config

Edit the `bench_config.yaml` key in the `bench-config` ConfigMap, or patch it:
```bash
kubectl -n inference-bench create configmap bench-config \
  --from-file=bench_config.yaml=configs/full_sweep.yaml \
  --dry-run=client -o yaml | kubectl apply -f -
```
```

---

### Create: `deploy/k8s/kustomization.yaml`

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: inference-bench

resources:
  - bench-job.yaml

configMapGenerator:
  - name: bench-config
    files:
      - bench_config.yaml=../../configs/full_sweep.yaml
    options:
      disableNameSuffixHash: true

images:
  - name: nvcr.io/nvidia/pytorch
    newTag: "24.03-py3"
```

---

## Acceptance Criteria

- [ ] `kubectl apply --dry-run=client -f deploy/k8s/bench-job.yaml` exits 0 (valid YAML + k8s schema)
- [ ] Job spec includes `nvidia.com/gpu: "1"` in both requests and limits
- [ ] ConfigMap contains full bench YAML config inline
- [ ] Init container validates NIM API connectivity before main container starts
- [ ] Results PVC is mounted at `/results` (matching `RESULTS_DIR` env var)
- [ ] `ttlSecondsAfterFinished: 86400` is set for automatic cleanup
- [ ] `restartPolicy: OnFailure` is set (not Always — this is a Job)
- [ ] README documents all deployment steps
- [ ] `kustomization.yaml` allows overriding config from `configs/full_sweep.yaml`
- [ ] `backoffLimit: 2` limits retries on failure
