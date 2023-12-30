### For Docker 
```
# For latest version
docker compose up -d 

# For old version
docker-compose up -d 
```

### For Kubernetes
#### Reference: https://artifacthub.io/packages/helm/milvus/milvus
```
helm repo add milvus https://milvus-io.github.io/milvus-helm/
helm repo update

helm fetch milvus/milvus
tar zxvf milvus-version.tgz
cd milvus/

vim values.yaml
cluster:
  enabled: false

standalone:
  persistence:
    persistentVolumeClaim:
      storageClass: "alicloud-disk-wffc"
      size: 20Gi
  
attu:
  enabled: true
  ingress:
    enabled: true
    hosts:
      - vdb.chatbot.mowenlab.cn
      
minio:
  mode: standalone
  persistence:
    storageClass: "alicloud-disk-wffc"
    accessMode: ReadWriteOnce
    size: 20Gi

etcd:
  replicaCount: 1
  persistence:
    storageClass: "alicloud-disk-wffc"
    accessMode: ReadWriteOnce
    size: 10Gi
  
pulsar:
  enabled: false
  
  
# Install:
helm install --create-namespace -n milvus milvus -f values.yaml .

# Uninstall
helm delete -n milvus mvilus
```