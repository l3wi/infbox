<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Technical Specification: Multi-GPU Code-Aware LLM Inference Stack

## Executive Overview

Spin-up should require only three user actions:

1. Provision an Ubuntu 22.04 server with ≥ 1 modern CUDA GPU (A100 40 GB or 4090 24 GB for single-GPU dev; H100/A100 cluster for prod).
2. `git clone` this repo; run `./bootstrap.sh`.
3. `make start` – the script pulls Qwen3-Coder-480B, builds LMCache + Cake, and launches vLLM.

The resulting stack gives developers an SSH (Cursor, VS Code Remote-SSH, or CLI) workspace whose whole codebase is indexed into KV cache respecting `.gitignore`, continuously refreshed by inotify, and served through a single OpenAI-compatible vLLM endpoint.

A single repo now supports two **presets**:

- `dev32`: Qwen2.5-Coder-32B for functional tests on one 24 GB–48 GB GPU
- `prod480`: Qwen3-Coder-480B for full-scale deployment on ≥4 A100 80 GB

Both share identical orchestration, so graduating from `dev32` to `prod480` is a flag change—not a rewrite.

## 1 Architecture Overview

A Git-synced code workspace is watched by an inotify daemon. File deltas are hashed, deduplicated against `.gitignore`, and prefetched into LMCache (FP16). Cake then streams missing KV chunks while vLLM computes the rest, serving an OpenAI-compatible endpoint.

![System architecture of configurable vLLM + LMCache + Cake stack using Qwen2.5-Coder-32B.](https://user-gen-media-assets.s3.amazonaws.com/gpt4o_images/b3641229-f8c2-4f04-a948-f973a263e728.png)

System architecture of configurable vLLM + LMCache + Cake stack using Qwen2.5-Coder-32B.

## 2 Component Matrix

| Container       | Image                          | Key Args / Env                                                                                           | Configurable?               |
| :-------------- | :----------------------------- | :------------------------------------------------------------------------------------------------------- | :-------------------------- |
| **vllm**        | `ghcr.io/vllm/vllm-openai:0.8` | `--model $MODEL_NAME` `--tensor-parallel-size $GPU_COUNT` `--kv-cache-dtype fp16` `--quantization $PREC` | MODEL_NAME, GPU_COUNT, PREC |
| **lmcache**     | `lmcache/lmcache:0.3.1`        | `LMCACHE_CHUNK_SIZE=256` `LMCACHE_LOCAL_CPU=true` `LMCACHE_MAX_LOCAL_CPU_SIZE=$CPU_GB`                   | CPU_GB, remote URL          |
| **cake**        | same                           | `CAKE_MODE=adaptive` `CAKE_MIN_RATIO=0.2`                                                                | off                         |
| **watcher**     | `python:3.12-alpine`           | `WATCH_DIR=/workspace` `IGNORE_FILE=.gitignore`                                                          | off                         |
| **git-sync**    | `alpine/git`                   | GITHUB_TOKEN via secret                                                                                  | off                         |
| **caddy** (TLS) | `caddy:2`                      | auto HTTPS                                                                                               | domain                      |

All tunables surface through `.env`. Example:

```dotenv
# .env.dev32
MODEL_NAME=Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4
PREC=int4
GPU_COUNT=1
CPU_GB=10
```

Switching to production is a single file:

```dotenv
# .env.prod480
MODEL_NAME=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
PREC=fp8
GPU_COUNT=4
CPU_GB=40
```

## 3 Quantization Rules

1 Always keep **KV cache FP16**—never lower.
2 Weights:
 - **FP8** if ≥40 GB VRAM per shard
 - **INT4/INT6** for 24 GB cards (dev mode)
3 Set `--gpu-memory-utilization` lower (0.75) when KV cache FP16 inflates usage.

![VRAM requirements by precision level for Qwen2.5-Coder-32B.](https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/ec6a79af8bec5ebbb605471f3b351b4d/b42a629a-08c2-46aa-bef9-b547f1f1b9c6/8e66f066.png)

VRAM requirements by precision level for Qwen2.5-Coder-32B.

## 4 docker-compose.yml (base)

```yaml
version: "3.9"

services:
  vllm:
    image: ghcr.io/vllm/vllm-openai:0.8
    command: >
      vllm serve ${MODEL_NAME}
      --tensor-parallel-size ${GPU_COUNT:-1}
      --kv-cache-dtype fp16
      --quantization ${PREC:-int4}
      --kv-transfer-config
      '{"kv_connector":"LMCacheConnectorV1Dynamic","kv_role":"kv_both","kv_connector_module_path":"lmcache.integration.vllm.lmcache_connector_v1"}'
      --max-model-len 131072
      --gpu-memory-utilization ${GPU_UTIL:-0.8}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes: [models:/models, cache:/cache, workspace:/workspace]
    ports: ["8000:8000"]
    env_file: [.env] # override with -f docker-compose.prod.yml

  lmcache:
    image: lmcache/lmcache:0.3.1
    environment:
      - LMCACHE_USE_EXPERIMENTAL=True
      - LMCACHE_CHUNK_SIZE=256
      - LMCACHE_LOCAL_CPU=True
      - LMCACHE_MAX_LOCAL_CPU_SIZE=${CPU_GB:-10}
      - LMCACHE_LOCAL_DISK=file:///cache/disk
      - LMCACHE_MAX_LOCAL_DISK_SIZE=50
    volumes: [cache:/cache]
    ports: ["8100:8100"]

  cake:
    image: lmcache/lmcache:0.3.1
    command: python -m lmcache.experimental.cake_loader --listen 8200
    depends_on: [lmcache]
    ports: ["8200:8200"]

  watcher:
    image: python:3.12-alpine
    command: python /scripts/watch_codebase.py
    volumes:
      - workspace:/workspace
      - ./scripts:/scripts
    environment:
      - WATCH_DIR=/workspace
      - IGNORE_FILE=.gitignore
      - VLLM_ENDPOINT=http://vllm:8000

  caddy:
    image: caddy:2
    ports: ["443:443", "80:80"]
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile

volumes:
  models: {}
  cache: {}
  workspace: {}
```

Override envs with `docker-compose --env-file .env.dev32 up -d`.

## 5 Bootstrap Flow

1 `./bootstrap.sh`
 - installs NVIDIA drivers, Docker, Compose
 - clones repo \& writes `.env.dev32`
2 `make start-dev` → pulls model, launches stack
3 Developer SSHes via Cursor; watcher streams edits into LMCache; vLLM answers IDE requests.

## 6 Task List

| \#  | Deliverable                          | Est (d) |
| :-- | :----------------------------------- | :------ |
| 1   | Terraform module for single-GPU VM   | 1       |
| 2   | Bootstrap script with env presets    | 1       |
| 3   | Compose base + prod override         | 1       |
| 4   | Int4 GPTQ model fetch utility        | 1       |
| 5   | File-watcher with gitignore parsing  | 2       |
| 6   | Cake loader param tuning             | 1       |
| 7   | Grafana dashboard (GPU, KV hit-rate) | 1       |
| 8   | Docs: README + upgrade guide         | 1       |

_Total: 9 days._

## 7 Future Flags

- `PREC=awq` once AWQ kernels land in vLLM 0.9.
- Enable `LMCACHE_ENABLE_P2P=true` for multi-node GPU clusters.
- Add Redis offload backend for cheap long-tail KV.

Updating the specification with these changes lets you start cheap on a **single RTX 4090** and flip to full-blown multi-A100 by editing one env file—no compose edits, no code rewrites.

<div style="text-align: center">⁂</div>

[^1]: https://www.reddit.com/r/LocalLLaMA/comments/1gp4g8a/hardware_requirements_to_run_qwen_25_32b/
[^2]: https://www.runpod.io/ai-faq/what-gpu-is-required-to-run-the-qwen-qwq-32b-model-from-hugging-face
[^3]: https://github.com/inferless/qwen2.5-coder-32b-instruct
[^4]: https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct
[^5]: https://wenku.csdn.net/answer/7eynirgd0p
[^6]: https://www.koyeb.com/deploy/qwen-2-5-coder-32b-instruct
[^7]: https://huggingface.co/Qwen/Qwen2.5-Coder-32B
[^8]: https://www.reddit.com/r/LocalLLaMA/comments/1greuto/gpu_inference_vram_calc_for_qwen25coder_32b_need/
[^9]: https://blog.csdn.net/jycjyc/article/details/145024321
[^10]: https://prompt.16x.engineer/blog/qwen-25-coder-32b-coding
[^11]: https://www.youtube.com/watch?v=OaRhIwr_jQc
[^12]: https://www.reddit.com/r/LocalLLaMA/comments/1gxs34g/comment_your_qwen_coder_25_setup_ts_here/
[^13]: https://www.byteplus.com/en/topic/417612
[^14]: https://www.hardware-corner.net/llm-database/Qwen/
[^15]: https://github.com/inferless/Qwen2.5-Coder-32B-Instruct
[^16]: https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct/discussions/28
[^17]: https://www.databasemart.com/ai/qwen
[^18]: https://github.com/inferless/Qwen2.5-Coder-32B-Instruct/blob/main/README.md
[^19]: https://thinktank.ottomator.ai/t/new-qwen-2-5-coder-32b-absolutely-crushing-it/529
[^20]: https://apxml.com/posts/gpu-system-requirements-qwen-models
[^21]: https://docs.vllm.ai/en/stable/serving/env_vars.html
[^22]: https://centlinux.com/vllm-docker/
[^23]: https://clear.ml/docs/latest/docs/webapp/applications/apps_model_deployment/
[^24]: https://docs.vllm.ai/en/v0.4.3/serving/env_vars.html
[^25]: https://www.reddit.com/r/LocalLLaMA/comments/1fdvbhi/current_best_way_to_have_multiple_models_serving/
[^26]: https://docs.zenml.io/stacks/stack-components/model-deployers/vllm
[^27]: https://docs.vllm.ai/en/v0.6.4/serving/env_vars.html
[^28]: https://docs.vllm.ai/en/stable/deployment/docker.html
[^29]: https://www.alibabacloud.com/help/en/ack/cloud-native-ai-suite/user-guide/deploy-a-vllm-inference-application
[^30]: https://vllm-ascend.readthedocs.io/en/latest/user_guide/configuration/env_vars.html
[^31]: https://github.com/vllm-project/vllm/issues/16065
[^32]: https://docs.vllm.ai/en/stable/getting_started/quickstart.html
[^33]: https://www.bookstack.cn/read/vllm-0.7.0-en/35b54472a41270ff.md
[^34]: https://collabnix.com/how-vllm-and-docker-are-changing-the-game-for-llm-deployments/
[^35]: https://ploomber.io/blog/vllm-deploy/
[^36]: https://www.bookstack.cn/read/vllm-0.6.2-en/6fc8528f6d039cc7.md
[^37]: https://github.com/vllm-project/vllm/issues/299
[^38]: https://docs.vllm.ai/en/latest/serving/distributed_serving.html
[^39]: https://github.com/vllm-project/vllm/issues/4407
[^40]: https://discuss.vllm.ai/t/does-vllm-support-deploy-multiple-docker-instance-on-one-gpu/657
[^41]: https://docs.lmcache.ai/api_reference/configurations.html
[^42]: https://cake-contrib.github.io/Cake.Recipe/docs/usage/getting-started-with-cake-recipe
[^43]: https://overcast.blog/multi-environment-deployments-with-docker-a-guide-890e193191b6
[^44]: https://docs.lmcache.ai/kv_cache/redis.html
[^45]: https://github.com/cnizzardini/cakephp-preloader
[^46]: https://stackoverflow.com/questions/72979713/docker-compose-can-i-set-priority-of-multiple-environment-files
[^47]: https://docs.vllm.ai/en/latest/examples/others/lmcache.html
[^48]: https://api.cakephp.org/3.0/class-Cake.Core.Configure.html
[^49]: https://github.com/docker/compose/issues/7326
[^50]: https://github.com/LMCache/lmcache-tests/blob/main/configs.py
[^51]: https://cakebuild.net/docs/writing-builds/preprocessor-directives/load
[^52]: https://forums.docker.com/t/combining-docker-compose-override-files-with-mutliple-env-files/119895
[^53]: https://docs.lmcache.ai/configuration/v0/v0_config.html
[^54]: https://cakebuild.net/docs/integrations/editors/rider/run-configurations
[^55]: https://stackoverflow.com/questions/51610294/docker-compose-with-multiple-enviroment-variables/51611269
[^56]: https://docs.lmcache.ai/configuration/v1/v1_config.html
[^57]: https://www.npmjs.com/package/@cake-hub/core
[^58]: https://stackoverflow.com/questions/29062522/different-env-file-but-same-yml-with-docker-compose
[^59]: https://kserve.github.io/website/0.15/modelserving/v1beta1/llm/huggingface/kv_cache_offloading/
[^60]: https://stackoverflow.com/questions/36650324/cakephp-3-how-to-load-components
