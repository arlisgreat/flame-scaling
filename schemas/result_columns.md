# Result ledger schema

每行对应一个已聚合的 `(run, identity, region, yaw_bin, sequence_family, metric)` 结果。

| 字段 | 类型 | 含义 |
|---|---|---|
| `run_id` | string | 由实验矩阵稳定生成 |
| `stage` | enum | `oracle/main/observation/role/apr` |
| `status` | enum | `planned/running/completed/failed/excluded` |
| `failure_reason` | string | 非 completed 时必填 |
| `git_commit` | string | 代码版本 |
| `dataset_version` | string | 数据 manifest hash |
| `split_id` | string | 嵌套身份 split |
| `identity_id_hash` | string | 不保存可识别原始 ID |
| `method` | string | hard/free/APR/角色干预 |
| `g_state,c_state,a_state,p_state` | string | 四角色状态 |
| `n_id,n_obs` | int | 身份数与每身份观测数 |
| `capacity` | string | S/M/L |
| `total_params,active_params` | int | 总参数与 gate 激活参数 |
| `gaussian_count` | int | 表示预算 |
| `train_tokens,train_steps` | int | 学习预算 |
| `gpu_hours,peak_mem_gb` | float | 实际资源 |
| `seed` | int | 配对随机种子 |
| `identity_group` | string | test identity 聚合组 |
| `region` | string | 冻结的 GT/独立区域 |
| `yaw_bin` | string | `0_30/30_60/gt_60/all` |
| `sequence_family` | string | tongue/mouth/eyes/jaw/emotion/speech/static 等 |
| `metric` | string | lpips/dists/psnr/depth/normal/mv_consistency 等 |
| `value` | float | identity 级聚合值 |
| `n_frames,n_cameras` | int | 聚合样本量，仅作透明度，不当独立样本 |
| `checkpoint_rule` | string | validation selection 规则 |

原始 per-frame 结果可另存 parquet，但统计 bootstrap 以 identity/sequence cluster 为单位。

