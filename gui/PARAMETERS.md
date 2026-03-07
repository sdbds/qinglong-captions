# GUI 参数映射文档

本文档记录所有 PowerShell 脚本参数与 GUI 的映射关系。

## 缓存脚本参数 (step2_cache)

### 基础参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| dataset_config | path | TOML 配置文件 | path_selector |
| vae | path | VAE 模型路径 | path_selector |
| vae_dtype | select | VAE 数据类型 | select [float32/float16/bfloat16] |
| device | select | 运行设备 | select [cuda/cpu] |
| batch_size | number | 批处理大小 | number_input |
| num_workers | number | 数据加载线程数 | number_input |
| skip_existing | checkbox | 跳过已存在的缓存 | checkbox |

### 架构特定参数

#### HunyuanVideo / HV 1.5 / FramePack / Kontext
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| vae_chunk_size | number | VAE chunk size | number_input |
| vae_tiling | checkbox | 启用 VAE tiling | checkbox |
| vae_spatial_tile_sample_min_size | number | 最小 tile 大小 | number_input |

#### Wan
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| vae_cache_cpu | checkbox | VAE 缓存到 CPU | checkbox |
| clip | path | CLIP 模型路径 | path_selector |

#### FramePack
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| image_encoder | path | Image Encoder 路径 | path_selector |
| f1 | checkbox | F1 模式 | checkbox |
| one_frame | checkbox | 单帧模式 | checkbox |
| one_frame_no_2x | checkbox | 禁用 2x 放大 | checkbox |
| one_frame_no_4x | checkbox | 禁用 4x 放大 | checkbox |

### Text Encoder 缓存参数

#### FLUX.2 / Z-Image / Qwen Image
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| text_encoder | path | Text Encoder 路径 | path_selector |
| fp8_text_encoder | checkbox | 使用 FP8 | checkbox |

#### HunyuanVideo / HV 1.5 / FramePack / Kontext / Long-CAT
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| text_encoder1 | path | Text Encoder 1 (LLM) | path_selector |
| text_encoder2 | path | Text Encoder 2 (CLIP) | path_selector |
| text_encoder_dtype | select | TE 数据类型 | select |
| fp8_llm | checkbox | 使用 FP8 LLM | checkbox |

#### Wan
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| t5 | path | T5 模型路径 | path_selector |
| fp8_t5 | checkbox | 使用 FP8 T5 | checkbox |

#### 通用 TE 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| batch_size | number | TE 批处理大小 | number_input |
| device | select | TE 运行设备 | select |
| num_workers | number | TE 数据加载线程 | number_input |
| skip_existing | checkbox | 跳过已存在的 TE 缓存 | checkbox |

### 调试参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| debug_mode | select | 调试模式 | select [image/console] |
| console_width | number | 控制台宽度 | number_input |
| console_back | select | 背景颜色 | select |
| console_num_images | number | 显示图片数量 | number_input |

## 训练脚本参数 (step3_train)

### 基础参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| dataset_config | path | TOML 配置文件 | path_selector |
| dit | path | DiT 模型路径 | path_selector |
| vae | path | VAE 模型路径 | path_selector |
| seed | number | 随机种子 | number_input |
| output_name | text | 输出模型名称 | text_input |
| output_dir | path | 输出目录 | path_selector |
| logging_dir | path | 日志目录 | path_selector |

### 架构特定参数

#### FLUX.2
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| model_version | select | 模型版本 | select [dev/klein-4b/klein-base-4b/klein-9b/klein-base-9b] |
| text_encoder | path | Text Encoder 路径 | path_selector |
| fp8_text_encoder | checkbox | FP8 Text Encoder | checkbox |
| fp8_scaled | checkbox | FP8 Scaled | checkbox |

#### Wan
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| task | select | 任务类型 | select |
| t5 | path | T5 模型路径 | path_selector |
| clip | path | CLIP 模型路径 | path_selector |
| fp8_t5 | checkbox | FP8 T5 | checkbox |
| vae_cache_cpu | checkbox | VAE 缓存到 CPU | checkbox |
| preserve_distribution_shape | checkbox | 保持分布形状 | checkbox |

#### HunyuanVideo
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| text_encoder1 | path | Text Encoder 1 | path_selector |
| text_encoder2 | path | Text Encoder 2 | path_selector |
| fp8_llm | checkbox | FP8 LLM | checkbox |
| vae_tiling | checkbox | VAE Tiling | checkbox |
| vae_chunk_size | number | VAE Chunk Size | number_input |
| vae_spatial_tile_sample_min_size | number | 最小 Tile 大小 | number_input |

#### FramePack
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| image_encoder | path | Image Encoder 路径 | path_selector |
| latent_window_size | number | Latent 窗口大小 | number_input |
| bulk_decode | checkbox | 批量解码 | checkbox |
| vanilla_sampling | checkbox | Vanilla 采样 | checkbox |
| f1 | checkbox | F1 模式 | checkbox |

### 训练配置参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| max_train_steps | number | 最大训练步数 | number_input |
| max_train_epochs | number | 最大训练轮数 | number_input |
| gradient_accumulation_steps | number | 梯度累加步数 | number_input |
| guidance_scale | number | Guidance Scale | number_input |

### 时间步采样参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| timestep_sampling | select | 时间步采样方法 | select [sigma/uniform/sigmoid/shift/flux2_shift] |
| discrete_flow_shift | number | Flow Shift | number_input |
| sigmoid_scale | number | Sigmoid 缩放 | number_input |
| weighting_scheme | select | 加权方案 | select |
| logit_mean | number | Logit 均值 | number_input |
| logit_std | number | Logit 标准差 | number_input |
| mode_scale | number | Mode 缩放 | number_input |
| min_timestep | number | 最小时序 | number_input |
| max_timestep | number | 最大时间步 | number_input |
| show_timesteps | select | 显示时序 | select [console/image] |

### 学习率参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| learning_rate | text | 学习率 | text_input |
| lr_scheduler | select | 学习率调度器 | select |
| lr_warmup_steps | number | Warmup 步数 | number_input |
| lr_decay_steps | number | 衰减步数 | number_input |
| lr_scheduler_num_cycles | number | 重启次数 | number_input |
| lr_scheduler_power | number | 多项式 Power | number_input |
| lr_scheduler_timescale | number | 时间缩放 | number_input |
| lr_scheduler_min_lr_ratio | number | 最小 LR 比率 | number_input |

### 网络结构参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| network_module | text | 网络模块 | text_input (固定值) |
| network_dim | number | Network Dim | number_input |
| network_alpha | number | Network Alpha | number_input |
| network_dropout | number | Dropout | number_input |
| dim_from_weights | checkbox | 从权重获取 Dim | checkbox |
| scale_weight_norms | number | Scale Weight Norms | number_input |

### 内存优化参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| gradient_checkpointing | checkbox | 梯度检查点 | checkbox |
| gradient_checkpointing_cpu_offload | checkbox | CPU Offload | checkbox |
| fp8_base | checkbox | FP8 基础模式 | checkbox |
| fp8_scaled | checkbox | FP8 Scaled | checkbox |
| fp8_text_encoder | checkbox | FP8 Text Encoder | checkbox |
| blocks_to_swap | number | Block Swap 数量 | number_input |
| use_pinned_memory_for_block_swap | checkbox | Pinned Memory | checkbox |
| img_in_txt_in_offloading | checkbox | Img/Txt Offloading | checkbox |

### 注意力参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| attn_mode | select | 注意力模式 | select [flash/xformers/sdpa/sageattn/torch] |
| split_attn | checkbox | Split Attention | checkbox |

### 精度参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| mixed_precision | select | 混合精度 | select [bf16/fp16/no] |
| full_bf16 | checkbox | Full BF16 | checkbox |
| vae_dtype | select | VAE 数据类型 | select |
| dit_dtype | select | DiT 数据类型 | select |

### Torch Compile 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| compile | checkbox | 启用 Compile | checkbox |
| compile_backend | select | 编译后端 | select [inductor/eager/aot_eager] |
| compile_mode | select | 编译模式 | select |
| compile_fullgraph | checkbox | Full Graph | checkbox |
| compile_dynamic | select | Dynamic | select [auto/true/false] |
| compile_cache_size_limit | number | 缓存大小限制 | number_input |

### TF32 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| cuda_allow_tf32 | checkbox | 允许 TF32 | checkbox |
| cuda_cudnn_benchmark | checkbox | CuDNN Benchmark | checkbox |

### 数据加载参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| max_data_loader_n_workers | number | 数据加载线程数 | number_input |
| persistent_data_loader_workers | checkbox | 持久化 Workers | checkbox |

### 优化器参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| optimizer_type | select | 优化器类型 | select |
| max_grad_norm | number | 最大梯度范数 | number_input |

### 保存参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| save_every_n_epochs | number | 每 N 轮保存 | number_input |
| save_every_n_steps | number | 每 N 步保存 | number_input |
| save_last_n_epochs | number | 保存最后 N 轮 | number_input |
| save_last_n_steps | number | 保存最后 N 步 | number_input |

### 训练状态参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| save_state | checkbox | 保存训练状态 | checkbox |
| save_state_on_train_end | checkbox | 只在训练结束保存 | checkbox |
| save_last_n_epochs_state | number | 保存最后 N 轮状态 | number_input |
| save_last_n_steps_state | number | 保存最后 N 步状态 | number_input |
| resume | path | 恢复训练路径 | path_selector |

### LoRA+ 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| enable_lora_plus | checkbox | 启用 LoRA+ | checkbox |
| loraplus_lr_ratio | number | LoRA+ LR 比率 | number_input |

### 目标 Block 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| enable_blocks | checkbox | 启用 Block 选择 | checkbox |
| exclude_patterns | text | 排除模式 | text_input |
| include_patterns | text | 包含模式 | text_input |

### LyCORIS 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| enable_lycoris | checkbox | 启用 LyCORIS | checkbox |
| conv_dim | number | 卷积 Dim | number_input |
| conv_alpha | number | 卷积 Alpha | number_input |
| algo | select | 算法 | select [lora/loha/ia3/lokr/dylora/diag-oft/full] |
| dropout | number | Dropout | number_input |
| preset | select | 预设 | select |
| factor | number | 因子 | number_input |
| decompose_both | checkbox | Decompose Both | checkbox |
| block_size | number | Block Size | number_input |
| use_tucker | checkbox | Use Tucker | checkbox |
| use_scalar | checkbox | Use Scalar | checkbox |
| train_norm | checkbox | Train Norm | checkbox |
| dora_wd | checkbox | DoRA WD | checkbox |
| full_matrix | checkbox | Full Matrix | checkbox |
| bypass_mode | checkbox | Bypass Mode | checkbox |
| rescaled | number | Rescaled | number_input |
| constrain | number | Constrain | number_input |

### 采样参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| enable_sample | checkbox | 启用采样 | checkbox |
| sample_at_first | checkbox | 训练开始时采样 | checkbox |
| sample_prompts | path | 采样提示词文件 | path_selector |
| sample_every_n_epochs | number | 每 N 轮采样 | number_input |
| sample_every_n_steps | number | 每 N 步采样 | number_input |

### 元数据参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| training_comment | text | 训练注释 | text_input |
| metadata_title | text | 元数据标题 | text_input |
| metadata_author | text | 元数据作者 | text_input |
| metadata_description | text | 元数据描述 | text_input |
| metadata_license | text | 元数据许可证 | text_input |
| metadata_tags | text | 元数据标签 | text_input |

### HuggingFace 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| async_upload | checkbox | 异步上传 | checkbox |
| huggingface_repo_id | text | Repo ID | text_input |
| huggingface_repo_type | text | Repo Type | text_input |
| huggingface_path_in_repo | text | Path in Repo | text_input |
| huggingface_token | text | Token | text_input |
| huggingface_repo_visibility | select | 可见性 | select [public/private] |
| save_state_to_huggingface | checkbox | 保存状态到 HF | checkbox |
| resume_from_huggingface | checkbox | 从 HF 恢复 | checkbox |

### DDP 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| multi_gpu | checkbox | 多 GPU | checkbox |
| ddp_timeout | number | DDP 超时 | number_input |
| ddp_gradient_as_bucket_view | checkbox | Gradient Bucket View | checkbox |
| ddp_static_graph | checkbox | Static Graph | checkbox |

### 权重参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| network_weights | path | LoRA 预训练权重 | path_selector |
| base_weights | text | 基础权重路径 | text_input |
| base_weights_multiplier | text | 基础权重倍率 | text_input |

### WandB 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| wandb_api_key | text | WandB API Key | text_input |

## 推理脚本参数 (step4_generate)

### 基础参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| dit | path | DiT 模型路径 | path_selector |
| vae | path | VAE 模型路径 | path_selector |
| save_path | path | 保存路径 | path_selector |
| seed | number | 随机种子 | number_input |

### 架构特定参数

#### FLUX.2
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| model_version | select | 模型版本 | select |
| text_encoder | path | Text Encoder 路径 | path_selector |
| fp8_text_encoder | checkbox | FP8 Text Encoder | checkbox |
| fp8_scaled | checkbox | FP8 Scaled | checkbox |
| control_image_path | path | 参考图像路径 | path_selector |
| no_resize_control | checkbox | 不调整控制图像大小 | checkbox |

#### Wan
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| task | select | 任务类型 | select |
| t5 | path | T5 模型路径 | path_selector |
| clip | path | CLIP 模型路径 | path_selector |
| fp8_t5 | checkbox | FP8 T5 | checkbox |
| negative_prompt | text | 负向提示词 | text_input |
| guidance_scale | number | Guidance Scale | number_input |
| vae_cache_cpu | checkbox | VAE 缓存到 CPU | checkbox |
| trim_tail_frames | number | 裁剪尾部帧数 | number_input |
| cpu_noise | checkbox | CPU 噪声 | checkbox |
| cfg_skip_mode | select | CFG Skip 模式 | select |
| cfg_apply_ratio | number | CFG 应用比率 | number_input |
| slg_layers | text | SLG 层 | text_input |
| slg_scale | number | SLG Scale | number_input |
| slg_start | number | SLG 开始 | number_input |
| slg_end | number | SLG 结束 | number_input |
| slg_mode | select | SLG 模式 | select |
| offload_inactive_dit | checkbox | 卸载非活跃 DiT | checkbox |
| timestep_boundary | text | 时序边界 | text_input |
| image_path | path | 图像路径 | path_selector |
| end_image_path | path | 结束图像路径 | path_selector |
| control_path | path | 控制视频路径 | path_selector |
| dit_high_noise | path | 高噪声 DiT | path_selector |

#### HunyuanVideo
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| text_encoder1 | path | Text Encoder 1 | path_selector |
| text_encoder2 | path | Text Encoder 2 | path_selector |
| fp8_llm | checkbox | FP8 LLM | checkbox |
| fp8_fast | checkbox | FP8 Fast | checkbox |
| vae_chunk_size | number | VAE Chunk Size | number_input |
| vae_spatial_tile_sample_min_size | number | 最小 Tile 大小 | number_input |
| split_attn | checkbox | Split Attention | checkbox |
| embedded_cfg_scale | number | Embedded CFG Scale | number_input |
| img_in_txt_in_offloading | checkbox | Img/Txt Offloading | checkbox |
| image_path | path | 图像路径 | path_selector |
| end_image_path | path | 结束图像路径 | path_selector |
| control_image_path | path | 控制图像路径 | path_selector |

#### FramePack
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| image_encoder | path | Image Encoder 路径 | path_selector |
| latent_window_size | number | Latent 窗口大小 | number_input |
| bulk_decode | checkbox | 批量解码 | checkbox |
| video_seconds | number | 视频秒数 | number_input |
| video_sections | text | 视频段落 | text_input |
| f1 | checkbox | F1 模式 | checkbox |
| one_frame_inference | text | 单帧推理 | text_input |
| image_mask_path | path | 图像掩码路径 | path_selector |
| end_image_mask_path | path | 结束图像掩码路径 | path_selector |
| magcache_calibration | checkbox | MegaCache 校准 | checkbox |
| enable_megacache | checkbox | 启用 MegaCache | checkbox |
| magcache_mag_ratios | text | MegaCache Mag Ratios | text_input |
| magcache_retention_ratio | number | MegaCache 保留比率 | number_input |
| magcache_threshold | number | MegaCache 阈值 | number_input |
| magcache_k | number | MegaCache K | number_input |

### 生成参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| prompt | text | 提示词 | textarea |
| negative_prompt | text | 负向提示词 | textarea |
| from_file | path | 提示词文件 | path_selector |
| image_size | text | 图像尺寸 | text_input |
| video_size | text | 视频尺寸 | text_input |
| video_length | number | 视频长度 | number_input |
| fps | number | FPS | number_input |
| infer_steps | number | 推理步数 | number_input |
| guidance_scale | number | Guidance Scale | number_input |
| embedded_cfg_scale | number | Embedded CFG Scale | number_input |
| flow_shift | number | Flow Shift | number_input |

### 内存优化参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| fp8 | checkbox | FP8 | checkbox |
| fp8_scaled | checkbox | FP8 Scaled | checkbox |
| fp8_text_encoder | checkbox | FP8 Text Encoder | checkbox |
| device | select | 设备 | select |
| attn_mode | select | 注意力模式 | select |
| blocks_to_swap | number | Block Swap 数量 | number_input |
| use_pinned_memory_for_block_swap | checkbox | Pinned Memory | checkbox |

### 输出参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| output_type | select | 输出类型 | select [images/video/both/latent] |
| no_metadata | checkbox | 不包含元数据 | checkbox |
| latent_path | path | Latent 路径 | path_selector |

### LoRA 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| lora_weight | text | LoRA 权重路径 | textarea |
| lora_multiplier | text | LoRA 倍率 | text_input |
| lora_weight_high_noise | text | 高噪声 LoRA 权重 | textarea |
| lora_multiplier_high_noise | text | 高噪声 LoRA 倍率 | text_input |
| include_patterns | text | 包含模式 | text_input |
| exclude_patterns | text | 排除模式 | text_input |
| save_merged_model | checkbox | 保存合并模型 | checkbox |
| lycoris | checkbox | LyCORIS | checkbox |

### Torch Compile 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| compile | checkbox | 启用 Compile | checkbox |
| compile_backend | select | 编译后端 | select |
| compile_mode | select | 编译模式 | select |
| compile_fullgraph | checkbox | Full Graph | checkbox |
| compile_dynamic | select | Dynamic | select |
| compile_cache_size_limit | number | 缓存大小限制 | number_input |

### 采样器参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| sample_solver | select | 采样求解器 | select [unipc/dpm++/vanilla] |
