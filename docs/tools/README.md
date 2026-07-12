# 工具文档索引

主 README 只保留安装、核心工作流和入口导航。每个页面的功能、输入输出、依赖与常用参数在这里分开维护。

## 核心工作流

| 页面 | 文档 | 入口 |
| --- | --- | --- |
| Import / Export | [数据导入与导出](import_export.md) | `lanceImport.ps1` / `lanceExport.ps1` |
| Split | [视频分割](video_split.md) | `2.0.video_spliter.ps1` |
| Tagger | [图像打标](tagging.md) | `3.tagger.ps1` |
| Caption | [字幕与多模态描述](captioning.md) | `4.captioner.ps1` |

## 独立工具

| 工具 | 文档 | 入口 |
| --- | --- | --- |
| 水印检测 | [WaterDetect](watermark_detection.md) | `2.1.image_watermark_detect.ps1` |
| 图片预处理 | [图片预处理与对齐](image_preprocessing.md) | `2.2.preprocess_images.ps1` |
| 图像评分 | [图像质量评分](image_scoring.md) | `2.3.image_reward_model.ps1` |
| PSD 图层导出 | [PSD Export](psd_export.md) | `2.4.psdexport.ps1` |
| 音频分轨 | [音频分轨](audio_separation.md) | `2.5.audio_separator.ps1` |
| Image2PSD | [Image2PSD / See-through](image2psd.md) | `2.6.image2psd.ps1` |
| 音乐转录 | [MuScriptor 音频转 MIDI](muscriptor.md) | `2.7.music_transcription.ps1` |
| 乐谱扫描 | [MuSViT 乐谱 embedding](sheet_music.md) | GUI Tools / `module.sheet_music_musvit` |
| 文档翻译 | [文本与文档翻译](text_translation.md) | `5.translate.ps1` |

参数来源和 profile 映射见 [配置指南](../configuration.md) 与 [GUI 参数映射](../../gui/PARAMETERS.md)。
