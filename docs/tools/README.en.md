# Tool Documentation Index

The root README keeps installation, the core workflow, and navigation. Each page below owns the function, inputs, outputs, dependencies, and important options for one workflow.

## Core Workflow

| Page | Guide | Entrypoint |
| --- | --- | --- |
| Import / Export | [Lance import and export](import_export.en.md) | `lanceImport.ps1` / `lanceExport.ps1` |
| Split | [Video splitting](video_split.en.md) | `2.0.video_spliter.ps1` |
| Tagger | [Image tagging](tagging.en.md) | `3.tagger.ps1` |
| Caption | [Captioning and multimodal descriptions](captioning.en.md) | `4.captioner.ps1` |

## Standalone Tools

| Tool | Guide | Entrypoint |
| --- | --- | --- |
| Watermark detection | [WaterDetect](watermark_detection.en.md) | `2.1.image_watermark_detect.ps1` |
| Image preprocessing | [Preprocessing and alignment](image_preprocessing.en.md) | `2.2.preprocess_images.ps1` |
| Image scoring | [Image quality scoring](image_scoring.en.md) | `2.3.image_reward_model.ps1` |
| PSD layer export | [PSD Export](psd_export.en.md) | `2.4.psdexport.ps1` |
| Audio separation | [Audio separation](audio_separation.en.md) | `2.5.audio_separator.ps1` |
| Image2PSD | [Image2PSD / See-through](image2psd.en.md) | `2.6.image2psd.ps1` |
| Music transcription | [MuScriptor audio to MIDI](muscriptor.en.md) | `2.7.music_transcription.ps1` |
| Sheet-music scan | [MuSViT embeddings](sheet_music.en.md) | GUI Tools / `module.sheet_music_musvit` |
| Document translation | [Text and document translation](text_translation.en.md) | `5.translate.ps1` |

See the [configuration guide](../configuration.en.md) and [GUI parameter map](../../gui/PARAMETERS.md) for profile and parameter ownership.
