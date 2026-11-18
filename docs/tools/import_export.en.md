# Lance Import And Export

Import and Export connect ordinary media directories with Lance datasets. Import first, run split/tag/caption jobs against the dataset, then export the intended version and captions.

```powershell
.\lanceImport.ps1
.\lanceExport.ps1
python -m module.lanceImport --help
python -m module.lanceexport --help
```

Import accepts media, text, sidecar captions, and existing data directories. `import_mode` filters media types, `tag` identifies the import version, and `data_storage_version` controls newly created Lance datasets. The default configuration may omit original binary blobs.

Export uses `lance_file` and `version` to select the dataset state. `caption_suffix`, `caption_extension`, and `allowed_caption_types` control caption filenames and media types. Do not run multiple writers against the same Lance dataset.

Standalone export copies media and captions to `./exports` or the chosen output directory. It reads the selected version's stored media when available and otherwise copies the original. Use `--caption_dir` for a separate caption destination. On a conflict with existing files of unknown ownership, choose an empty destination rather than removing your files. Use `--in_place` only when you intend to publish captions beside the source media.

Paginated Markdown is produced only by a standalone export. In-place publication writes the main document without claiming a neighboring directory; audio/video clipping is still available. Unknown clip directories are not overwritten, and verified unchanged derivatives can be reused on a rerun.
