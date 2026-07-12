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
