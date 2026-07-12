# PSD Layer Export

PSD Export extracts layers from PSD folders into PNG datasets and can optionally build or export Lance data. It is a script-only workflow and is not currently a GUI Tools page.

```powershell
.\2.4.psdexport.ps1
python -m utils.psd_dataset_pipeline --help
```

The profile is `psdexport`. Options cover hidden layers, line-art merging, fixed seven-layer output, resizing, and Lance export. Complex blend modes, masks, and smart objects may differ from Photoshop rendering, so inspect representative PSD files first.
