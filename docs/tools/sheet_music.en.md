# MuSViT Sheet-Music OMR

The Sheet Music tool transcribes score images and PDF pages with the pinned
MuSViT ONNX model. It writes MusicXML, MIDI, or both, plus Kern and token
diagnostics for every page.

The main entrypoint is the GUI Tools page. The equivalent CLI is:

```powershell
.\.venv\Scripts\python.exe -m module.sheet_music_musvit INPUT_PATH `
  --output_format both
```

`INPUT_PATH` may be an image, PDF, or directory. Directory scans can be
recursive. PDFs are rendered and transcribed one page at a time; rendered page
images are released before the next page is requested.

Without `--output_dir`, file inputs write to `musvit_omr_output` beside the
input file, while directory inputs write to `musvit_omr_output` inside that
directory.

Each image gets its own output directory. Each PDF gets `page_0001`,
`page_0002`, and so on. A successful page contains:

- `tokens.json`
- `score.krn`
- `metadata.json`
- `score.musicxml` and/or `score.mid`

The root `manifest.json` records successes, skips, and failures. Invalid or
truncated notation retains diagnostics but is not exported as MusicXML or
MIDI. After every PDF page succeeds, the PDF output directory also receives
one page-ordered `score.musicxml` and/or `score.mid`. A page or aggregation
failure removes the whole-document exports while retaining page diagnostics.
Whole-document aggregation aligns parts on the shared Kern barline grid.
Kern-declared `.`-only silent measures receive hidden rest padding when
music21 omits them, so later pages cannot shift out of alignment.
Clef, key, meter, and common/cut-time symbols are re-anchored from their Kern
positions. Identical page-header context is omitted when pages merge, while
actual context changes remain untouched.

The dependency profile is `musvit-onnx`. The default model revision is pinned
in `config/model.toml`. PDF DPI controls rasterization before the model's fixed
1024 x 1024 bilinear resize; a higher DPI is not a guaranteed accuracy gain.
