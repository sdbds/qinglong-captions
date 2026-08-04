# Auto-Rig Spine Runtime Probe

This probe links against an externally supplied official Spine C++ Runtime and
does not vendor or redistribute it. Auto-rig's Spine exporter targets 4.2, so
build the release gate with a 4.2 `spine-cpp` checkout. A 4.3 build is useful
only as the version-mismatch negative test and must reject 4.2 skeleton data.
The validated 4.2 source revision is official `spine-runtimes` commit
`b81e5a58ed38704aee4f866f0e0ac672623ce914`.

```powershell
cmake -S tools/auto_rig_spine_runtime `
  -B .cache/auto_rig_spine_runtime_4_2_build `
  -DSPINE_CPP_ROOT=F:/path/to/spine-runtimes/spine-cpp
cmake --build .cache/auto_rig_spine_runtime_4_2_build --config Release
```

Run it directly:

```powershell
.cache/auto_rig_spine_runtime_4_2_build/Release/auto_rig_spine_runtime.exe `
  --skeleton item/rig/spine/skeleton.json `
  --atlas item/rig/spine/skeleton.atlas `
  --report runtime-report.json
```

Or pass the executable to the one-item pipeline as `spine_runtime_path`. The
Python validator requires Runtime 4.2, exercises every exported animation for
its full duration at 60 FPS, and records only content digests and evidence. The
v3 protocol separates maximum world-vertex displacement, displacement
normalized by the changed setup support, and slot-alpha delta from generic
state changes. Breath uses the normalized geometry signal. Talk accepts either
a visible normalized geometry change or a complete opacity/attachment
crossfade, so a valid native alpha-only mouth swap is not rejected. It never
writes the local Runtime path into public artifacts.
