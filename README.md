# Synesthesia Engine

A live, in-browser music visualizer. Play music from your microphone, a file, or a browser tab, and procedural WebGL2 shaders react at 60 fps. Small PyTorch models, exported to ONNX and run in the browser, listen to the audio and choose the shader parameters. Visitors vote between two versions, and those votes train a reward model that improves the mapping over time. No audio ever leaves your browser. Work in progress: see `docs/design.md` for the design and roadmap.
