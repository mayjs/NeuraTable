# NeuraTable

NeuraTable is a framework and tool to augment open source photography workflows with machine learning tools.

## State of the Project

NeuraTable is still in a very early stage of development.
It's more of a proof of concept for now, but I hope to grow this into a useful and easy to use application in the future.

## Technology

NeuraTable can run neural networks in the [ONNX](https://onnx.ai/) format.
It uses [the Wonnx runtime](https://github.com/webonnx/wonnx) to run these models on all platforms where Vulkan is available.

As of right now, NeuraTable does not ship any ONNX models.
The inference logic is mostly inspired from [nind-denoise](https://github.com/trougnouf/nind-denoise), and my plan is to ship
a pretrained nind-denoise model for version 1.

If you want to give the current state of the project a go, you can try to convert one of the nind-denoise UNet snapshots to
ONNX and run that through the `neuratable_run_onnx` binary.

## How to Run

NeuraTable currently provides an executable called `neuratable_run_onnx`.
To build this executable, you'll need a stable rust toolchain and Vulkan available on your system.
If you use NixOS, you can just use `nix develop` to get a shell with the dependencies in place.

To run a network on an example, use `cargo run --release -- <PATH_TO_MODEL.onnx> <PATH_TO_INPUT.jpg> <PATH_TO_OUTPUT.jpg>`
