// SPDX-License-Identifier: Apache-2.0
// The execution providers a model session may run on: WebGPU, or WebAssembly on the CPU.

export type ExecutionProvider = "webgpu" | "wasm";
export const EXECUTION_PROVIDERS: readonly ExecutionProvider[] = ["webgpu", "wasm"];
