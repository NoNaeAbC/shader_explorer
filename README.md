
# Shader Explorer

Compile glsl and slang compute shaders to assembly

## Build And Install

Meson is unable to use `import('cmake')` with Slang, so `./build_slang.sh` exists to build Slang before configuring this project.

```sh
$ ./build_slang.sh
$ meson setup build --prefix "$PWD/install"
$ ninja -C build
$ ninja -C build install
```

The supported runtime entry point is `install/bin/shader_explorer`. It uses the installed Mesa layout under the configured prefix and reads Vulkan ICD JSON files from `install/share/vulkan/icd.d` and shim libraries from `install/lib64`.

## Example Usage

```sh
$ install/bin/shader_explorer --gpu intel-tgl test_shader/complex_compute.slang
```

## CLI

```text
shader_explorer [--version] [--gpu <key>] [--lang <auto|slang|glsl>]
                [--target <info|spirv|final_nir|asm>]
                [--binding-model <classic|push_descriptor|descriptor_buffer>]
                [--spirv-target <max|1.0|1.1|1.2|1.3|1.4|1.5|1.6>]
                [--subgroup-size <n>] [--require-full-subgroups]
                [--no-color] [--output <-|file>] [--list-gpus]
                <shader.slang|shader.glsl>
```

Notable options:

- `--version` prints the program version and exits.
- `--list-gpus` prints the available GPU presets.
- `--no-color` disables ANSI color in text output.
- `--output <path>` writes output to a file; color is disabled automatically for file output.

Available GPUs:
  amd-bonaire - AMD RADV (bonaire)
  amd-gfx1201 - AMD RADV (gfx1201)
  amd-navi10 - AMD RADV (navi10)
  amd-navi21 - AMD RADV (navi21)
  amd-navi31 - AMD RADV (navi31)
  amd-navi33 - AMD RADV (navi33)
  amd-pitcairn - AMD RADV (pitcairn)
  amd-polaris10 - AMD RADV (polaris10)
  amd-polaris12 - AMD RADV (polaris12)
  amd-raphael_mendocino - AMD RADV (raphael_mendocino)
  amd-raven - AMD RADV (raven)
  amd-raven2 - AMD RADV (raven2)
  amd-renoir - AMD RADV (renoir)
  amd-stoney - AMD RADV (stoney)
  amd-strix1 - AMD RADV (strix1)
  amd-vangogh - AMD RADV (vangogh)
  amd-vega10 - AMD RADV (vega10)
  amd-vega20 - AMD RADV (vega20)
  arm-g31 - Panfrost PanVK (G31)
  arm-g310 - Panfrost PanVK (G310)
  arm-g51 - Panfrost PanVK (G51)
  arm-g52 - Panfrost PanVK (G52)
  arm-g610 - Panfrost PanVK (G610)
  arm-g72 - Panfrost PanVK (G72)
  arm-g720 - Panfrost PanVK (G720) variant 4
  arm-g725 - Panfrost PanVK (G725) variant 4
  arm-g76 - Panfrost PanVK (G76)
  intel-adl - Intel ANV (adl)
  intel-aml - Intel ANV (aml)
  intel-arl - Intel ANV (arl)
  intel-bdw - Intel ANV (bdw)
  intel-bmg - Intel ANV (bmg)
  intel-bxt - Intel ANV (bxt)
  intel-byt - Intel ANV (byt)
  intel-cfl - Intel ANV (cfl)
  intel-chv - Intel ANV (chv)
  intel-cml - Intel ANV (cml)
  intel-dg1 - Intel ANV (dg1)
  intel-dg2 - Intel ANV (dg2)
  intel-ehl - Intel ANV (ehl)
  intel-glk - Intel ANV (glk)
  intel-hsw - Intel ANV (hsw)
  intel-icl - Intel ANV (icl)
  intel-ivb - Intel ANV (ivb)
  intel-jsl - Intel ANV (jsl)
  intel-kbl - Intel ANV (kbl)
  intel-lnl - Intel ANV (lnl)
  intel-mtl - Intel ANV (mtl)
  intel-ptl - Intel ANV (ptl)
  intel-rkl - Intel ANV (rkl)
  intel-rpl - Intel ANV (rpl)
  intel-sg1 - Intel ANV (sg1)
  intel-skl - Intel ANV (skl)
  intel-tgl - Intel ANV (tgl)
  intel-whl - Intel ANV (whl)
  nvidia-gk104 - Nouveau NVK (GK104, GeForce GTX 680, chipset e4)
  nvidia-gk110 - Nouveau NVK (GK110, GeForce GTX 780, chipset f0)
  nvidia-gm107 - Nouveau NVK (GM107, GeForce GTX 750, chipset 117)
  nvidia-gm204 - Nouveau NVK (GM204, GeForce GTX 980, chipset 124)
  nvidia-gp104 - Nouveau NVK (GP104, GeForce GTX 1080, chipset 134)
  nvidia-gv100 - Nouveau NVK (GV100, TITAN V, chipset 140)
  nvidia-tu102 - Nouveau NVK (TU102, GeForce RTX 2080, chipset 162)
  qualcomm-618 - Freedreno Turnip (A618)
  qualcomm-630 - Freedreno Turnip (A630)
  qualcomm-660 - Freedreno Turnip (A660)
  qualcomm-702 - Freedreno Turnip (A702)
  qualcomm-730 - Freedreno Turnip (A730)
  qualcomm-740 - Freedreno Turnip (A740)
  qualcomm-750 - Freedreno Turnip (A750)
  qualcomm-830 - Freedreno Turnip (A830)
```
