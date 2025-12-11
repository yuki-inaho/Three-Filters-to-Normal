# tftn - Three-Filters-to-Normal CLI Tools

## Tools

| Tool | Description |
|------|-------------|
| `tftn` | Surface normal estimation (CUDA) |
| `bin2png` | Convert float32 .bin depth to uint16 PNG |

## Quick Start

```bash
# Build
just build-apps

# Run with sample data
just tftn-sample

# Display result
just tftn-sample --set tftn_show true
```

## tftn - Surface Normal Estimation

### Basic Usage

```bash
# Output to file
./apps/build/tftn -i depth.png -o normal.png --fx 500 --fy 500

# Display only (no save)
./apps/build/tftn -i depth.png --show --fx 500 --fy 500
```

### Full Example

```bash
./apps/build/tftn \
  -i /path/to/depth.png \
  -o /path/to/output/normal.png \
  --fx 600 --fy 600 \
  --uo 320 --vo 240 \
  --scale 1.5 \
  -k sobel -a median
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input PATH` | Input depth image (uint16 PNG) | **required** |
| `-o, --output PATH` | Output normal map (16-bit PNG) | - |
| `-s, --show` | Display result only (no save) | false |
| `--fx N` | Focal length X | 500 |
| `--fy N` | Focal length Y | 500 |
| `--uo N` | Principal point X | 320 |
| `--vo N` | Principal point Y | 240 |
| `--scale N` | Depth scale factor | 1.0 |
| `-k, --kernel TYPE` | Gradient kernel: `basic` or `sobel` | basic |
| `-a, --aggregation TYPE` | nz aggregation: `mean` or `median` | mean |
| `-h, --help` | Show help message | - |

### Input Format

- **Format**: uint16 PNG (single channel)
- **Values**: Depth values (after applying `--scale`, represents actual depth)
- **Note**: If depth exceeds uint16 range (0-65535), use `bin2png` to normalize

### Output Format

- **Format**: 16-bit RGB PNG
- **Channels**: BGR order (OpenCV convention)
  - B: nx (normalized to 0-65535)
  - G: ny (normalized to 0-65535)
  - R: nz (normalized to 0-65535)

## bin2png - Depth Conversion Tool

Convert float32 binary depth files to normalized uint16 PNG.

### Usage

```bash
./apps/build/bin2png \
  -i depth.bin \
  -o depth.png \
  -W 640 -H 480 \
  --scale 600
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input PATH` | Input .bin file (float32 array) | **required** |
| `-o, --output PATH` | Output .png file (uint16) | **required** |
| `-W, --width N` | Image width | 640 |
| `-H, --height N` | Image height | 480 |
| `--scale N` | Depth multiplier (offset) | 600 |

### Output Files

- `depth.png` - Normalized uint16 depth image
- `depth.meta` - Normalization parameters

### Workflow

```bash
# 1. Convert .bin to .png
./apps/build/bin2png -i depth.bin -o depth.png --scale 600

# 2. Check the meta file for scale value
cat depth.meta
# depth_scale=1.94382

# 3. Use tftn with the scale from meta file
./apps/build/tftn -i depth.png -o normal.png --scale 1.94382 --fx 1400 --fy 1380
```

## justfile Recipes

```bash
# Build tools
just build-apps

# Run with sample data
just tftn-sample

# Display result (no save)
just tftn-sample --set tftn_show true

# Change algorithm
just tftn-sample --set tftn_kernel sobel --set tftn_aggregation median

# Regenerate sample PNG from .bin
just bin2png-sample

# Clean build
just clean-apps
```

## Sample Data

Located in `apps/sample/`:

| File | Description |
|------|-------------|
| `torusknot_depth.png` | Sample depth image (640x480, normalized) |
| `torusknot_depth.meta` | Normalization parameters |
| `torusknot_params.txt` | Camera parameters (fx=1400, fy=1380, uo=350, vo=200) |

## Algorithm Options

### Gradient Kernel (`-k`)

| Value | Description |
|-------|-------------|
| `basic` | 2-point gradient (fast) |
| `sobel` | Sobel 3x3 weighted gradient (smoother) |

### nz Aggregation (`-a`)

| Value | Description |
|-------|-------------|
| `mean` | Average of valid nz candidates (recommended) |
| `median` | Median of valid nz candidates |

## Examples

### Process a depth image with default parameters

```bash
./apps/build/tftn -i depth.png -o normal.png
```

### Use specific camera intrinsics

```bash
./apps/build/tftn -i depth.png -o normal.png \
  --fx 525 --fy 525 --uo 319.5 --vo 239.5
```

### Preview result without saving

```bash
./apps/build/tftn -i depth.png --show --fx 525 --fy 525
```

### Use Sobel gradient with median aggregation

```bash
./apps/build/tftn -i depth.png -o normal.png \
  --fx 525 --fy 525 \
  -k sobel -a median
```

### Process normalized depth with scale factor

```bash
./apps/build/tftn -i normalized_depth.png -o normal.png \
  --scale 2.5 --fx 500 --fy 500
```
