import os
import struct
import hashlib
import uuid
import numpy as np
from plyfile import PlyData
from dataclasses import dataclass
from enum import Enum
from sklearn.cluster import KMeans

CHUNK_SIZE = 256
TEXTURE_WIDTH = 2048
SH_DIMENSIONS = 15 * 3


def decode_morton_2d_16x16(t):
    t = (t & 0xFF) | ((t & 0xFE) << 7)
    t &= 0x5555
    t = (t ^ (t >> 1)) & 0x3333
    t = (t ^ (t >> 2)) & 0x0f0f
    return (t & 0xF, t >> 8)


def splat_index_to_texture_index(idx):
    xy = decode_morton_2d_16x16(idx)
    width = TEXTURE_WIDTH // 16
    idx >>= 8
    x = (idx % width) * 16 + xy[0]
    y = (idx // width) * 16 + xy[1]
    return int(y * TEXTURE_WIDTH + x)


class VectorFormat(Enum):
    Float32 = 0
    Norm16 = 1
    Norm11 = 2
    Norm6 = 3

class ColorFormat(Enum):
    Float32x4 = 0
    Float16x4 = 1
    Norm8x4 = 2
    BC7 = 3

class SHFormat(Enum):
    Float32 = 0
    Float16 = 1
    Norm11 = 2
    Norm6 = 3
    Cluster64k = 4
    Cluster32k = 5
    Cluster16k = 6
    Cluster8k = 7
    Cluster4k = 8


QUALITY_PRESETS = {
    "VeryLow": {
        "pos": VectorFormat.Norm11,
        "scale": VectorFormat.Norm6,
        "color": ColorFormat.BC7,
        "sh": SHFormat.Cluster4k
    },
    "Low": {
        "pos": VectorFormat.Norm11,
        "scale": VectorFormat.Norm6,
        "color": ColorFormat.Norm8x4,
        "sh": SHFormat.Cluster16k
    },
    "Medium": {
        "pos": VectorFormat.Norm11,
        "scale": VectorFormat.Norm11,
        "color": ColorFormat.Norm8x4,
        "sh": SHFormat.Norm6
    },
    "High": {
        "pos": VectorFormat.Norm16,
        "scale": VectorFormat.Norm16,
        "color": ColorFormat.Float16x4,
        "sh": SHFormat.Norm11
    },
    "VeryHigh": {
        "pos": VectorFormat.Float32,
        "scale": VectorFormat.Float32,
        "color": ColorFormat.Float32x4,
        "sh": SHFormat.Float32
    }
}


@dataclass
class GaussianSplatData:
    positions: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    colors: np.ndarray
    sh_coeffs: np.ndarray


class GaussianAssetCreator:
    def __init__(self, quality="VeryHigh"):
        self.quality = QUALITY_PRESETS[quality]
        self.chunk_size = CHUNK_SIZE
        self.texture_width = TEXTURE_WIDTH
        self.bounds_min = None
        self.bounds_max = None

    def read_ply(self, ply_path):
        ply_data = PlyData.read(ply_path)
        vertices = ply_data['vertex']
        
        positions = np.column_stack([
            vertices['x'], vertices['y'], vertices['z']
        ]).astype(np.float32)
        
        log_scales = np.column_stack([
            vertices['scale_0'], vertices['scale_1'], vertices['scale_2']
        ]).astype(np.float32)
        scales = np.abs(np.exp(log_scales))
        
        rotations = np.column_stack([
            vertices['rot_0'], vertices['rot_1'], vertices['rot_2'], vertices['rot_3']
        ]).astype(np.float32)
        
        sh_dc = np.column_stack([
            vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']
        ]).astype(np.float32)
        kSH_C0 = 0.2820948
        rgb = sh_dc * kSH_C0 + 0.5
        
        opacity_raw = vertices['opacity'].astype(np.float32).reshape(-1, 1)
        opacity = 1.0 / (1.0 + np.exp(-opacity_raw))
        
        colors = np.hstack([rgb, opacity])
        
        sh_rest = []
        for i in range(15):
            sh_rest.append(vertices[f'f_rest_{i}'])
            sh_rest.append(vertices[f'f_rest_{i+15}'])
            sh_rest.append(vertices[f'f_rest_{i+30}'])
        sh_rest = np.column_stack(sh_rest).astype(np.float32)
        sh_coeffs = sh_rest
        
        self.bounds_min = positions.min(axis=0)
        self.bounds_max = positions.max(axis=0)
        
        return GaussianSplatData(
            positions=positions,
            scales=scales,
            rotations=rotations,
            colors=colors,
            sh_coeffs=sh_coeffs
        )

    def morton_part1by2(self, x):
        x &= 0x1fffff
        x = (x ^ (x << 32)) & 0x1f00000000ffff
        x = (x ^ (x << 16)) & 0x1f0000ff0000ff
        x = (x ^ (x << 8)) & 0x100f00f00f00f00f
        x = (x ^ (x << 4)) & 0x10c30c30c30c30c3
        x = (x ^ (x << 2)) & 0x1249249249249249
        return int(x)

    def morton_encode_3d(self, v):
        return (self.morton_part1by2(v[2]) << 2) | (self.morton_part1by2(v[1]) << 1) | self.morton_part1by2(v[0])

    def reorder_morton(self, positions):
        k_scaler = (1 << 21) - 1
        inv_bounds_size = 1.0 / (self.bounds_max - self.bounds_min)
        
        morton_codes = []
        for i in range(len(positions)):
            pos = ((positions[i] - self.bounds_min) * inv_bounds_size * k_scaler).astype(np.uint32)
            pos_int = [int(pos[0]), int(pos[1]), int(pos[2])]
            code = self.morton_encode_3d(pos_int)
            morton_codes.append((code, i))
        
        morton_codes.sort(key=lambda x: (x[0], x[1]))
        return [idx for _, idx in morton_codes]

    def encode_positions(self, positions):
        fmt = self.quality["pos"]
        num_splats = len(positions)
        
        if fmt == VectorFormat.Float32:
            return positions.astype(np.float32).tobytes()
        
        chunk_count = (num_splats + self.chunk_size - 1) // self.chunk_size
        normalized = positions.copy()
        for i in range(chunk_count):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, num_splats)
            chunk_pos = positions[start:end]
            
            chunk_min = chunk_pos.min(axis=0)
            chunk_max = chunk_pos.max(axis=0)
            chunk_max = np.maximum(chunk_max, chunk_min + 1.0e-5)
            
            normalized[start:end] = (chunk_pos - chunk_min) / (chunk_max - chunk_min)
        
        normalized = np.clip(normalized, 0.0, 1.0)
        
        if fmt == VectorFormat.Norm16:
            x = np.clip(normalized[:, 0] * 65535.5, 0, 65535).astype(np.uint32)
            y = np.clip(normalized[:, 1] * 65535.5, 0, 65535).astype(np.uint32)
            z = np.clip(normalized[:, 2] * 65535.5, 0, 65535).astype(np.uint32)
            packed = (x) | (y << 16) | (z << 32)
            return np.array(packed, dtype=np.uint64).tobytes()
        
        if fmt == VectorFormat.Norm11:
            x = np.clip(normalized[:, 0] * 2047.5, 0, 2047).astype(np.uint32)
            y = np.clip(normalized[:, 1] * 1023.5, 0, 1023).astype(np.uint32)
            z = np.clip(normalized[:, 2] * 2047.5, 0, 2047).astype(np.uint32)
            packed = (x << 21) | (y << 11) | z
            return np.array(packed, dtype=np.uint32).tobytes()
        
        else:
            raise NotImplementedError(f"Position format {fmt} not implemented")

    def encode_scales(self, scales, rotations, sh_indices=None):
        scale_fmt = self.quality["scale"]
        
        rot_size = 4
        scale_size = 12 if scale_fmt == VectorFormat.Float32 else 4
        total_size = rot_size + scale_size
        
        num_splats = len(scales)
        if scale_fmt != VectorFormat.Float32:
            chunk_count = (num_splats + self.chunk_size - 1) // self.chunk_size
            normalized_scales = scales.copy()
            
            for i in range(chunk_count):
                start = i * self.chunk_size
                end = min(start + self.chunk_size, num_splats)
                chunk_scales = scales[start:end]
                
                chunk_min = chunk_scales.min(axis=0)
                chunk_max = chunk_scales.max(axis=0)
                chunk_max = np.maximum(chunk_max, chunk_min + 1.0e-5)
                
                normalized_scales[start:end] = (chunk_scales - chunk_min) / (chunk_max - chunk_min)
            
            normalized_scales = np.clip(normalized_scales, 0.0, 1.0)
        else:
            normalized_scales = scales
        
        output = np.zeros(len(scales) * total_size, dtype=np.uint8)
        ptr = 0
        
        for i in range(len(scales)):
            rot = rotations[i]
            rot = self._normalize_swizzle_rotation(rot)
            packed_rot = self._pack_smallest3_rotation(rot)
            rot_encoded = self._encode_quat_to_norm10(packed_rot)
            output[ptr:ptr+4] = np.frombuffer(rot_encoded, dtype=np.uint8)
            ptr += rot_size
            
            scale = normalized_scales[i]
            if scale_fmt == VectorFormat.Float32:
                scale_bytes = struct.pack('<fff', scale[0], scale[1], scale[2])
                output[ptr:ptr+12] = np.frombuffer(scale_bytes, dtype=np.uint8)
            else:
                scale_encoded = self._encode_vector(scale, VectorFormat.Norm11)
                output[ptr:ptr+4] = np.frombuffer(scale_encoded, dtype=np.uint8)
            ptr += scale_size
        
        return output.tobytes()

    def encode_colors(self, colors):
        fmt = self.quality["color"]
        num_colors = len(colors)
        
        if fmt == ColorFormat.Float32x4:
            if colors.shape[1] != 4:
                alpha = np.ones((num_colors, 1), dtype=np.float32)
                colors = np.hstack([colors, alpha])
            
            width = self.texture_width
            height = (num_colors + width - 1) // width
            height = ((height + 15) // 16) * 16
            
            img_data = np.zeros((height, width, 4), dtype=np.float32)
            for i in range(num_colors):
                tex_idx = splat_index_to_texture_index(i)
                row = tex_idx // width
                col = tex_idx % width
                img_data[row, col] = colors[i]
            
            if width == 2048 and height == 16:
                img_data = img_data.reshape(16, 2048, 4)
                img_data_128x256 = np.zeros((256, 128, 4), dtype=np.float32)
                for row in range(16):
                    for col in range(2048):
                        new_col = col % 128
                        new_row = row * 16 + (col // 128)
                        img_data_128x256[new_row, new_col] = img_data[row, col]
                img_data = img_data_128x256
            
            return img_data.tobytes()
        
        elif fmt == ColorFormat.Float16x4:
            if colors.shape[1] != 4:
                alpha = np.ones((num_colors, 1), dtype=np.float32)
                colors = np.hstack([colors, alpha])
            
            width = self.texture_width
            height = (num_colors + width - 1) // width
            height = ((height + 15) // 16) * 16
            
            img_data = np.zeros((height, width, 4), dtype=np.float16)
            for i in range(num_colors):
                tex_idx = splat_index_to_texture_index(i)
                row = tex_idx // width
                col = tex_idx % width
                img_data[row, col] = colors[i].astype(np.float16)
            
            return img_data.tobytes()
        
        elif fmt == ColorFormat.Norm8x4:
            if colors.shape[1] != 4:
                alpha = np.ones((num_colors, 1), dtype=np.float32)
                colors = np.hstack([colors, alpha])
            
            width = self.texture_width
            height = (num_colors + width - 1) // width
            height = ((height + 15) // 16) * 16
            
            img_data = np.zeros((height, width, 4), dtype=np.uint8)
            for i in range(num_colors):
                tex_idx = splat_index_to_texture_index(i)
                row = tex_idx // width
                col = tex_idx % width
                img_data[row, col] = (colors[i] * 255.5).clip(0, 255).astype(np.uint8)
            
            return img_data.tobytes()
        
        elif fmt == ColorFormat.BC7:
            try:
                import bc7enc
            except ImportError:
                raise ImportError("bc7enc library is required for BC7 color format. Please install it with 'pip install bc7enc'")
            
            height = (num_colors + self.texture_width - 1) // self.texture_width
            height = ((height + 15) // 16) * 16
            
            img_data = np.zeros((height, self.texture_width, 4), dtype=np.uint8)
            for i in range(num_colors):
                tex_idx = splat_index_to_texture_index(i)
                row = tex_idx // self.texture_width
                col = tex_idx % self.texture_width
                img_data[row, col] = (colors[i] * 255.5).clip(0, 255).astype(np.uint8)
            
            bc7_data = bc7enc.compress(img_data, format='bc7')
            return bc7_data
        
        else:
            raise NotImplementedError(f"Color format {fmt} not implemented")

    def encode_sh(self, sh_coeffs):
        fmt = self.quality["sh"]
        num_splats = len(sh_coeffs)
        clustered = fmt.value >= SHFormat.Cluster64k.value
        
        if clustered:
            cluster_count = self._get_cluster_count(fmt)
            sh_indices, clustered_sh = self._cluster_sh(sh_coeffs, cluster_count)
            sh_table_data = self._encode_sh_table(clustered_sh, SHFormat.Float16)
            indices_data = np.array(sh_indices, dtype=np.uint16).tobytes()
            return sh_table_data + indices_data, sh_indices
        else:
            return self._encode_sh_direct(sh_coeffs, fmt), None

    def encode_chunks(self, positions, scales, colors, sh_coeffs):
        num_splats = len(positions)
        chunk_count = (num_splats + self.chunk_size - 1) // self.chunk_size
        chunks = []
        
        for i in range(chunk_count):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, num_splats)
            
            chunk_pos = positions[start:end]
            chunk_scl = scales[start:end]
            chunk_col = colors[start:end]
            chunk_shs = sh_coeffs[start:end]
            
            chunk_minpos = chunk_pos.min(axis=0)
            chunk_maxpos = chunk_pos.max(axis=0)
            chunk_maxpos = np.maximum(chunk_maxpos, chunk_minpos + 1.0e-5)
            
            chunk_minscl = chunk_scl.min(axis=0)
            chunk_maxscl = chunk_scl.max(axis=0)
            chunk_maxscl = np.maximum(chunk_maxscl, chunk_minscl + 1.0e-5)
            
            chunk_mincol = chunk_col.min(axis=0)
            chunk_maxcol = chunk_col.max(axis=0)
            chunk_maxcol = np.maximum(chunk_maxcol, chunk_mincol + 1.0e-5)
            
            chunk_minshs = chunk_shs.min(axis=0)
            chunk_maxshs = chunk_shs.max(axis=0)
            chunk_maxshs = np.maximum(chunk_maxshs, chunk_minshs + 1.0e-5)
            
            chunk_data = struct.pack('<ff', chunk_minpos[0], chunk_maxpos[0])
            chunk_data += struct.pack('<ff', chunk_minpos[1], chunk_maxpos[1])
            chunk_data += struct.pack('<ff', chunk_minpos[2], chunk_maxpos[2])
            
            sclX = self._float32_to_f16(chunk_minscl[0]) | (self._float32_to_f16(chunk_maxscl[0]) << 16)
            sclY = self._float32_to_f16(chunk_minscl[1]) | (self._float32_to_f16(chunk_maxscl[1]) << 16)
            sclZ = self._float32_to_f16(chunk_minscl[2]) | (self._float32_to_f16(chunk_maxscl[2]) << 16)
            chunk_data += struct.pack('<III', sclX, sclY, sclZ)
            
            colR = self._float32_to_f16(chunk_mincol[0]) | (self._float32_to_f16(chunk_maxcol[0]) << 16)
            colG = self._float32_to_f16(chunk_mincol[1]) | (self._float32_to_f16(chunk_maxcol[1]) << 16)
            colB = self._float32_to_f16(chunk_mincol[2]) | (self._float32_to_f16(chunk_maxcol[2]) << 16)
            colA = self._float32_to_f16(chunk_mincol[3]) | (self._float32_to_f16(chunk_maxcol[3]) << 16)
            chunk_data += struct.pack('<IIII', colR, colG, colB, colA)
            
            shR = self._float32_to_f16(chunk_minshs[0]) | (self._float32_to_f16(chunk_maxshs[0]) << 16)
            shG = self._float32_to_f16(chunk_minshs[1]) | (self._float32_to_f16(chunk_maxshs[1]) << 16)
            shB = self._float32_to_f16(chunk_minshs[2]) | (self._float32_to_f16(chunk_maxshs[2]) << 16)
            chunk_data += struct.pack('<III', shR, shG, shB)
            
            chunks.append(chunk_data)
        
        return b''.join(chunks)

    def _get_cluster_count(self, sh_format):
        return {
            SHFormat.Cluster64k: 65536,
            SHFormat.Cluster32k: 32768,
            SHFormat.Cluster16k: 16384,
            SHFormat.Cluster8k: 8192,
            SHFormat.Cluster4k: 4096
        }[sh_format]

    def _cluster_sh(self, sh_coeffs, cluster_count):
        sh_flat = sh_coeffs.reshape(-1, SH_DIMENSIONS)
        kmeans = KMeans(
            n_clusters=cluster_count,
            n_init=1,
            max_iter=50,
            random_state=42
        )
        sh_indices = kmeans.fit_predict(sh_flat)
        clustered_sh = kmeans.cluster_centers_
        return sh_indices, clustered_sh

    def _encode_sh_table(self, sh_table, fmt):
        if fmt == SHFormat.Float16:
            return sh_table.astype(np.float16).tobytes()
        elif fmt == SHFormat.Float32:
            return sh_table.astype(np.float32).tobytes()
        else:
            raise NotImplementedError(f"SH table format {fmt} not implemented")

    def _encode_sh_direct(self, sh_coeffs, fmt):
        output = []
        
        for sh in sh_coeffs:
            sh_groups = sh.reshape(15, 3)
            
            if fmt == SHFormat.Float32:
                sh_with_padding = np.zeros((16, 3), dtype=np.float32)
                sh_with_padding[:15] = sh_groups
                output.append(sh_with_padding.astype(np.float32).tobytes())
            
            elif fmt == SHFormat.Float16:
                output.append(sh_groups.astype(np.float16).tobytes())
            
            elif fmt == SHFormat.Norm11:
                for group in sh_groups:
                    encoded = self._encode_float3_to_norm11(group)
                    output.append(struct.pack('<I', encoded))
            
            elif fmt == SHFormat.Norm6:
                for group in sh_groups:
                    encoded = self._encode_float3_to_norm565(group)
                    output.append(struct.pack('<H', encoded))
            
            else:
                raise NotImplementedError(f"SH format {fmt} not implemented")
        
        return b''.join(output)

    def _normalize_swizzle_rotation(self, wxyz):
        q = wxyz / np.linalg.norm(wxyz)
        return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)

    def _pack_smallest3_rotation(self, q):
        abs_q = np.abs(q)
        index = 0
        max_v = abs_q[0]
        if abs_q[1] > max_v:
            index = 1
            max_v = abs_q[1]
        if abs_q[2] > max_v:
            index = 2
            max_v = abs_q[2]
        if abs_q[3] > max_v:
            index = 3
            max_v = abs_q[3]
        
        if index == 0:
            q = np.array([q[1], q[2], q[3], q[0]])
        elif index == 1:
            q = np.array([q[0], q[2], q[3], q[1]])
        elif index == 2:
            q = np.array([q[0], q[1], q[3], q[2]])
        
        three = q[:3] * (1 if q[3] >= 0 else -1)
        three = (three * np.sqrt(2)) * 0.5 + 0.5
        
        return np.array([three[0], three[1], three[2], index / 3.0], dtype=np.float32)

    def _encode_quat_to_norm10(self, quat):
        x = np.clip(quat[0] * 1023.5, 0, 1023).astype(np.uint32)
        y = np.clip(quat[1] * 1023.5, 0, 1023).astype(np.uint32)
        z = np.clip(quat[2] * 1023.5, 0, 1023).astype(np.uint32)
        w = np.clip(quat[3] * 3.5, 0, 3).astype(np.uint32)
        
        packed = (x) | (y << 10) | (z << 20) | (w << 30)
        return struct.pack('<I', packed)

    def _encode_vector(self, vec, fmt):
        if fmt == VectorFormat.Float32:
            return struct.pack('<fff', vec[0], vec[1], vec[2])
        
        elif fmt == VectorFormat.Norm16:
            x = np.clip(vec[0] * 65535.5, 0, 65535).astype(np.uint32)
            y = np.clip(vec[1] * 65535.5, 0, 65535).astype(np.uint32)
            z = np.clip(vec[2] * 65535.5, 0, 65535).astype(np.uint32)
            packed = (x) | (y << 16) | (z << 32)
            return struct.pack('<Q', packed)
        
        elif fmt == VectorFormat.Norm11:
            encoded = self._encode_float3_to_norm11(vec)
            return struct.pack('<I', encoded)
        
        elif fmt == VectorFormat.Norm6:
            encoded = self._encode_float3_to_norm655(vec)
            return struct.pack('<H', encoded)
        
        else:
            raise NotImplementedError(f"Vector format {fmt} not implemented")

    def _encode_float3_to_norm11(self, v):
        x = np.clip(v[0] * 2047.5, 0, 2047).astype(np.uint32)
        y = np.clip(v[1] * 1023.5, 0, 1023).astype(np.uint32)
        z = np.clip(v[2] * 2047.5, 0, 2047).astype(np.uint32)
        return (x) | (y << 11) | (z << 21)

    def _encode_float3_to_norm655(self, v):
        x = np.clip(v[0] * 63.5, 0, 63).astype(np.uint32)
        y = np.clip(v[1] * 31.5, 0, 31).astype(np.uint32)
        z = np.clip(v[2] * 31.5, 0, 31).astype(np.uint32)
        return (x) | (y << 6) | (z << 11)

    def _encode_float3_to_norm565(self, v):
        x = np.clip(v[0] * 31.5, 0, 31).astype(np.uint32)
        y = np.clip(v[1] * 63.5, 0, 63).astype(np.uint32)
        z = np.clip(v[2] * 31.5, 0, 31).astype(np.uint32)
        return (x) | (y << 5) | (z << 11)
    
    def _float32_to_f16(self, f):
        f16 = struct.pack('<e', f)
        return struct.unpack('<H', f16)[0]

    def _generate_guid(self, data_bytes):
        return hashlib.md5(data_bytes).hexdigest().lower()

    def _calculate_data_hash(self, *data_bytes_list):
        hash128 = Hash128()
        for data_bytes in data_bytes_list:
            chunk_size = 4096
            for i in range(0, len(data_bytes), chunk_size):
                chunk = data_bytes[i:i+chunk_size]
                hash128.update(chunk)
        return hash128
    
    def _generate_meta_files(self, output_dir, file_names, guids):
        meta_template = '''fileFormatVersion: 2
guid: {guid}
NativeFormatImporter:
  externalObjects: {{}}
  mainObjectFileID: 0
  userData: 
  assetBundleName: 
  assetBundleVariant: 
'''
        
        for data_type, file_name in file_names.items():
            file_path = os.path.join(output_dir, file_name)
            meta_path = f"{file_path}.meta"
            
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(meta_template.format(guid=guids[data_type]))

    def create_asset(self, ply_path, output_dir, truth_asset_path=None):
        print(f"Reading PLY file: {ply_path}")
        data = self.read_ply(ply_path)
        
        print(f"numbers of points: {len(data.positions)}")
        print(f"Bounds: min={self.bounds_min}, max={self.bounds_max}")
        
        print("Morton encoding and sorting...")
        sorted_indices = self.reorder_morton(data.positions)
        
        data.positions = data.positions[sorted_indices]
        data.scales = data.scales[sorted_indices]
        data.rotations = data.rotations[sorted_indices]
        data.colors = data.colors[sorted_indices]
        data.sh_coeffs = data.sh_coeffs[sorted_indices]
        
        print("Encoding positions...")
        pos_data = self.encode_positions(data.positions)
        
        print("Encoding colors...")
        col_data = self.encode_colors(data.colors)
        
        print("Encoding scales and rotations...")
        oth_data = self.encode_scales(data.scales, data.rotations)
        
        print("Encoding SH coefficients...")
        sh_data, sh_indices = self.encode_sh(data.sh_coeffs)
        
        print("Generating GUIDs...")
        pos_guid = self._generate_guid(pos_data)
        col_guid = self._generate_guid(col_data)
        oth_guid = self._generate_guid(oth_data)
        sh_guid = self._generate_guid(sh_data)
        
        print("Calculating data hash...")
        data_hash = self._calculate_data_hash(pos_data, col_data, oth_data, sh_data)
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("Writing binary data files...")
        asset_name = os.path.splitext(os.path.basename(ply_path))[0]
        file_names = {
            'pos': f'{asset_name}_pos.bytes',
            'col': f'{asset_name}_col.bytes',
            'oth': f'{asset_name}_oth.bytes',
            'shs': f'{asset_name}_shs.bytes'
        }
        
        with open(os.path.join(output_dir, file_names['pos']), 'wb') as f:
            f.write(pos_data)
        with open(os.path.join(output_dir, file_names['col']), 'wb') as f:
            f.write(col_data)
        with open(os.path.join(output_dir, file_names['oth']), 'wb') as f:
            f.write(oth_data)
        with open(os.path.join(output_dir, file_names['shs']), 'wb') as f:
            f.write(sh_data)
        
        print("Creating Unity .asset file...")
        self.create_asset_file(
            output_dir=output_dir,
            asset_name=asset_name,
            pos_guid=pos_guid,
            col_guid=col_guid,
            oth_guid=oth_guid,
            shs_guid=sh_guid,
            splat_count=len(data.positions),
            bounds_min=self.bounds_min,
            bounds_max=self.bounds_max,
            data_hash=data_hash
        )
        
        print("Generating meta files...")
        guids_dict = {
            'pos': pos_guid,
            'col': col_guid,
            'oth': oth_guid,
            'shs': sh_guid
        }
        self._generate_meta_files(output_dir, file_names, guids_dict)
        
        print(f"Done! Asset saved to: {output_dir}")
        print(f"  - {asset_name}.asset")
        print(f"  - pos.bytes ({len(pos_data)} bytes)")
        print(f"  - col.bytes ({len(col_data)} bytes)")
        print(f"  - oth.bytes ({len(oth_data)} bytes)")
        print(f"  - shs.bytes ({len(sh_data)} bytes)")

    def create_asset_file(self, output_dir, asset_name, pos_guid, col_guid, oth_guid, shs_guid, 
                         splat_count, bounds_min, bounds_max, data_hash):
        """Create Unity .asset file"""
        asset_path = os.path.join(output_dir, f'{asset_name}.asset')
        
        content = f"""%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!114 &11400000
MonoBehaviour:
  m_ObjectHideFlags: 0
  m_CorrespondingSourceObject: {{fileID: 0}}
  m_PrefabInstance: {{fileID: 0}}
  m_PrefabAsset: {{fileID: 0}}
  m_GameObject: {{fileID: 0}}
  m_Enabled: 1
  m_EditorHideFlags: 0
  m_Script: {{fileID: 11500000, guid: 33b71fae31e6c7d438e8566dc713e666, type: 3}}
  m_Name: {asset_name}
  m_EditorClassIdentifier: 
  m_FormatVersion: 20231020
  m_SplatCount: {splat_count}
  m_BoundsMin: {{x: {bounds_min[0]:.7f}, y: {bounds_min[1]:.7f}, z: {bounds_min[2]:.7f}}}
  m_BoundsMax: {{x: {bounds_max[0]:.7f}, y: {bounds_max[1]:.7f}, z: {bounds_max[2]:.7f}}}
  m_DataHash:
    serializedVersion: 2
    Hash: {data_hash.to_hex()}
  m_PosFormat: {self.quality['pos'].value}
  m_ScaleFormat: {self.quality['scale'].value}
  m_SHFormat: {self.quality['sh'].value}
  m_ColorFormat: {self.quality['color'].value}
  m_PosData: {{fileID: 4900000, guid: {pos_guid}, type: 3}}
  m_ColorData: {{fileID: 4900000, guid: {col_guid}, type: 3}}
  m_OtherData: {{fileID: 4900000, guid: {oth_guid}, type: 3}}
  m_SHData: {{fileID: 4900000, guid: {shs_guid}, type: 3}}
  m_ChunkData: {{fileID: 0}}
  m_Cameras: []
"""
        
        with open(asset_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        
        return asset_path


class Hash128:
    def __init__(self):
        self.hash = hashlib.md5()
        
    def update(self, data):
        self.hash.update(data)
        
    def to_hex(self):
        return self.hash.hexdigest().lower()
        
    def to_bytes(self):
        return self.hash.digest()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PLY file to Unity Gaussian Splat Asset')
    parser.add_argument('ply_path', nargs='?', default='point_cloud_truck.ply', help='Path to input PLY file (default: point_cloud.ply)')
    parser.add_argument('output_dir', nargs='?', default='output', help='Output directory for assets (default: output)')
    parser.add_argument('--quality', choices=QUALITY_PRESETS.keys(), default='VeryHigh',
                      help='Quality preset (default: VeryHigh)')
    parser.add_argument('--truth', help='Path to truth asset file for camera data')
    
    args = parser.parse_args()
    
    creator = GaussianAssetCreator(quality=args.quality)
    creator.create_asset(
        ply_path=args.ply_path,
        output_dir=args.output_dir,
        truth_asset_path=args.truth
    )
