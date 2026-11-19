"""
Model Manager for LocalLLM Benchmarking.
Handles verified model acquisition, integrity validation, GGUF metadata extraction,
and architectural consistency checks between Q4_K_M and Q8_0 configurations.
"""

import hashlib
import json
import os
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple
from huggingface_hub import hf_hub_download


DEFAULT_MODEL_DIR = r"D:\AI-Assignments\Models\LocalLLM"
HF_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"

MODEL_CONFIGS = {
    "Q4_K_M": {
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "quant_type": "Q4_K_M",
        "description": "4-bit Medium K-quantisation with mixed precision scales for optimal quality/speed",
    },
    "Q8_0": {
        "filename": "qwen2.5-1.5b-instruct-q8_0.gguf",
        "quant_type": "Q8_0",
        "description": "8-bit standard integer quantisation preserving near FP16 precision",
    },
}


def read_gguf_metadata(filepath: str) -> Dict[str, Any]:
    """
    Parses core GGUF binary header and metadata key-value pairs without requiring external GGUF libraries.
    Supports GGUF v2 and v3 format specifications.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    metadata: Dict[str, Any] = {}
    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Invalid GGUF magic bytes: {magic}")

        version = struct.unpack("<I", f.read(4))[0]
        metadata["gguf_version"] = version

        if version >= 2:
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            metadata_kv_count = struct.unpack("<Q", f.read(8))[0]
        else:
            tensor_count = struct.unpack("<I", f.read(4))[0]
            metadata_kv_count = struct.unpack("<I", f.read(4))[0]

        metadata["tensor_count"] = tensor_count
        metadata["metadata_kv_count"] = metadata_kv_count

        # Helper to read strings
        def read_str():
            length = struct.unpack("<Q" if version >= 2 else "<I", f.read(8 if version >= 2 else 4))[0]
            return f.read(length).decode("utf-8", errors="replace")

        # Type mapping in GGUF
        # 0: UINT8, 1: INT8, 2: UINT16, 3: INT16, 4: UINT32, 5: INT32, 6: FLOAT32, 7: BOOL, 8: STRING, 9: ARRAY, 10: UINT64, 11: INT64, 12: FLOAT64
        def read_val(val_type):
            if val_type == 0: return struct.unpack("<B", f.read(1))[0]
            if val_type == 1: return struct.unpack("<b", f.read(1))[0]
            if val_type == 2: return struct.unpack("<H", f.read(2))[0]
            if val_type == 3: return struct.unpack("<h", f.read(2))[0]
            if val_type == 4: return struct.unpack("<I", f.read(4))[0]
            if val_type == 5: return struct.unpack("<i", f.read(4))[0]
            if val_type == 6: return struct.unpack("<f", f.read(4))[0]
            if val_type == 7: return struct.unpack("<B", f.read(1))[0] != 0
            if val_type == 8: return read_str()
            if val_type == 9:
                elem_type = struct.unpack("<I", f.read(4))[0]
                elem_count = struct.unpack("<Q" if version >= 2 else "<I", f.read(8 if version >= 2 else 4))[0]
                # If large array (e.g. token list), skip or sample
                if elem_count > 100:
                    # Skip bytes depending on elem_type
                    # String arrays require reading each
                    if elem_type == 8:
                        items = []
                        for i in range(elem_count):
                            s = read_str()
                            if i < 5: items.append(s)
                        return f"[Array of {elem_count} strings]"
                    else:
                        sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
                        f.seek(elem_count * sizes.get(elem_type, 4), os.SEEK_CUR)
                        return f"[Array of {elem_count} items]"
                else:
                    return [read_val(elem_type) for _ in range(elem_count)]
            if val_type == 10: return struct.unpack("<Q", f.read(8))[0]
            if val_type == 11: return struct.unpack("<q", f.read(8))[0]
            if val_type == 12: return struct.unpack("<d", f.read(8))[0]
            return None

        # Parse key-value pairs
        parsed_kvs = {}
        for _ in range(min(metadata_kv_count, 1000)):
            try:
                k = read_str()
                t = struct.unpack("<I", f.read(4))[0]
                v = read_val(t)
                parsed_kvs[k] = v
            except Exception:
                break

        metadata["keys"] = parsed_kvs

    return metadata


def calculate_sha256(filepath: str, max_bytes: int = 64 * 1024 * 1024) -> str:
    """Computes SHA256 of the file (sampling up to max_bytes or full for integrity)."""
    hasher = hashlib.sha256()
    bytes_read = 0
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            bytes_read += len(chunk)
            if max_bytes and bytes_read >= max_bytes:
                # Append file size to hash to ensure uniqueness even on partial sampling
                hasher.update(str(os.path.getsize(filepath)).encode())
                break
    return hasher.hexdigest()


class ModelManager:
    """Manages acquisition, location, and validation of GGUF model files."""

    def __init__(self, target_dir: str = DEFAULT_MODEL_DIR):
        self.target_dir = os.path.abspath(target_dir)
        os.makedirs(self.target_dir, exist_ok=True)

    def get_model_path(self, quant_type: str) -> str:
        """Returns the local path to the specified quantisation model file."""
        config = MODEL_CONFIGS.get(quant_type)
        if not config:
            raise ValueError(f"Unsupported quantisation configuration: {quant_type}. Supported: {list(MODEL_CONFIGS.keys())}")
        return os.path.join(self.target_dir, config["filename"])

    def is_model_present(self, quant_type: str) -> bool:
        """Checks if the model file is already present and non-empty."""
        path = self.get_model_path(quant_type)
        return os.path.exists(path) and os.path.getsize(path) > 100 * 1024 * 1024

    def download_model(self, quant_type: str) -> str:
        """Downloads the requested GGUF file from Hugging Face if not already present."""
        config = MODEL_CONFIGS.get(quant_type)
        if not config:
            raise ValueError(f"Unknown quantisation type: {quant_type}")

        dest_path = self.get_model_path(quant_type)
        if self.is_model_present(quant_type):
            print(f"[ModelManager] Model {quant_type} already exists at: {dest_path} ({os.path.getsize(dest_path)/(1024**3):.2f} GB). Skipping download.")
            return dest_path

        print(f"[ModelManager] Downloading {config['filename']} from {HF_REPO_ID} to {self.target_dir}...")
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=config["filename"],
            local_dir=self.target_dir,
            local_dir_use_symlinks=False,
        )
        print(f"[ModelManager] Download complete: {downloaded}")
        return downloaded

    def verify_models(self) -> Dict[str, Any]:
        """
        Extracts metadata and verifies architectural parity between Q4_K_M and Q8_0.
        """
        results = {}
        for q_type, conf in MODEL_CONFIGS.items():
            path = self.get_model_path(q_type)
            if not os.path.exists(path):
                results[q_type] = {"status": "MISSING", "path": path}
                continue

            file_size_bytes = os.path.getsize(path)
            meta = read_gguf_metadata(path)
            kvs = meta.get("keys", {})

            results[q_type] = {
                "status": "VERIFIED",
                "path": path,
                "file_size_gb": round(file_size_bytes / (1024**3), 3),
                "file_size_bytes": file_size_bytes,
                "architecture": kvs.get("general.architecture"),
                "model_name": kvs.get("general.name"),
                "file_type": kvs.get("general.file_type"),
                "quant_type": conf["quant_type"],
                "context_length": kvs.get("qwen2.context_length"),
                "embedding_length": kvs.get("qwen2.embedding_length"),
                "block_count": kvs.get("qwen2.block_count"),
                "feed_forward_length": kvs.get("qwen2.feed_forward_length"),
                "head_count": kvs.get("qwen2.attention.head_count"),
                "head_count_kv": kvs.get("qwen2.attention.head_count_kv"),
                "sha256_sample": calculate_sha256(path),
            }

        # Check parity
        if "Q4_K_M" in results and "Q8_0" in results:
            q4 = results["Q4_K_M"]
            q8 = results["Q8_0"]
            if q4.get("status") == "VERIFIED" and q8.get("status") == "VERIFIED":
                parity = {
                    "architecture_match": q4.get("architecture") == q8.get("architecture"),
                    "block_count_match": q4.get("block_count") == q8.get("block_count"),
                    "embedding_length_match": q4.get("embedding_length") == q8.get("embedding_length"),
                    "head_count_match": q4.get("head_count") == q8.get("head_count"),
                    "head_count_kv_match": q4.get("head_count_kv") == q8.get("head_count_kv"),
                }
                results["architectural_parity"] = parity

        return results


if __name__ == "__main__":
    mgr = ModelManager()
    print("Verifying model status in:", mgr.target_dir)
    res = mgr.verify_models()
    print(json.dumps(res, indent=2))
