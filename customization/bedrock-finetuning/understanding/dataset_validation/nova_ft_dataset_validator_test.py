# tests/test_s3location_validation.py
import pytest
from pydantic import ValidationError

# Adjust import if your module has a different name/path
from nova_ft_dataset_validator import ConverseDatasetSample

def _img(s3_uri: str, bucket_owner: str | None):
    s3 = {"uri": s3_uri}
    if bucket_owner is not None:
        s3["bucketOwner"] = bucket_owner
    return {
        "image": {
            "format": "png",
            "source": {"s3Location": s3},
        }
    }

def _base_sample(contents):
    """
    Build a minimal, valid SFT-style sample:
    - user message with images
    - assistant message with plain text (no media)
    """
    return {
        "schemaVersion": "bedrock-conversation-2024",
        "system": [{"text": "System Prompt"}],
        "messages": [
            {"role": "user", "content": contents},
            {"role": "assistant", "content": [{"text": "ok"}]},
        ],
    }

def test_pro_sft_all_have_bucket_owner_passes():
    sample = _base_sample([
        _img("s3://bucket/path/img1.png", "738020477083"),
        _img("s3://bucket/path/img2.png", "738020477083"),
        _img("s3://bucket/path/img3.png", "738020477083"),
    ])
    # Should NOT raise
    ConverseDatasetSample.model_validate(
        sample,
        context={"model_name": "pro", "task_type": "sft"},
    )

def test_pro_sft_missing_bucket_owner_raises():
    # First image missing bucketOwner
    sample = _base_sample([
        _img("s3://bucket/path/img1.png", None),
        _img("s3://bucket/path/img2.png", "738020477083"),
    ])
    with pytest.raises(ValidationError):
        ConverseDatasetSample.model_validate(
            sample,
            context={"model_name": "pro", "task_type": "sft"},
        )

def test_pro_sft_empty_bucket_owner_raises():
    # First image has empty string (falsy)
    sample = _base_sample([
        _img("s3://bucket/path/img1.png", ""),
        _img("s3://bucket/path/img2.png", "738020477083"),
    ])
    with pytest.raises(ValidationError):
        ConverseDatasetSample.model_validate(
            sample,
            context={"model_name": "pro", "task_type": "sft"},
        )

def test_micro_sft_missing_bucket_owner_allowed():
    # Missing bucketOwner should be allowed when model != pro
    sample = _base_sample([
        _img("s3://bucket/path/img1.png", None),
        _img("s3://bucket/path/img2.png", None),
    ])
    ConverseDatasetSample.model_validate(
        sample,
        context={"model_name": "micro", "task_type": "sft"},
    )

def test_pro_dpo_missing_bucket_owner_allowed():
    # Missing bucketOwner should be allowed when task_type != sft
    sample = _base_sample([
        _img("s3://bucket/path/img1.png", None),
        _img("s3://bucket/path/img2.png", "738020477083"),
    ])
    ConverseDatasetSample.model_validate(
        sample,
        context={"model_name": "pro", "task_type": "dpo"},
    )
