# tests/test_s3location_validation.py
import pytest
from pydantic import ValidationError
from nova_ft_dataset_validator import ConverseDatasetSample

def _img_block(s3_uri: str, bucket_owner: str | None) -> dict:
    """Create an image block object"""
    s3 = {"uri": s3_uri}
    if bucket_owner is not None:
        s3["bucketOwner"] = bucket_owner
    return {
        "image": {
            "format": "png",
            "source": {"s3Location": s3},
        }
    }

def _base_bedrock_conversation_schema(contents) -> dict:
    """Create a bedrock conversaion schema object """
    return {
        "schemaVersion": "bedrock-conversation-2024",
        "system": [{"text": "System Prompt"}],
        "messages": [
            {"role": "user", "content": contents},
            {"role": "assistant", "content": [{"text": "ok"}]},
        ],
    }

@pytest.mark.parametrize(
    "model_name, task_type, bucket_owners, should_raise",
    [
        # included model, included task, all images have valid bucketOwner --> pass
        ("pro", "sft", ["738020477083", "738020477083", "738020477083"], False),

        # included model, included task, first image missing bucketOwner --> fail
        ("pro", "sft", [None, "738020477083"], True),

        # included model, included task, first image empty string bucketOwner --> fail
        ("pro", "sft", ["", "738020477083"], True),

        # model != pro — bucketOwner not required --> pass
        ("micro", "sft", [None, None], False),

        # task != sft — bucketOwner not required --> pass
        ("pro", "dpo", [None, "738020477083"], False),
    ]
)
def test_s3location_bucket_owner_enforcement(model_name, task_type, bucket_owners, should_raise):
    # generate image blocks for each sample
    contents = [_img_block(f"s3://bucket/img{i}.png", bucket_owner) for i, bucket_owner in enumerate(bucket_owners)] 
    sample = _base_bedrock_conversation_schema(contents)

    if should_raise: # validate model_validate raises --> invalid input
        with pytest.raises(ValidationError):
            ConverseDatasetSample.model_validate(
                sample,
                context={"model_name": model_name, "task_type": task_type},
            )
    else:  # validate model_validate does not raise --> valid input
        ConverseDatasetSample.model_validate(
            sample,
            context={"model_name": model_name, "task_type": task_type},
        )
