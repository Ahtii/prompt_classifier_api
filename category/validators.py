from rest_framework import serializers
from rest_framework import status


def validate_prompt(value):
    if value:
        return value.strip()
    raise serializers.ValidationError(
        detail={"message": "Prompt cannot be empty string."},
        code=status.HTTP_400_BAD_REQUEST
    )
