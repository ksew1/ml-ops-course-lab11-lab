import boto3
import os
from settings import Settings


def download_directory_from_s3(bucket_name, remote_directory_name, local_directory):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=remote_directory_name):
        if not os.path.exists(os.path.dirname(local_directory)):
            os.makedirs(os.path.dirname(local_directory))

        rel_path = os.path.relpath(obj.key, remote_directory_name)
        local_file_path = os.path.join(local_directory, rel_path)

        if not os.path.exists(os.path.dirname(local_file_path)):
            os.makedirs(os.path.dirname(local_file_path))

        if obj.key.endswith("/"):
            continue

        print(f"Downloading {obj.key} to {local_file_path}")
        bucket.download_file(obj.key, local_file_path)


def main():
    settings = Settings()
    settings.make_dirs()

    s3_client = boto3.client("s3")

    print(f"Downloading artifacts from {settings.s3_bucket}...")

    print(f"Downloading {settings.s3_classifier_key}...")
    s3_client.download_file(
        settings.s3_bucket, settings.s3_classifier_key, settings.classifier_joblib_path
    )

    download_directory_from_s3(
        settings.s3_bucket, settings.s3_model_prefix, settings.sentence_transformer_dir
    )

    print("Download complete.")


if __name__ == "__main__":
    main()
