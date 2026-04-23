from setuptools import setup, find_packages

found_packages = find_packages(include=['src', 'src.*'])
packages = ["my_vora_omni." + p for p in found_packages]

setup(
    name="my_vora_omni",
    version="0.1.7", 
    packages=packages,
    package_dir={"my_vora_omni.src": "src"},
    
    install_requires=[
        "torch",
        "transformers==5.5.0", 
        "pillow",
        "modelscope",
        "json_repair",
        "pandas",
        "datasets",
        "accelerate",
        "peft>=0.12.0",         # PeftMixedModel 이슈 해결을 위한 버전업
        "torchcodec",
        "torchvision",
        "ms-swift>=4.0.2", 
    ],
)