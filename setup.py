from setuptools import setup, find_packages
setup(
    name='BertSum',
    version='0.1',
    packages=find_packages(exclude=("bert_data","json_data", "logs","models", "raw_data", "results", "urls")),
    description='Fine-tune BERT for Extractive Summarization(https://arxiv.org/pdf/1903.10318.pdf)',
)

