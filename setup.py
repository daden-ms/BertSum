from setuptools import setup, find_packages
setup(
    name='BertSum',
    version='0.1',
    author='Yang Liu',
    author_email='https://github.com/daden-ms/BertSum',
    url='https://github.com/daden-ms/BertSum', 
    description='Fine-tune BERT for Extractive Summarization(https://arxiv.org/pdf/1903.10318.pdf)',
    packages=find_packages(exclude=("bert_data","json_data", "logs","models", "raw_data", "results", "urls")),
)

