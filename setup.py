from setuptools import setup, find_packages
from pathlib import Path

# Get the long description from the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name='ultragpt',
    version='7.0.0',
    license="MIT",
    author='Ranit Bhowmick',
    author_email='mail@ranitbhowmick.com',
    description='UltraGPT: A modular multi-provider AI library for advanced reasoning and step pipelines with OpenAI and Claude support',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages('src', include=["ultragpt", "ultragpt.*"]),
    package_dir={'': 'src'},
    url='https://github.com/Kawai-Senpai/UltraGPT',
    project_urls={
        'Bug Reports': 'https://github.com/Kawai-Senpai/UltraGPT/issues',
        'Source': 'https://github.com/Kawai-Senpai/UltraGPT',
        'Documentation': 'https://github.com/Kawai-Senpai/UltraGPT/tree/main/docs',
    },
    install_requires=[
        'pydantic>=2.10.4',
        'langchain-core>=1.0.1',
        'langchain-openai>=1.0.1',
        'langchain-anthropic>=1.0.0',
        'ultraprint>=3.5.0',
        'google-api-python-client>=2.0.0',
        'requests>=2.25.0',
        'beautifulsoup4>=4.9.0',
        'readability-lxml>=0.8.0',
        'lxml>=4.6.0'
    ],
    extras_require={
        'env': ['python-dotenv>=0.19.0'],
        'dev': ['pytest>=6.0', 'pytest-cov>=2.0'],
        'all': ['python-dotenv>=0.19.0']
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='ai, gpt, openai, claude, anthropic, reasoning, pipeline, tools, multi-provider',
    python_requires='>=3.6',
)
