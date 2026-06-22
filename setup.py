from setuptools import find_packages, setup


setup(
    name="utopic",
    version="0.1.5",
    description="Python package manager for the Utopic native runtime",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    package_data={"utopic": ["native/*", "models.json", "node/*"]},
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "utopic=utopic.cli:main",
            "utopic-server=utopic.server:main",
            "utopic-mcp=utopic.mcp:main",
            "utopic-acp=utopic.acp:main",
        ],
    },
)
