from setuptools import setup, find_packages

setup(
    use_scm_version={
        "version_scheme": "post-release",
        "write_to": "sism/_version.py",
    },
    setup_requires=["setuptools_scm"],
    packages=find_packages(),
)