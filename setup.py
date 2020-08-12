from setuptools import setup, find_packages

# include readme info
with open('README.md', 'r') as myfile:
	long_description = myfile.read()

setup(name='ctseg',
      version='0.0.1',
      description='Lung and infection segmentation in CT scans from COVID-19 patients',
      long_description=long_description,
      url='https://github.com/Mahmood-Hoseini/COVID19-CT-Scan-Segmentation',
      author='Mahmood Hoseini',
      author_email='Mahmood.HoseiniF@gmail.com',
      license='MIT',
      packages=['ctseg', 'ctseg.models'])
